"""
End-to-end check: Wan2.1-T2V-1.3B (diffusers) with its attention swapped, on AMD GPUs.

    # 1. encode the prompts once (UMT5-XXL, ~22 GB fp32 -- run where it fits) and save them
    python example/rocm/wan_e2e.py --save-embeds embeds.pt
    # 2. denoise + decode with each attention implementation (same seed), then compare
    python example/rocm/wan_e2e.py --embeds embeds.pt --attn sdpa      --out runs/sdpa
    python example/rocm/wan_e2e.py --embeds embeds.pt --attn sage      --out runs/sage
    python example/rocm/wan_e2e.py --compare runs/sdpa runs/sage

--attn:
  sdpa            torch.nn.functional.scaled_dot_product_attention (diffusers' default "native" backend)
  sage-diffusers  diffusers' own "sage" backend: transformer.set_attention_backend("sage") -> sageattention.sageattn
  sage            sageattention.sageattn called directly (same kernels; lets us pick P.V precision)
  sage-fp8        sageattn(..., pv_dtype="fp8")   (CDNA3 only)
  triton-fa       nunchaku-rocm's FP16 Triton flash attention (bench/rocm/nunchaku_triton_attn.py)
  aiter           aiter.flash_attn_func (CK / asm FlashAttention, MI300X)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "bench", "rocm"))

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
PROMPT = (
    "A red fox trotting through a snowy birch forest at sunrise, soft golden light, "
    "breath visible in the cold air, cinematic, highly detailed"
)
NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards"
)


def save_embeds(path):
    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(MODEL, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to("cuda")
    with torch.no_grad():
        pe, ne = pipe.encode_prompt(PROMPT, NEGATIVE, do_classifier_free_guidance=True, device="cuda")
    torch.save({"prompt_embeds": pe.cpu(), "negative_prompt_embeds": ne.cpu(), "prompt": PROMPT}, path)
    print("saved", path, tuple(pe.shape))


class AttnSwap:
    """Replaces diffusers' dispatch_attention_fn inside the Wan transformer and times every call."""

    def __init__(self, kind):
        import diffusers.models.transformers.transformer_wan as tw

        self.tw = tw
        self.orig = tw.dispatch_attention_fn
        self.kind = kind
        self.calls = 0
        tw.dispatch_attention_fn = self

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, backend=None, **kw):
        # query/key/value: [B, N, H, D] (NHD)
        self.calls += 1
        if self.kind == "sdpa" or attn_mask is not None:
            return self.orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                             backend=backend, **kw)  # fmt: skip
        if self.kind in ("sage", "sage-fp8"):
            from sageattention import sageattn

            pv = "fp8" if self.kind == "sage-fp8" else "fp16"
            return sageattn(query, key, value, tensor_layout="NHD", is_causal=is_causal, pv_dtype=pv)
        if self.kind == "aiter":
            import aiter

            return aiter.flash_attn_func(query, key, value, causal=is_causal)
        if self.kind == "triton-fa":
            from nunchaku_triton_attn import attention_fp16

            B, N, H, D = query.shape
            o = torch.empty(B, N, H * D, device=query.device, dtype=query.dtype)
            attention_fp16(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), o, D**-0.5)
            return o.view(B, N, H, D)
        raise ValueError(self.kind)


def run(args):
    # only the transformer, VAE and scheduler are needed here (the prompts come pre-encoded): never let a missing
    # text encoder (22 GB) be fetched
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from diffusers import AutoencoderKLWan, WanPipeline

    emb = torch.load(args.embeds)
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL, vae=vae, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if args.attn == "sage-diffusers":
        pipe.transformer.set_attention_backend("sage")
        swap = None
    else:
        swap = AttnSwap(args.attn)

    pe = emb["prompt_embeds"].to("cuda", torch.bfloat16)
    ne = emb["negative_prompt_embeds"].to("cuda", torch.bfloat16)
    step_times = []

    def cb(p, i, t, kw):
        torch.cuda.synchronize()
        step_times.append(time.perf_counter())
        if args.verbose:
            print(f"  step {i} {step_times[-1] - (step_times[-2] if len(step_times) > 1 else t_start[0]):.2f}s", flush=True)
        return kw

    t_start = [time.perf_counter()]

    def denoise(steps):
        g = torch.Generator(device="cuda").manual_seed(args.seed)
        return pipe(
            prompt_embeds=pe,
            negative_prompt_embeds=ne,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            num_inference_steps=steps,
            guidance_scale=5.0,
            generator=g,
            output_type="latent",
            callback_on_step_end=cb,
        ).frames

    # warm-up (kernel compilation / autotuning) on a 2-step run
    tw = time.perf_counter()
    denoise(2)
    torch.cuda.synchronize()
    print(f"warm-up {time.perf_counter() - tw:.1f}s", flush=True)
    t_start[0] = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    step_times.clear()
    t0 = time.perf_counter()
    latents = denoise(args.steps)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    peak_denoise = torch.cuda.max_memory_allocated() / 2**30

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "latents.npy"), latents.float().cpu().numpy())
    info = dict(
        attn=args.attn,
        gpu=torch.cuda.get_device_name(),
        frames=args.frames,
        height=args.height,
        width=args.width,
        steps=args.steps,
        denoise_s=t1 - t0,
        step_s_median=float(np.median(np.diff([t0] + step_times))),
        peak_gib_denoise=peak_denoise,
        attn_calls=swap.calls if swap else None,
    )
    if args.no_decode:
        json.dump(info, open(os.path.join(args.out, "info.json"), "w"), indent=1)
        print(json.dumps(info))
        return
    decode(vae, pipe.video_processor, latents, args.out, args.vae_tiling)
    torch.cuda.synchronize()
    info["decode_s"] = time.perf_counter() - t1
    info["peak_gib_total"] = torch.cuda.max_memory_allocated() / 2**30
    json.dump(info, open(os.path.join(args.out, "info.json"), "w"), indent=1)
    print(json.dumps(info))


def decode(vae, video_processor, latents, out, tiling=False):
    with torch.no_grad():
        lat = latents.to(device="cuda", dtype=vae.dtype)
        mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(lat)
        std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(lat)
        if tiling:
            vae.enable_tiling()
        video = vae.decode(lat / std + mean, return_dict=False)[0]
        video = video_processor.postprocess_video(video, output_type="np")[0]  # [F, H, W, 3] in [0, 1]
    frames = (video * 255).round().astype(np.uint8)
    np.save(os.path.join(out, "frames.npy"), frames)
    try:
        from PIL import Image

        for i in sorted({0, len(frames) // 2, len(frames) - 1}):
            Image.fromarray(frames[i]).save(os.path.join(out, f"frame{i:03d}.png"))
    except ImportError:
        pass


def decode_only(run_dir):
    """Decode a --no-decode run's latents (e.g. on another GPU) so --compare can score the videos."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from diffusers import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor

    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).to("cuda")
    vp = VideoProcessor(vae_scale_factor=vae.config.scale_factor_spatial)
    latents = torch.from_numpy(np.load(os.path.join(run_dir, "latents.npy")))
    decode(vae, vp, latents, run_dir)
    print("decoded", run_dir)


def compare(a, b):
    fa, fb = np.load(os.path.join(a, "frames.npy")).astype(np.float64), np.load(os.path.join(b, "frames.npy")).astype(
        np.float64
    )
    la, lb = np.load(os.path.join(a, "latents.npy")), np.load(os.path.join(b, "latents.npy"))
    mse = ((fa - fb) ** 2).mean()
    psnr = 10 * np.log10(255.0**2 / max(mse, 1e-12))
    per_frame = [10 * np.log10(255.0**2 / max(((x - y) ** 2).mean(), 1e-12)) for x, y in zip(fa, fb)]
    rel_l2 = np.linalg.norm(la - lb) / np.linalg.norm(la)
    cos = float((la * lb).sum() / (np.linalg.norm(la) * np.linalg.norm(lb)))
    ia, ib = json.load(open(os.path.join(a, "info.json"))), json.load(open(os.path.join(b, "info.json")))
    print(f"{a} vs {b}: video PSNR {psnr:.2f} dB (per-frame min {min(per_frame):.2f}), latent rel-L2 {rel_l2:.4f}, "
          f"latent cos {cos:.5f}")  # fmt: skip
    print(f"  denoise {ia['denoise_s']:.1f}s -> {ib['denoise_s']:.1f}s ({ia['denoise_s'] / ib['denoise_s']:.2f}x), "
          f"step {ia['step_s_median']:.2f}s -> {ib['step_s_median']:.2f}s, "
          f"peak (denoise) {ia['peak_gib_denoise']:.1f} / {ib['peak_gib_denoise']:.1f} GiB")  # fmt: skip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save-embeds")
    p.add_argument("--embeds")
    p.add_argument("--attn", default="sage")
    p.add_argument("--out", default="runs/x")
    p.add_argument("--frames", type=int, default=33)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vae-tiling", action="store_true")
    p.add_argument("--compare", nargs=2)
    p.add_argument("--no-decode", action="store_true", help="save latents only (decode later with --decode-only)")
    p.add_argument("--decode-only", help="decode <run dir>/latents.npy")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.decode_only:
        decode_only(args.decode_only)
    elif args.save_embeds:
        save_embeds(args.save_embeds)
    elif args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
