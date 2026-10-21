"""
A/B-test ComfyUI's attention backends on an API-format workflow: same seeds, timings, output similarity.

    python example/rocm/comfyui_ab.py --comfy /path/to/ComfyUI \
        --workflow example/rocm/comfyui/wan2.1_t2v_1.3B.json --modes default,sage --runs 3 --out runs/comfy_ab

For every mode a fresh ComfyUI server (``main.py``) is started on a free local port with that mode's
launch flags, the workflow is submitted ``--runs`` times through the HTTP API (seed ``--seed + i`` for run
i, identical across modes, so run i of every mode is directly comparable), and the server is stopped.
Run 1 includes model loading, Triton autotuning and kernel compilation; the summary reports the
median of the later runs.

Modes:
  default   ComfyUI as launched without attention flags (on ROCm usually PyTorch SDPA; see the log line)
  aotriton  default + TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 (PyTorch's flash / mem-efficient SDPA kernels
            on GPUs where ROCm PyTorch marks them experimental, e.g. RDNA3)
  sage      ``--use-sage-attention`` (ComfyUI calls ``sageattention.sageattn``)
  kitchen   no flag, a "Model Attention Backend" node set to "comfy kitchen attention" (ComfyUI's own
            INT8 attention) inserted after the diffusion-model loader, if this ComfyUI has it

Per mode and run it records the wall time of the prompt ("Prompt executed in ..."), the sampler's
time per step (from ComfyUI's websocket progress events: mean over steps 2..N), the peak device memory in use
(``/system_stats`` polled every 0.25 s; device-wide, includes other processes) and saves every output
image/frame losslessly (save nodes are replaced by PNG ``SaveImage``). Outputs of each mode are compared
with the first mode's (PSNR over all frames, per run). Extra ComfyUI flags (``--comfy-arg=--highvram``)
apply to every mode. Results: ``<out>/summary.json`` plus each server's log in ``<out>/<mode>/server.log``.
Needs only the standard library, numpy and Pillow (both ComfyUI requirements).
"""

import argparse
import base64
import copy
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

import numpy as np
from PIL import Image

MODES = {
    "default": {"args": [], "attention_node": None},
    "aotriton": {"args": [], "attention_node": None, "env": {"TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL": "1"}},
    "sage": {"args": ["--use-sage-attention"], "attention_node": None},
    "kitchen": {"args": [], "attention_node": "comfy kitchen attention"},
}
SAVE_NODES = {"SaveImage", "PreviewImage", "SaveAnimatedWEBP", "SaveAnimatedPNG", "SaveWEBM", "SaveVideo"}
MODEL_LOADERS = {"UNETLoader", "CheckpointLoaderSimple", "UnetLoaderGGUF"}
SAMPLERS = {"KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comfy", required=True, help="ComfyUI directory (the one containing main.py)")
    p.add_argument("--workflow", required=True, help="API-format workflow JSON")
    p.add_argument("--modes", default="default,sage", help="comma-separated, from: " + ", ".join(MODES))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, help="override the sampler's step count")
    p.add_argument("--out", default="runs/comfy_ab")
    p.add_argument("--extra-model-paths", help="passed as --extra-model-paths-config")
    p.add_argument("--comfy-arg", action="append", default=[], help="extra main.py argument (repeatable)")
    p.add_argument("--python", default=sys.executable, help="interpreter for ComfyUI (default: this one)")
    p.add_argument("--timeout", type=float, default=3600, help="seconds per run")
    return p.parse_args()


class Server:
    def __init__(self, args, mode_args, log_path, env=None):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        cmd = [args.python, "-u", os.path.join(args.comfy, "main.py"), "--listen", "127.0.0.1",
               "--port", str(self.port), "--disable-auto-launch", *args.comfy_arg, *mode_args]  # fmt: skip
        if args.extra_model_paths:
            cmd += ["--extra-model-paths-config", os.path.abspath(args.extra_model_paths)]
        self.log_path = log_path
        self.log = open(log_path, "w", encoding="utf-8", errors="replace")
        print("  $", " ".join(cmd), flush=True)
        self.proc = subprocess.Popen(cmd, cwd=args.comfy, stdout=self.log, stderr=subprocess.STDOUT,
                                     env={**os.environ, **(env or {})})  # fmt: skip
        t0 = time.time()
        while True:
            try:
                self.call("/system_stats")
                break
            except OSError:
                if self.proc.poll() is not None or time.time() - t0 > 600:
                    raise SystemExit(f"ComfyUI did not start; see {log_path}")
                time.sleep(1)
        print(f"  ComfyUI up in {time.time() - t0:.1f} s (port {self.port})", flush=True)

    def call(self, path, payload=None):
        req = urllib.request.Request(self.url + path, headers={"Content-Type": "application/json"},
                                     data=None if payload is None else json.dumps(payload).encode())  # fmt: skip
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())

    def log_text(self):
        self.log.flush()
        with open(self.log_path, encoding="utf-8", errors="replace") as f:
            return re.sub(r"\x1b\[[0-9;]*m", "", f.read())

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


class ProgressListener:
    """Minimal stdlib WebSocket client for ComfyUI's /ws: timestamps of the sampler's per-step progress events."""

    def __init__(self, port, client_id):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall((f"GET /ws?clientId={client_id} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                           f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                           "Sec-WebSocket-Version: 13\r\n\r\n").encode())  # fmt: skip
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.sock.recv(4096)
        if b" 101 " not in buf.split(b"\r\n", 1)[0]:
            raise OSError("websocket upgrade failed")
        self.buf = buf.split(b"\r\n\r\n", 1)[1]
        self.sock.settimeout(0.5)
        self.events, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _read(self, n):
        while len(self.buf) < n:
            if self.stop.is_set():
                raise EOFError
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                raise EOFError
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def _send(self, opcode, payload=b""):
        mask = os.urandom(4)
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(bytes([0x80 | opcode, 0x80 | len(payload)]) + mask + data)

    def _loop(self):
        try:
            while not self.stop.is_set():
                b0, b1 = self._read(2)
                n = b1 & 0x7F
                if n == 126:
                    n = int.from_bytes(self._read(2), "big")
                elif n == 127:
                    n = int.from_bytes(self._read(8), "big")
                payload = self._read(n)
                op = b0 & 0x0F
                if op == 0x9:  # ping
                    self._send(0xA, payload)
                elif op == 0x8:
                    break
                elif op == 0x1:
                    msg = json.loads(payload)
                    if msg.get("type") == "progress":
                        d = msg["data"]
                        self.events.append((time.perf_counter(), d.get("prompt_id"), d.get("node"), d["value"], d["max"]))
        except (EOFError, OSError, ValueError):
            pass

    def step_time(self, prompt_id):
        """Mean seconds per step of the longest progress sequence of this prompt, excluding its first step."""
        seqs = {}
        for t, pid, node, value, mx in self.events:
            if pid == prompt_id and mx > 1:
                seqs.setdefault((node, mx), []).append((value, t))
        best = None
        for (node, mx), ev in seqs.items():
            ev = sorted(dict(ev).items())  # last timestamp per step value
            if len(ev) >= 3 and (best is None or mx > best[0]):
                best = (mx, (ev[-1][1] - ev[0][1]) / (ev[-1][0] - ev[0][0]))
        return best

    def close(self):
        self.stop.set()
        self.thread.join(5)
        self.sock.close()


def vram_used(server):
    devs = server.call("/system_stats").get("devices", [])
    return max((d["vram_total"] - d["vram_free"] for d in devs), default=0)


def prepare_workflow(wf, mode, seed, steps, prefix):
    wf = copy.deepcopy(wf)
    for nid, node in list(wf.items()):
        ct, inp = node["class_type"], node["inputs"]
        if ct in SAMPLERS:
            for key in ("seed", "noise_seed"):
                if key in inp:
                    inp[key] = seed
            if steps is not None and "steps" in inp:
                inp["steps"] = steps
        if ct in ("RandomNoise",) and "noise_seed" in inp:
            inp["noise_seed"] = seed
        if ct in SAVE_NODES:
            wf[nid] = {"class_type": "SaveImage", "inputs": {"images": inp["images"], "filename_prefix": prefix}}
    if MODES[mode]["attention_node"]:
        loaders = [nid for nid, n in wf.items() if n["class_type"] in MODEL_LOADERS]
        if not loaders:
            raise SystemExit("kitchen mode: no diffusion-model loader node found")
        for lid in loaders:
            new = f"ab_attn_{lid}"
            wf[new] = {"class_type": "ModelAttentionBackend",
                       "inputs": {"model": [lid, 0], "attention": MODES[mode]["attention_node"]}}  # fmt: skip
            for nid, n in wf.items():
                if nid == new:
                    continue
                for k, v in n["inputs"].items():
                    if isinstance(v, list) and len(v) == 2 and v[0] == lid and v[1] == 0:
                        n["inputs"][k] = [new, 0]
    return wf


def run_once(server, wf, timeout):
    client_id = uuid.uuid4().hex
    listener = ProgressListener(server.port, client_id)
    pid = server.call("/prompt", {"prompt": wf, "client_id": client_id})["prompt_id"]
    peak, t0 = [vram_used(server)], time.time()
    done = threading.Event()

    def poll():
        while not done.is_set():
            try:
                peak[0] = max(peak[0], vram_used(server))
            except OSError:
                pass
            time.sleep(0.25)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    try:
        while True:
            h = server.call(f"/history/{pid}")
            if pid in h:
                break
            if server.proc.poll() is not None:
                raise SystemExit(f"ComfyUI exited; see {server.log_path}")
            if time.time() - t0 > timeout:
                raise SystemExit("timed out")
            time.sleep(0.5)
    finally:
        done.set()
        th.join()
        listener.close()
    entry = h[pid]
    if entry.get("status", {}).get("status_str") != "success":
        msgs = [m for m in entry.get("status", {}).get("messages", []) if m[0] == "execution_error"]
        raise SystemExit(f"workflow failed: {json.dumps(msgs)[:2000]}")
    images = [img for out in entry.get("outputs", {}).values() for img in out.get("images", [])]
    return time.time() - t0, peak[0], images, listener.step_time(pid)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main():
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m!r}; choose from {', '.join(MODES)}")
    with open(args.workflow) as f:
        wf = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    comfy_out = os.path.join(args.comfy, "output")
    summary = {"workflow": os.path.basename(args.workflow), "runs": args.runs, "modes": {}}
    tag = uuid.uuid4().hex[:6]

    for mode in modes:
        print(f"== mode {mode}", flush=True)
        mdir = os.path.join(args.out, mode)
        os.makedirs(mdir, exist_ok=True)
        env = MODES[mode].get("env", {})
        server = Server(args, MODES[mode]["args"], os.path.join(mdir, "server.log"), env)
        res = {"launch_args": MODES[mode]["args"], "env": env, "attention_node": MODES[mode]["attention_node"], "runs": []}
        try:
            text = server.log_text()
            res["attention_log"] = [ln.strip() for ln in text.splitlines() if re.search(r"(?i)using .*attention|sage attention", ln)]
            for line in res["attention_log"]:
                print("  log:", line, flush=True)
            for i in range(args.runs):
                prefix = f"ab_{tag}_{mode}_run{i + 1}"
                before = len(server.log_text())
                wall, peak, images, step = run_once(
                    server, prepare_workflow(wf, mode, args.seed + i, args.steps, prefix), args.timeout
                )
                chunk = server.log_text()[before:]
                m = re.findall(r"Prompt executed in ([\d.]+) seconds", chunk)
                # ComfyUI falls back to PyTorch attention (logging an error) if sageattn raises
                warn = [ln.strip() for ln in chunk.splitlines() if re.search(r"(?i)error running sage|attention backend .* unavailable", ln)]
                if warn:
                    print(f"  WARNING: {len(warn)} attention fallback(s), e.g. {warn[0]}", flush=True)
                rdir = os.path.join(mdir, f"run{i + 1}")
                os.makedirs(rdir, exist_ok=True)
                files = []
                for img in images:
                    src = os.path.join(comfy_out, img.get("subfolder", ""), img["filename"])
                    shutil.move(src, os.path.join(rdir, img["filename"]))
                    files.append(img["filename"])
                r = {"seed": args.seed + i, "wall_s": round(wall, 2),
                     "prompt_s": float(m[-1]) if m else None,
                     "steps": step[0] if step else None, "s_per_step": round(step[1], 4) if step else None,
                     "peak_vram_gib": round(peak / 2**30, 2), "attention_fallbacks": len(warn),
                     "files": sorted(files)}  # fmt: skip
                res["runs"].append(r)
                print(f"  run {i + 1}: prompt {r['prompt_s']} s, sampler {r['s_per_step']} s/step ({r['steps']} steps), "
                      f"peak VRAM {r['peak_vram_gib']} GiB, {len(files)} images", flush=True)  # fmt: skip
        finally:
            server.stop()
        warm = res["runs"][1:] or res["runs"]
        per_step = [r["s_per_step"] for r in warm if r["s_per_step"]]
        res["median_prompt_s"] = statistics.median(r["prompt_s"] for r in warm if r["prompt_s"] is not None)
        res["median_s_per_step"] = round(statistics.median(per_step), 3) if per_step else None
        summary["modes"][mode] = res

    ref = modes[0]
    for mode in modes:
        res = summary["modes"][mode]
        if mode != ref:
            vals = []
            for i, (r, rr) in enumerate(zip(res["runs"], summary["modes"][ref]["runs"])):
                load = lambda m, fs: np.stack([np.asarray(Image.open(os.path.join(args.out, m, f"run{i + 1}", f)).convert("RGB")) for f in fs])  # noqa: E731
                vals.append(round(float(psnr(load(ref, rr["files"]), load(mode, r["files"]))), 2))
            res[f"psnr_vs_{ref}_db"] = vals

    print("\n| mode | launch flags / node | sampler s/step | prompt time | speedup | peak VRAM | PSNR vs " + ref + " |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    base = summary["modes"][ref]
    for mode in modes:
        r = summary["modes"][mode]
        how = " ".join([f"{k}={v}" for k, v in r["env"].items()] + r["launch_args"])
        how = how or (f'node: {r["attention_node"]}' if r["attention_node"] else "(none)")
        if any(x["attention_fallbacks"] for x in r["runs"]):
            how += " (FELL BACK, see log)"
        sp = base["median_s_per_step"] / r["median_s_per_step"] if r["median_s_per_step"] and base["median_s_per_step"] else None
        peak = max(x["peak_vram_gib"] for x in r["runs"])
        ps = r.get(f"psnr_vs_{ref}_db", "reference")
        print(f"| {mode} | {how} | {r['median_s_per_step']} | {r['median_prompt_s']:.1f} s | "
              f"{f'{sp:.2f}x' if sp else '?'} | {peak} GiB | {ps} |")  # fmt: skip
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\nwrote", os.path.join(args.out, "summary.json"))


if __name__ == "__main__":
    main()
