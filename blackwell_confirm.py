#!/usr/bin/env python3
"""Blackwell (sm_120) llama.cpp prebuilt validation for Unsloth Studio PR #6494.

PR #6494 ("Studio macOS fixes") also dropped an obsolete Windows Blackwell CUDA
pin (the immutable b9360 cuda-13.1 build). Codex worried that removing it could
leave a Blackwell GPU on a CUDA 13.0-13.2 driver with no usable GPU prebuilt
(upstream ships only cuda-12.4, which predates Blackwell, and cuda-13.3, which a
13.0-13.2 driver is gated off). The claim is that the Unsloth llama.cpp fork --
which Studio installs FIRST -- already covers Blackwell with toolkit-12.8 builds,
making the pin obsolete.

This script settles it empirically on real Blackwell silicon. It drives the actual
PR installer (pure stdlib, downloaded from the PR head) -- it does NOT reimplement
anything.

Difference from windows_confirm.ps1 in this repo: that script hand-mirrors the
installer's OLD selection (it still carries the b9360 cuda-13.1 pin) and only tests
the UPSTREAM build. This one runs the real post-#6494 installer (pin removed) and
exercises the FORK-first path -- i.e. it tests exactly what the PR changed. Reports:

  PART 1  Host probe via the PR's own detect_host(): GPU, compute cap (confirm
          sm_120), CUDA driver version, OS.
  PART 2  The pin-removal worry, tested directly: resolve_upstream_asset_choice()
          = what the UPSTREAM-ONLY (fork-down) path would give THIS host. GPU
          build, CPU bundle, or hard fallback?
  PART 3  Real install via the PR installer (fork-first, the normal Studio path):
          which asset does it actually pick for this Blackwell GPU?
  PART 4  GPU inference smoke: run the installed llama-server with full offload
          (-ngl 99) on a tiny model, confirm it loaded layers on the GPU and
          answered "2+2=4".
  PART 5  (--full) Empirical upstream check: download upstream's cuda-12.4 and
          cuda-13.3 builds directly and GPU-test each on this driver (Windows).

Run:   python test_blackwell.py            # parts 1-4
       python test_blackwell.py --full     # also part 5
No third-party packages required. Needs internet + an NVIDIA GPU + nvidia-smi.
Paste the whole output back.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

PR_REF = os.environ.get("UNSLOTH_PR_REF", "d55834e05b0f79d6aa0e251fe9b511b01f39a4e8")
INSTALLER_URL = (
    f"https://raw.githubusercontent.com/unslothai/unsloth/{PR_REF}/studio/install_llama_prebuilt.py"
)
FORK_REPO = os.environ.get("FORK_REPO", "unslothai/llama.cpp")
UPSTREAM_REPO = "ggml-org/llama.cpp"
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
)
WORK = Path(os.environ.get("BLACKWELL_WORKDIR", "blackwell_test")).resolve()
PORT = int(os.environ.get("PORT", "18099"))
IS_WIN = os.name == "nt"
EXE = ".exe" if IS_WIN else ""


def hr(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72, flush=True)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [cached] {dest.name} ({dest.stat().st_size/1e6:.0f} MB)", flush=True)
        return dest
    print(f"  downloading {url}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "blackwell-test"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"  -> {dest.name} ({dest.stat().st_size/1e6:.0f} MB)", flush=True)
    return dest


def http_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "blackwell-test"})
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def load_installer() -> "object":
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / "install_llama_prebuilt.py"
    download(INSTALLER_URL, path)
    spec = importlib.util.spec_from_file_location("install_llama_prebuilt", path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's @dataclass definitions can resolve
    # cls.__module__ via sys.modules (Python 3.12+ raises AttributeError otherwise).
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, path


def find_binary(root: Path, stem: str) -> Path | None:
    name = stem + EXE
    cands = [
        root / "build" / "bin" / "Release" / name,
        root / "build" / "bin" / name,
        root / name,
    ]
    hit = next((c for c in cands if c.is_file()), None)
    if hit:
        return hit
    return next((p for p in root.rglob(name) if p.is_file()), None)


def extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)


def _gpu_mem_used_mib() -> int | None:
    """Total GPU memory in use across visible devices (MiB), or None."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=15)
        vals = [int(x) for x in r.stdout.split() if x.strip().isdigit()]
        return sum(vals) if vals else None
    except Exception:
        return None


def _gpu_mem_for_pid_mib(pid: int) -> int | None:
    """GPU memory (MiB) attributed to a specific pid via nvidia-smi, or None
    (None can mean 'not using GPU' OR 'per-process listing hidden', e.g. some
    containers -- the memory-delta check below is the robust fallback)."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        for line in r.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
                return int(parts[1]) if parts[1].isdigit() else None
    except Exception:
        return None
    return None


def gpu_smoke(server: Path, gguf: Path, label: str, port: int) -> dict:
    """Start llama-server with full GPU offload; confirm real GPU use three ways
    (log markers, per-pid GPU memory, and a GPU memory delta) plus a correct answer."""
    log = WORK / f"server_{label}.log"
    lf = open(log, "w")
    args = [str(server), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
            "-ngl", "99", "-c", "1024", "--no-warmup"]
    kw = dict(stdout=lf, stderr=subprocess.STDOUT)
    if not IS_WIN:
        kw["start_new_session"] = True
    mem0 = _gpu_mem_used_mib()
    proc = subprocess.Popen(args, env=dict(os.environ), **kw)

    def tail() -> str:
        try:
            return open(log, errors="replace").read()
        except OSError:
            return ""

    def kill():
        try:
            if IS_WIN:
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.terminate()

    healthy = False
    for i in range(150):
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
                if resp.status == 200:
                    healthy = True
                    break
        except Exception:
            pass
        time.sleep(2)

    # Ground-truth GPU signals while the model is resident.
    pid_mem = _gpu_mem_for_pid_mib(proc.pid) if healthy else None
    mem1 = _gpu_mem_used_mib() if healthy else None
    mem_delta = (mem1 - mem0) if (mem0 is not None and mem1 is not None) else None

    text = tail()
    # Broader log markers across llama.cpp logger variants.
    m = re.search(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers to GPU", text)
    layers_line = m.group(0) if m else ""
    cuda_dev = re.search(r"(ggml_cuda_init[^\n]*|Device \d+: [^\n]*|CUDA0[^\n]*(?:buffer|model)[^\n]*|using device CUDA[^\n]*)", text)
    device_line = cuda_dev.group(0).strip() if cuda_dev else ""
    log_says_gpu = bool(m and int(m.group(1)) > 0) or bool(re.search(r"ggml_cuda_init|CUDA0|offloading|using device CUDA", text))

    offloaded = bool(log_says_gpu or (pid_mem and pid_mem > 0) or (mem_delta is not None and mem_delta > 100))

    answer = ""
    if healthy:
        try:
            payload = json.dumps({
                "messages": [{"role": "user", "content": "What is 2+2? Reply with only the number."}],
                "temperature": 0, "max_tokens": 16,
            }).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
            answer = (body.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        except Exception as e:
            answer = f"(request failed: {e})"
    kill()

    answer_ok = bool(re.search(r"\b4\b|four", answer, re.IGNORECASE))
    lines = text.splitlines()
    return {
        "label": label, "started": proc.returncode != 0 or healthy, "healthy": healthy,
        "gpu_offloaded": offloaded,
        "gpu_layers": layers_line or "(no 'offloaded N/N' line; relying on GPU-memory check)",
        "device_line": device_line,
        "gpu_mem_mib": f"{mem0}->{mem1} MiB (delta {mem_delta})" if mem_delta is not None else "(nvidia-smi mem unavailable)",
        "pid_mem_mib": pid_mem,
        "answer": answer.strip()[:80], "answer_ok": answer_ok,
        "log_tail": "\n".join(lines[:35] + (["..."] if len(lines) > 50 else []) + lines[-15:]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="also run PART 5 (raw upstream build test)")
    ap.add_argument("--keep", action="store_true", help="keep the workdir (don't clean)")
    args = ap.parse_args()

    print(f"workdir: {WORK}")
    print(f"installer pinned at PR ref: {PR_REF}")
    M, installer_path = load_installer()

    # -------------------------------------------------------------- PART 1
    hr("PART 1 - HOST (via the PR installer's own detect_host())")
    host = M.detect_host()
    caps = host.compute_caps or []
    max_sm = max((int(c.replace(".", "")) for c in caps), default=None)
    print(f"OS / arch         : {host.system} / {host.machine}")
    print(f"nvidia-smi present: {bool(host.nvidia_smi)}")
    print(f"GPU compute_caps  : {caps}")
    print(f"max sm            : {max_sm}  (Blackwell sm_120 -> {max_sm == 120}; sm>=120 -> {bool(max_sm and max_sm >= 120)})")
    print(f"driver CUDA ver   : {host.driver_cuda_version}")
    print(f"has_usable_nvidia : {host.has_usable_nvidia}")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        print(f"nvidia-smi gpu    : {smi.stdout.strip()}")
    except Exception as e:
        print(f"nvidia-smi gpu    : (failed: {e})")

    is_blackwell = bool(max_sm and max_sm >= M._BLACKWELL_MIN_SM)
    if not is_blackwell:
        print("\nNOTE: this GPU is NOT sm_120+ (the code's Blackwell class). The pin-removal\n"
              "concern is specific to Blackwell; results below are still informative.")

    # -------------------------------------------------------------- PART 2
    hr("PART 2 - THE WORRY: upstream-only (fork-down) selection for THIS host")
    try:
        latest = http_json(f"https://api.github.com/repos/{UPSTREAM_REPO}/releases/latest")
        tag = latest["tag_name"]
    except Exception as e:
        tag = None
        print(f"could not fetch latest upstream tag ({e}); skipping PART 2")
    if tag:
        print(f"upstream latest tag: {tag}")
        try:
            # resolve_asset_choice is the fork-down/upstream entry: it raises
            # "fork required" for Linux+NVIDIA and runs the Windows Blackwell
            # filter + driver gate for Windows -- exactly the pin scenario.
            choice = M.resolve_asset_choice(host, tag)
            gpu = "cuda" in (choice.install_kind or "") or "cuda" in (choice.name or "").lower()
            print(f"upstream-only would install: {choice.name}")
            print(f"  install_kind={choice.install_kind!r}  -> {'GPU build' if gpu else 'CPU/other (NO GPU)'}")
            if not gpu and is_blackwell:
                print("  => CONFIRMS the worry: with the fork unavailable, this Blackwell host\n"
                      "     would fall to a non-GPU build on the upstream path.")
            elif gpu:
                print("  => upstream itself serves this host a GPU build (driver is 13.3-capable).")
        except M.PrebuiltFallback as e:
            print(f"upstream-only -> PrebuiltFallback: {e}")
            if host.is_linux:
                print("  (expected on Linux: upstream CUDA fallback is intentionally unavailable;\n"
                      "   Linux CUDA REQUIRES the fork's published bundle.)")
            elif is_blackwell:
                print("  => CONFIRMS the worry on the fork-down path (no usable upstream GPU build).")

    # -------------------------------------------------------------- PART 3
    hr("PART 3 - REAL INSTALL via the PR installer (fork-first, normal Studio path)")
    install_dir = WORK / "install"
    shutil.rmtree(install_dir, ignore_errors=True)
    cmd = [sys.executable, str(installer_path), "--install-dir", str(install_dir),
           "--published-repo", FORK_REPO]
    print("running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    for line in out.splitlines():
        if re.search(r"Installing prebuilt|selected|cuda|Blackwell|sm_120|install_kind|asset|GPU|fallback", line, re.I):
            print("  |", line)
    marker = install_dir / "UNSLOTH_PREBUILT_INFO.json"
    installed = None
    if marker.is_file():
        installed = json.load(open(marker))
        print(f"\ninstalled marker: {json.dumps(installed)}")
    print(f"installer exit code: {r.returncode}")

    # -------------------------------------------------------------- PART 4
    hr("PART 4 - GPU INFERENCE SMOKE on the installed (fork) build")
    server = find_binary(install_dir, "llama-server")
    result4 = None
    if server is None:
        print("llama-server not found under the install dir; cannot run GPU smoke.")
        for p in install_dir.rglob("llama-*"):
            print("  found:", p)
    else:
        print(f"llama-server: {server}")
        gguf = download(MODEL_URL, WORK / "model.gguf")
        result4 = gpu_smoke(server, gguf, "fork", PORT)
        print(f"  started       : {result4['started']}  healthy: {result4['healthy']}")
        print(f"  GPU offloaded : {result4['gpu_offloaded']}   [{result4['gpu_layers']}]")
        print(f"  GPU memory    : {result4['gpu_mem_mib']}   per-pid: {result4['pid_mem_mib']} MiB")
        if result4["device_line"]:
            print(f"  device        : {result4['device_line']}")
        print(f"  answer (2+2)  : {result4['answer']!r}  -> correct: {result4['answer_ok']}")
        print("  --- server load log (head+tail) ---")
        print("  " + result4["log_tail"].replace("\n", "\n  "))

    # -------------------------------------------------------------- PART 5
    results5 = []
    if args.full:
        hr("PART 5 - EMPIRICAL: raw upstream cuda-12.4 vs cuda-13.3 on this GPU/driver")
        if not (host.is_windows and host.is_x86_64):
            print("PART 5 targets the Windows upstream CUDA builds (the pin scenario).")
            print("Upstream ships no Linux CUDA bundle, so this is skipped off Windows.")
        elif not tag:
            print("no upstream tag resolved; skipping.")
        else:
            assets = M.github_release_assets(UPSTREAM_REPO, tag)
            for cu in ("12.4", "13.3"):
                name = f"llama-{tag}-bin-win-cuda-{cu}-x64.zip"
                cudart = f"cudart-llama-bin-win-cuda-{cu}-x64.zip"
                if name not in assets:
                    print(f"\n[{cu}] {name} not in this release; skipping")
                    continue
                d = WORK / f"upstream_cuda{cu}"
                shutil.rmtree(d, ignore_errors=True)
                extract(download(assets[name], WORK / name), d)
                if cudart in assets:
                    extract(download(assets[cudart], WORK / cudart), d)  # overlay DLLs
                srv = find_binary(d, "llama-server")
                if srv is None:
                    print(f"[{cu}] llama-server not found after extract; skipping")
                    continue
                res = gpu_smoke(srv, WORK / "model.gguf", f"upstream{cu}", PORT + 1 + len(results5))
                results5.append((cu, res))
                print(f"\n[upstream cuda-{cu}] started={res['started']} healthy={res['healthy']} "
                      f"gpu_offloaded={res['gpu_offloaded']} answer_ok={res['answer_ok']}")
                print(f"   [{res['gpu_layers']}]")
                if not res["gpu_offloaded"]:
                    print("   " + res["log_tail"].replace("\n", "\n   "))

    # -------------------------------------------------------------- SUMMARY
    hr("SUMMARY (paste this back)")
    print(f"GPU max sm           : {max_sm}  (Blackwell sm_120+: {is_blackwell})")
    print(f"driver CUDA version  : {host.driver_cuda_version}")
    print(f"fork install marker  : {installed.get('asset') if installed else '(none)'}")
    if result4:
        print(f"fork GPU offload     : {result4['gpu_offloaded']}   answer 2+2 correct: {result4['answer_ok']}")
        print(f"fork GPU memory      : {result4['gpu_mem_mib']}  (per-pid {result4['pid_mem_mib']} MiB)")
    if tag:
        print(f"upstream-only path   : see PART 2 above (the fork-down scenario)")
    for cu, res in results5:
        print(f"raw upstream cuda{cu} : gpu_offloaded={res['gpu_offloaded']} answer_ok={res['answer_ok']}")
    print("\nInterpretation:")
    print("- If 'fork GPU offload' is True + answer correct => Studio's normal path gives")
    print("  this Blackwell GPU a working GPU build. Pin removal is safe for real usage.")
    print("- PART 2 shows what the rare fork-DOWN path would do for this exact host.")

    if not args.keep:
        try:
            (WORK / "model.gguf").unlink(missing_ok=True)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
