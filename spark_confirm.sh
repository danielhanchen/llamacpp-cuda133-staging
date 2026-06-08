#!/usr/bin/env bash
#
# spark_confirm.sh
# Standalone check that the Unsloth llama.cpp arm64 (sm_121) prebuilt actually
# runs on a DGX Spark / GB10 box (native aarch64 Linux or WSL2 on Windows-on-ARM).
#
# It is self-contained: it detects the host, downloads a prebuilt arm64 CUDA 13
# bundle, locates your CUDA runtime, then runs real GPU inference + tool calling
# and prints a PASS/FAIL report you can paste back.
#
# Nothing is installed system-wide and the GPU driver is never touched. Everything
# lands under $WORK (default ~/spark_llama_test) and can be deleted afterwards.
#
# Usage:
#   bash spark_confirm.sh
# Optional overrides (env vars):
#   WORK=/path/to/scratch         where to download/extract (default ~/spark_llama_test)
#   BUNDLE_URL=...                prebuilt arm64 bundle tar.gz
#   BUNDLE_SHA256=...             expected sha256 of the bundle (empty = skip verify)
#   GGUF_URL=...                  test model GGUF (default unsloth Llama-3.2-1B Q4_K_M)
#   PORT=8131                     llama-server port
#   CUDA_LIB_DIR=/path/lib        force the dir holding libcudart.so.13 / libcublas.so.13
#   KEEP=1                        keep $WORK after finishing
#
set -uo pipefail

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
WORK="${WORK:-$HOME/llama_prebuilt_test}"
PORT="${PORT:-8131}"
# Default bundle is picked by CPU arch:
#   x86_64  -> the REAL published Unsloth x64 cuda13 bundle (H100 / RTX 6000 / B200)
#   aarch64 -> the staging arm64 cuda13 bundle built for DGX Spark / GB10 (sm_121)
_ARCH_DEFAULT="$(uname -m)"
if [ "$_ARCH_DEFAULT" = "x86_64" ]; then
  _DEF_BUNDLE="https://github.com/unslothai/llama.cpp/releases/download/b9518/app-b9518-linux-x64-cuda13-portable.tar.gz"
else
  _DEF_BUNDLE="https://github.com/danielhanchen/llamacpp-cuda133-staging/releases/download/spark-test-b9518/app-b9518-linux-arm64-cuda13-portable.tar.gz"
fi
BUNDLE_URL="${BUNDLE_URL:-$_DEF_BUNDLE}"
BUNDLE_SHA256="${BUNDLE_SHA256:-}"
GGUF_URL="${GGUF_URL:-https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-}"
KEEP="${KEEP:-0}"

PASS_N=0
FAIL_N=0
WARN_N=0
SERVER_PID=""

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
ok()    { printf '  [PASS] %s\n' "$*"; PASS_N=$((PASS_N+1)); }
bad()   { printf '  [FAIL] %s\n' "$*"; FAIL_N=$((FAIL_N+1)); }
warn()  { printf '  [WARN] %s\n' "$*"; WARN_N=$((WARN_N+1)); }
info()  { printf '         %s\n' "$*"; }
hr()    { printf -- '---------------------------------------------------------------\n'; }

cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" >/dev/null 2>&1
  if [ "$KEEP" != "1" ]; then
    # keep the report log, drop the big files
    rm -f "$WORK"/*.gguf "$WORK"/bundle.tar.gz >/dev/null 2>&1
  fi
}
trap cleanup EXIT

mkdir -p "$WORK"
cd "$WORK" || { echo "cannot cd $WORK"; exit 1; }

echo
bold "=== Unsloth llama.cpp prebuilt confirmation (Linux CUDA: B200 / H100 / RTX 6000 / Spark) ==="
echo "scratch dir : $WORK"
echo "bundle      : $BUNDLE_URL"
echo "model       : $GGUF_URL"
hr

# --------------------------------------------------------------------------- #
# 1. Host detection
# --------------------------------------------------------------------------- #
bold "1) Host detection"
OS="$(uname -s)"
ARCH="$(uname -m)"
info "uname     : $OS $ARCH ($(uname -r))"
IS_WSL=0
if grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then IS_WSL=1; info "WSL       : yes (Windows Subsystem for Linux)"; else info "WSL       : no"; fi
if [ -r /etc/os-release ]; then . /etc/os-release; info "distro    : ${PRETTY_NAME:-unknown}"; fi

case "$ARCH" in
  aarch64|arm64) ok "architecture is aarch64 (arm64) - correct for DGX Spark / GB10" ;;
  *) bad "architecture is '$ARCH', not aarch64 - this script tests the arm64 bundle; a Spark/GB10 should be aarch64. Continuing anyway." ;;
esac

if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad "nvidia-smi not found - cannot see the GPU. On WSL ensure the NVIDIA Windows driver + WSL CUDA are installed."
else
  DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  CUDA_VER="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)"
  info "driver    : ${DRV:-unknown}   (nvidia-smi CUDA: ${CUDA_VER:-unknown})"
  info "GPU(s)    :"
  nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader 2>/dev/null | sed 's/^/           - /'
  ok "GPU(s) visible via nvidia-smi"
  case "$CC" in
    12.1) ok "compute capability 12.1 (sm_121) detected - native SASS path in the bundle" ;;
    "" )  warn "could not read compute capability" ;;
    *)    warn "compute capability is $CC (expected 12.1 for GB10). Bundle will still cover it (native if listed, else PTX-JIT)." ;;
  esac
fi
hr

# --------------------------------------------------------------------------- #
# 2. Locate the CUDA runtime (the bundle ships NO cudart - it uses yours)
# --------------------------------------------------------------------------- #
bold "2) Locate CUDA 13 runtime libraries (libcudart.so.13 / libcublas.so.13)"
CAND_DIRS=()
[ -n "$CUDA_LIB_DIR" ] && CAND_DIRS+=("$CUDA_LIB_DIR")
[ -n "${CUDA_HOME:-}" ] && CAND_DIRS+=("$CUDA_HOME/lib64" "$CUDA_HOME/lib" "$CUDA_HOME/targets/sbsa-linux/lib")
[ -n "${CUDA_PATH:-}" ] && CAND_DIRS+=("$CUDA_PATH/lib64" "$CUDA_PATH/lib")
for d in /usr/local/cuda /usr/local/cuda-13 /usr/local/cuda-13.*; do
  CAND_DIRS+=("$d/lib64" "$d/targets/sbsa-linux/lib")
done
# PyTorch-provided cudart (pip nvidia-cuda-runtime-cu13 / torch/lib)
if command -v python3 >/dev/null 2>&1; then
  PYL="$(python3 - <<'PY' 2>/dev/null
import importlib.util, os
for mod in ("torch","nvidia.cuda_runtime","nvidia.cublas"):
    try:
        spec = importlib.util.find_spec(mod)
        if spec and spec.submodule_search_locations:
            base = list(spec.submodule_search_locations)[0]
            for sub in ("lib","../cuda_runtime/lib","../cublas/lib"):
                print(os.path.normpath(os.path.join(base, sub)))
    except Exception:
        pass
PY
)"
  while IFS= read -r line; do [ -n "$line" ] && CAND_DIRS+=("$line"); done <<< "$PYL"
fi

RTLIB=""
for d in "${CAND_DIRS[@]}"; do
  if [ -e "$d/libcudart.so.13" ]; then RTLIB="$d"; break; fi
done
if [ -z "$RTLIB" ]; then
  # last resort: ask the linker cache
  RTLIB="$(dirname "$(ldconfig -p 2>/dev/null | awk '/libcudart\.so\.13/{print $NF; exit}')" 2>/dev/null)"
fi

# driver lib (libcuda.so.1): system, or WSL stub dir
DRVDIR=""
for d in /usr/lib/wsl/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /lib/aarch64-linux-gnu; do
  [ -e "$d/libcuda.so.1" ] && { DRVDIR="$d"; break; }
done

if [ -n "$RTLIB" ] && [ -e "$RTLIB/libcudart.so.13" ]; then
  ok "found libcudart.so.13 in: $RTLIB"
  ls "$RTLIB"/libcublas.so.13 >/dev/null 2>&1 && info "libcublas.so.13 present too" || warn "libcublas.so.13 not in that dir (may be elsewhere on the path)"
else
  bad "could not find libcudart.so.13. Install CUDA 13 (or 'pip install torch' with a cu13 build), or set CUDA_LIB_DIR=/path/to/lib"
  info "the arm64 bundle deliberately does not ship cudart; it pairs with your runtime"
fi
[ -n "$DRVDIR" ] && info "driver libcuda.so.1 in: $DRVDIR"
RUNPATH="$RTLIB:${DRVDIR}:${LD_LIBRARY_PATH:-}"
hr

# --------------------------------------------------------------------------- #
# 3. Download + extract the prebuilt bundle
# --------------------------------------------------------------------------- #
bold "3) Download + extract the arm64 prebuilt"
if curl -fL --retry 3 -o bundle.tar.gz "$BUNDLE_URL"; then
  SZ="$(du -h bundle.tar.gz | cut -f1)"
  ok "downloaded bundle ($SZ)"
else
  bad "download failed from $BUNDLE_URL"
  echo; bold "Cannot continue without the bundle."; exit 1
fi
if [ -n "$BUNDLE_SHA256" ]; then
  GOT="$(sha256sum bundle.tar.gz | awk '{print $1}')"
  if [ "$GOT" = "$BUNDLE_SHA256" ]; then ok "sha256 verified"; else bad "sha256 mismatch (got $GOT)"; fi
else
  warn "no BUNDLE_SHA256 provided - skipping integrity check"
fi
rm -rf bundle && mkdir -p bundle && tar -xzf bundle.tar.gz -C bundle
B="$WORK/bundle"
if [ -x "$B/llama-server" ] && [ -e "$B/libggml-cuda.so" ]; then ok "extracted (llama-server + libggml-cuda.so present)"; else bad "bundle missing llama-server or libggml-cuda.so"; fi
hr

# --------------------------------------------------------------------------- #
# 4. Inspect bundle metadata
# --------------------------------------------------------------------------- #
bold "4) Bundle metadata (BUILD_INFO.txt)"
if [ -r "$B/BUILD_INFO.txt" ]; then
  grep -iE "version|variant|toolkit|supported sms|arch|backend" "$B/BUILD_INFO.txt" | sed 's/^/         /'
  grep -qi "arch: arm64" "$B/BUILD_INFO.txt" && ok "bundle arch = arm64" || warn "BUILD_INFO arch is not arm64"
  grep -qiE "toolkit version: 13" "$B/BUILD_INFO.txt" && ok "CUDA 13 toolkit" || warn "toolkit is not 13.x"
  if grep -iE "supported sms" "$B/BUILD_INFO.txt" | grep -q "121"; then ok "native sm_121 SASS listed (DGX Spark / GB10)"; else warn "sm_121 not in supported_sms (would run via PTX-JIT)"; fi
else
  warn "no BUILD_INFO.txt in bundle"
fi
hr

# --------------------------------------------------------------------------- #
# 5. Library resolution (the cross-version moment)
# --------------------------------------------------------------------------- #
bold "5) Dynamic library resolution of the CUDA backend"
LDD_OUT="$(LD_LIBRARY_PATH="$B:$RUNPATH" ldd "$B/libggml-cuda.so" 2>&1)"
echo "$LDD_OUT" | grep -iE "cudart|cublas|libcuda\.so" | sed 's/^/         /'
if echo "$LDD_OUT" | grep -qi "not found"; then
  bad "some libraries did not resolve:"; echo "$LDD_OUT" | grep -i "not found" | sed 's/^/         /'
else
  ok "all CUDA libraries resolved (no 'not found')"
fi
hr

# --------------------------------------------------------------------------- #
# 6. Version
# --------------------------------------------------------------------------- #
bold "6) llama-server --version"
VOUT="$(LD_LIBRARY_PATH="$B:$RUNPATH" "$B/llama-server" --version 2>&1 | head -3)"
echo "$VOUT" | sed 's/^/         /'
echo "$VOUT" | grep -qiE "version: [0-9]+" && ok "binary runs and reports a version" || bad "binary did not report a version"
hr

# --------------------------------------------------------------------------- #
# 7. Get a small test model
# --------------------------------------------------------------------------- #
bold "7) Download a small test GGUF"
GGUF="$WORK/$(basename "$GGUF_URL")"
if [ -s "$GGUF" ]; then ok "model already present ($(du -h "$GGUF" | cut -f1))";
elif curl -fL --retry 3 -o "$GGUF" "$GGUF_URL"; then ok "downloaded model ($(du -h "$GGUF" | cut -f1))";
else bad "model download failed"; fi
hr

# --------------------------------------------------------------------------- #
# 8. GPU inference
# --------------------------------------------------------------------------- #
bold "8) GPU inference with full offload (-ngl 99)"
SRVLOG="$WORK/server.log"
LD_LIBRARY_PATH="$B:$RUNPATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$B/llama-server" -m "$GGUF" -ngl 99 --host 127.0.0.1 --port "$PORT" -c 2048 --jinja \
  > "$SRVLOG" 2>&1 &
SERVER_PID=$!
READY=0
for i in $(seq 1 60); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)"
  [ "$code" = "200" ] && { READY=1; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || break
  sleep 1
done
if [ "$READY" = "1" ]; then
  ok "server came up and is healthy on :$PORT"
  grep -iE "CUDA0|GB10|NVIDIA|ARCHS|offloaded|layers to GPU" "$SRVLOG" | head -4 | sed 's/^/         /'
  grep -qiE "CUDA0|ARCHS" "$SRVLOG" && ok "GPU device + CUDA backend active" || warn "no explicit CUDA device line in log (check above)"
else
  bad "server failed to become ready - last log lines:"; tail -15 "$SRVLOG" | sed 's/^/         /'
fi

if [ "$READY" = "1" ]; then
  RESP="$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"In one short sentence, what is the capital of Japan?"}],"max_tokens":40,"temperature":0}')"
  CONTENT="$(printf '%s' "$RESP" | (python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || echo ""))"
  [ -z "$CONTENT" ] && CONTENT="$(printf '%s' "$RESP" | grep -oE '"content":"[^"]*"' | head -1)"
  info "model reply: $CONTENT"
  printf '%s' "$CONTENT" | grep -qi "tokyo" && ok "coherent GPU generation (mentions Tokyo)" || warn "generation returned but answer unexpected (see reply above)"
fi
hr

# --------------------------------------------------------------------------- #
# 9. Tool calling
# --------------------------------------------------------------------------- #
bold "9) Tool calling (OpenAI-style function call)"
if [ "$READY" = "1" ]; then
  TRESP="$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "messages":[{"role":"user","content":"What is the weather in Paris? Use the get_weather tool."}],
    "tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather for a location","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],
    "tool_choice":"auto","max_tokens":128,"temperature":0}')"
  FIN="$(printf '%s' "$TRESP" | (python3 -c 'import sys,json;d=json.load(sys.stdin)["choices"][0];print(d.get("finish_reason"));
import json as j
tc=d["message"].get("tool_calls")
print(j.dumps(tc[0]["function"]) if tc else "none")' 2>/dev/null || echo ""))"
  echo "$FIN" | sed 's/^/         /'
  if printf '%s' "$FIN" | grep -qi "get_weather"; then ok "model emitted a get_weather tool call"; else warn "no tool call emitted (small 1B model may decline; core GPU path already proven above)"; fi
else
  warn "skipped (server not ready)"
fi
hr

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
bold "=== SUMMARY ==="
echo "PASS: $PASS_N   WARN: $WARN_N   FAIL: $FAIL_N"
echo
if [ "$FAIL_N" = "0" ]; then
  bold "RESULT: CONFIRMED - the arm64 (sm_121) prebuilt runs on this Spark."
else
  bold "RESULT: NOT fully confirmed - $FAIL_N hard failure(s) above. Paste this whole output back."
fi
echo "(server log: $SRVLOG)"
exit 0
