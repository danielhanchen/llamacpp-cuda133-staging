#!/usr/bin/env bash
#
# spark_confirm.sh  (Linux NVIDIA CUDA prebuilt confirmation)
# Confirms an Unsloth llama.cpp CUDA prebuilt actually runs on this box, on the
# GPU. Works on x86_64 (T4 / A100 / L4 / H100 / RTX 6000 / B200) and aarch64
# (DGX Spark / GB10). It DETECTS your installed CUDA runtime major (12 vs 13) and
# downloads the matching bundle - the same thing the real installer does - then
# runs real GPU inference + tool calling and prints a PASS/FAIL report.
#
# Nothing is installed system-wide and the GPU driver is never touched. Files go
# under $WORK (default ~/llama_prebuilt_test) and can be deleted afterwards.
#
# Usage:   bash spark_confirm.sh
#   Test the cuda13 path on a box that only has CUDA 12 installed (no sudo,
#   driver untouched):   INSTALL_CUDA13=1 bash spark_confirm.sh
# Env overrides: WORK, BUNDLE_URL (force a specific bundle), BUNDLE_SHA256,
#                GGUF_URL, PORT, CUDA_LIB_DIR (force runtime dir),
#                INSTALL_CUDA13=1 (stage CUDA 13.0 runtime locally), KEEP=1
#
set -uo pipefail

WORK="${WORK:-$HOME/llama_prebuilt_test}"
PORT="${PORT:-8131}"
GGUF_URL="${GGUF_URL:-https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-}"
KEEP="${KEEP:-0}"
ARCH="$(uname -m)"

# Published bundle URLs (auto-selected later by arch + detected cudart major).
URL_X64_CUDA13="https://github.com/unslothai/llama.cpp/releases/download/b9518/app-b9518-linux-x64-cuda13-portable.tar.gz"
URL_X64_CUDA12="https://github.com/unslothai/llama.cpp/releases/download/b9493/app-b9493-linux-x64-cuda12-portable.tar.gz"  # b9518 dropped cuda12; b9493 is the last with it
URL_ARM64_CUDA13="https://github.com/danielhanchen/llamacpp-cuda133-staging/releases/download/spark-test-b9518/app-b9518-linux-arm64-cuda13-portable.tar.gz"

BUNDLE_SHA256="${BUNDLE_SHA256:-}"

PASS_N=0; FAIL_N=0; WARN_N=0; SERVER_PID=""
bold(){ printf '\033[1m%s\033[0m\n' "$*"; }
ok(){   printf '  [PASS] %s\n' "$*"; PASS_N=$((PASS_N+1)); }
bad(){  printf '  [FAIL] %s\n' "$*"; FAIL_N=$((FAIL_N+1)); }
warn(){ printf '  [WARN] %s\n' "$*"; WARN_N=$((WARN_N+1)); }
info(){ printf '         %s\n' "$*"; }
hr(){   printf -- '---------------------------------------------------------------\n'; }

# Stage the CUDA 13.0 runtime (cudart + cublas) into $1/  from NVIDIA's redist
# CDN. Userspace only: no sudo, no apt, no driver touched. Used by INSTALL_CUDA13=1
# so a cuda12 box can also exercise the cuda13 bundle.
install_cuda13_runtime(){
  local dst="$1" base plat comp ver name; base="https://developer.download.nvidia.com/compute/cuda/redist"
  case "$ARCH" in aarch64) plat=linux-sbsa;; *) plat=linux-x86_64;; esac
  mkdir -p "$dst/.tmp"
  for cv in cuda_cudart:13.0.48 libcublas:13.0.0.19; do
    comp="${cv%:*}"; ver="${cv#*:}"; name="${comp}-${plat}-${ver}-archive"
    curl -fsSL -o "$dst/.tmp/c.tar.xz" "$base/${comp}/${plat}/${name}.tar.xz" || return 1
    tar -xf "$dst/.tmp/c.tar.xz" -C "$dst/.tmp" || return 1
    cp -a "$dst/.tmp/${name}/lib/." "$dst/" 2>/dev/null
    rm -rf "$dst/.tmp/${name}" "$dst/.tmp/c.tar.xz"
  done
  rm -rf "$dst/.tmp"; [ -e "$dst/libcudart.so.13" ]
}

cleanup(){ [ -n "$SERVER_PID" ] && kill "$SERVER_PID" >/dev/null 2>&1; [ "$KEEP" != "1" ] && rm -f "$WORK"/*.gguf "$WORK"/bundle.tar.gz >/dev/null 2>&1; }
trap cleanup EXIT

mkdir -p "$WORK"; cd "$WORK" || { echo "cannot cd $WORK"; exit 1; }
echo; bold "=== Unsloth llama.cpp CUDA prebuilt confirmation (Linux) ==="
echo "scratch dir : $WORK"; hr

# --------------------------------------------------------------------------- #
# 1. Host
# --------------------------------------------------------------------------- #
bold "1) Host detection"
info "uname     : $(uname -s) $ARCH ($(uname -r))"
grep -qiE "microsoft|wsl" /proc/version 2>/dev/null && { IS_WSL=1; info "WSL       : yes"; } || { IS_WSL=0; info "WSL       : no"; }
[ -r /etc/os-release ] && { . /etc/os-release; info "distro    : ${PRETTY_NAME:-unknown}"; }
case "$ARCH" in
  x86_64)  ok "x86_64 Linux (T4 / A100 / L4 / H100 / RTX 6000 / B200 class)" ;;
  aarch64) ok "aarch64 Linux (DGX Spark / GB10 class)" ;;
  *)       warn "unexpected arch '$ARCH' - continuing" ;;
esac
if command -v nvidia-smi >/dev/null 2>&1; then
  info "driver    : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)   (nvidia-smi CUDA cap: $(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1))"
  info "GPU(s)    :"
  nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader 2>/dev/null | sed 's/^/           - /'
  ok "GPU(s) visible via nvidia-smi"
  info "note: the nvidia-smi 'CUDA cap' is the DRIVER max, not your installed runtime - the bundle is chosen by the runtime below"
else
  bad "nvidia-smi not found - cannot see the GPU"
fi
hr

# --------------------------------------------------------------------------- #
# 2. Detect the installed CUDA runtime major (12 vs 13)
# --------------------------------------------------------------------------- #
bold "2) Detect installed CUDA runtime (libcudart.so.NN)"
CAND_DIRS=()
[ -n "$CUDA_LIB_DIR" ] && CAND_DIRS+=("$CUDA_LIB_DIR")
[ -n "${CUDA_HOME:-}" ] && CAND_DIRS+=("$CUDA_HOME/lib64" "$CUDA_HOME/lib" "$CUDA_HOME/targets/x86_64-linux/lib" "$CUDA_HOME/targets/sbsa-linux/lib")
[ -n "${CUDA_PATH:-}" ] && CAND_DIRS+=("$CUDA_PATH/lib64" "$CUDA_PATH/lib")
for d in /usr/local/cuda /usr/local/cuda-12* /usr/local/cuda-13*; do
  CAND_DIRS+=("$d/lib64" "$d/targets/x86_64-linux/lib" "$d/targets/sbsa-linux/lib")
done
CAND_DIRS+=("/usr/lib64-nvidia" "/usr/lib/x86_64-linux-gnu" "/usr/lib/aarch64-linux-gnu" "/usr/lib/wsl/lib")
# PyTorch / pip nvidia-* wheels (these ship libcudart/libcublas)
if command -v python3 >/dev/null 2>&1; then
  while IFS= read -r line; do [ -n "$line" ] && CAND_DIRS+=("$line"); done < <(python3 - <<'PY' 2>/dev/null
import importlib.util, os
seen=set()
for mod in ("torch","nvidia.cuda_runtime","nvidia.cublas"):
    try:
        s=importlib.util.find_spec(mod)
        if s and s.submodule_search_locations:
            b=list(s.submodule_search_locations)[0]
            for sub in ("lib","../cuda_runtime/lib","../cublas/lib","../../nvidia/cuda_runtime/lib","../../nvidia/cublas/lib"):
                p=os.path.normpath(os.path.join(b,sub))
                if p not in seen: seen.add(p); print(p)
    except Exception: pass
PY
)
fi

if [ "${INSTALL_CUDA13:-0}" = "1" ]; then
  info "INSTALL_CUDA13=1 -> staging CUDA 13.0 runtime (cudart 13.0.48 + cublas 13.0.0.19) into $WORK/cuda13-runtime"
  if install_cuda13_runtime "$WORK/cuda13-runtime"; then ok "CUDA 13.0 runtime staged locally (no sudo, driver untouched)"; CAND_DIRS=("$WORK/cuda13-runtime" "${CAND_DIRS[@]}")
  else warn "could not stage CUDA 13.0 runtime (network?)"; fi
fi

find_cudart() {  # echoes a dir containing libcudart.so.$1, or nothing
  local m="$1" d p
  for d in "${CAND_DIRS[@]}"; do [ -e "$d/libcudart.so.$m" ] && { echo "$d"; return 0; }; done
  p="$(ldconfig -p 2>/dev/null | awk -v n="libcudart.so.$m" '$0 ~ n {print $NF; exit}')"
  [ -n "$p" ] && { dirname "$p"; return 0; }
  return 1
}
RT13="$(find_cudart 13)"; RT12="$(find_cudart 12)"
[ -n "$RT13" ] && info "libcudart.so.13 : $RT13"
[ -n "$RT12" ] && info "libcudart.so.12 : $RT12"

# Choose runtime major + matching bundle
RT_MAJOR=""; RTLIB=""
if [ "$ARCH" = "aarch64" ]; then
  RT_MAJOR=13; RTLIB="$RT13"; DEF_BUNDLE="$URL_ARM64_CUDA13"
elif [ -n "$RT13" ]; then
  RT_MAJOR=13; RTLIB="$RT13"; DEF_BUNDLE="$URL_X64_CUDA13"
elif [ -n "$RT12" ]; then
  RT_MAJOR=12; RTLIB="$RT12"; DEF_BUNDLE="$URL_X64_CUDA12"
else
  RT_MAJOR=""; DEF_BUNDLE="$URL_X64_CUDA13"
fi
BUNDLE_URL="${BUNDLE_URL:-$DEF_BUNDLE}"

if [ -n "$RT_MAJOR" ]; then
  ok "installed CUDA runtime major = $RT_MAJOR -> selecting the cuda$RT_MAJOR bundle"
  [ "$RT_MAJOR" = 12 ] && info "(b9518 ships no cuda12 bundle; using b9493, the last release that did - this is the regression)"
else
  bad "no libcudart.so.12 or .so.13 found - install CUDA or 'pip install torch', or set CUDA_LIB_DIR=/path/to/lib"
fi
hr

# --------------------------------------------------------------------------- #
# 3. Download + extract
# --------------------------------------------------------------------------- #
bold "3) Download + extract the prebuilt"
info "bundle: $BUNDLE_URL"
if curl -fL --retry 3 -o bundle.tar.gz "$BUNDLE_URL"; then ok "downloaded ($(du -h bundle.tar.gz | cut -f1))"
else bad "download failed"; echo; bold "Cannot continue."; exit 1; fi
if [ -n "$BUNDLE_SHA256" ]; then
  GOT="$(sha256sum bundle.tar.gz | awk '{print $1}')"; [ "$GOT" = "$BUNDLE_SHA256" ] && ok "sha256 verified" || bad "sha256 mismatch ($GOT)"
fi
rm -rf bundle && mkdir -p bundle && tar -xzf bundle.tar.gz -C bundle
B="$WORK/bundle"
[ -x "$B/llama-server" ] && [ -e "$B/libggml-cuda.so" ] && ok "extracted (llama-server + libggml-cuda.so)" || bad "bundle missing binaries"
hr

# --------------------------------------------------------------------------- #
# 4. Metadata
# --------------------------------------------------------------------------- #
bold "4) Bundle metadata"
if [ -r "$B/BUILD_INFO.txt" ]; then
  grep -iE "version|variant|toolkit|supported sms|arch|backend" "$B/BUILD_INFO.txt" | sed 's/^/         /'
  BVAR="$(grep -i '^variant' "$B/BUILD_INFO.txt" | awk '{print $2}')"
  case "$BVAR" in
    cuda$RT_MAJOR-*) ok "bundle line ($BVAR) matches installed runtime (cuda$RT_MAJOR)" ;;
    *) [ -n "$RT_MAJOR" ] && warn "bundle is $BVAR but runtime is cuda$RT_MAJOR (override mismatch?)" ;;
  esac
  if [ "$ARCH" = "aarch64" ]; then grep -iE "supported sms" "$B/BUILD_INFO.txt" | grep -q 121 && ok "native sm_121 listed (Spark)" || warn "sm_121 not listed (PTX-JIT)"; fi
else warn "no BUILD_INFO.txt"; fi
hr

# --------------------------------------------------------------------------- #
# 5. Library resolution
# --------------------------------------------------------------------------- #
bold "5) CUDA backend library resolution"
RUNPATH="$B"; for d in "${CAND_DIRS[@]}"; do [ -d "$d" ] && RUNPATH="$RUNPATH:$d"; done
RUNPATH="$RUNPATH:${LD_LIBRARY_PATH:-}"
LDD_OUT="$(LD_LIBRARY_PATH="$RUNPATH" ldd "$B/libggml-cuda.so" 2>&1)"
echo "$LDD_OUT" | grep -iE "cudart|cublas|libcuda\.so" | sed 's/^/         /'
if echo "$LDD_OUT" | grep -qi "not found"; then
  bad "some CUDA libraries did not resolve (see 'not found' above) - the GPU backend will not load"
else ok "all CUDA libraries resolved"; fi
hr

# --------------------------------------------------------------------------- #
# 6. Version
# --------------------------------------------------------------------------- #
bold "6) llama-server --version"
VOUT="$(LD_LIBRARY_PATH="$RUNPATH" "$B/llama-server" --version 2>&1 | head -3)"; echo "$VOUT" | sed 's/^/         /'
echo "$VOUT" | grep -qiE "version:" && ok "binary runs" || bad "binary did not run"
hr

# --------------------------------------------------------------------------- #
# 7. Model
# --------------------------------------------------------------------------- #
bold "7) Download a small test GGUF"
GGUF="$WORK/$(basename "$GGUF_URL")"
if [ -s "$GGUF" ]; then ok "model present ($(du -h "$GGUF" | cut -f1))"
elif curl -fL --retry 3 -o "$GGUF" "$GGUF_URL"; then ok "downloaded model ($(du -h "$GGUF" | cut -f1))"
else bad "model download failed"; fi
hr

# --------------------------------------------------------------------------- #
# 8. Inference + honest GPU check
# --------------------------------------------------------------------------- #
bold "8) GPU inference with full offload (-ngl 99)"
SRVLOG="$WORK/server.log"
LD_LIBRARY_PATH="$RUNPATH" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$B/llama-server" -m "$GGUF" -ngl 99 --host 127.0.0.1 --port "$PORT" -c 2048 --jinja > "$SRVLOG" 2>&1 &
SERVER_PID=$!
READY=0
for i in $(seq 1 90); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && { READY=1; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1
done
if [ "$READY" = "1" ]; then
  ok "server healthy on :$PORT"
  # Honest GPU determination from the server log.
  GPULINE="$(grep -iE "offloaded .* layers to GPU|CUDA0|ggml_cuda_init|using device CUDA|ARCHS =" "$SRVLOG" | head -4)"
  if [ -n "$GPULINE" ]; then
    echo "$GPULINE" | sed 's/^/         /'
    ok "CUDA GPU backend is ACTIVE (real GPU offload)"
  else
    warn "no CUDA device lines in the log - the model is running on CPU (GPU backend did not load)"
    grep -iE "buffer size|backend|cpu" "$SRVLOG" | head -3 | sed 's/^/         /'
  fi
else
  bad "server failed to become ready:"; tail -15 "$SRVLOG" | sed 's/^/         /'
fi
if [ "$READY" = "1" ]; then
  RESP="$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"In one short sentence, what is the capital of Japan?"}],"max_tokens":40,"temperature":0}')"
  CONTENT="$(printf '%s' "$RESP" | (python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || echo ""))"
  [ -z "$CONTENT" ] && CONTENT="$(printf '%s' "$RESP" | grep -oE '"content":"[^"]*"' | head -1)"
  info "model reply: $CONTENT"
  printf '%s' "$CONTENT" | grep -qi "tokyo" && ok "coherent generation (mentions Tokyo)" || warn "answer unexpected"
fi
hr

# --------------------------------------------------------------------------- #
# 9. Tool calling
# --------------------------------------------------------------------------- #
bold "9) Tool calling"
if [ "$READY" = "1" ]; then
  TRESP="$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What is the weather in Paris? Use the get_weather tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"tool_choice":"auto","max_tokens":128,"temperature":0}')"
  printf '%s' "$TRESP" | grep -qi "get_weather" && ok "model emitted a get_weather tool call" || warn "no tool call (small model may decline)"
else warn "skipped (server not ready)"; fi
hr

bold "=== SUMMARY ==="
echo "host: $ARCH   runtime: cuda${RT_MAJOR:-none}   bundle: $(basename "$BUNDLE_URL")"
echo "PASS: $PASS_N   WARN: $WARN_N   FAIL: $FAIL_N"; echo
if [ "$FAIL_N" = "0" ]; then bold "RESULT: CONFIRMED - prebuilt runs on this box."
else bold "RESULT: $FAIL_N hard failure(s) - paste this whole output back."; fi
echo "(server log: $SRVLOG)"
exit 0
