#!/usr/bin/env bash
#
# amd_confirm.sh  (Linux AMD ROCm prebuilt confirmation)
# Confirms the llama.cpp build the Unsloth installer would fetch for an AMD GPU
# on Linux actually runs, on the GPU. AMD does NOT use the unslothai bundles -
# the installer pulls lemonade-sdk/llamacpp-rocm, selected by the GPU's gfx arch:
#   llama-<lemo_tag>-ubuntu-rocm-<gfx_family>-x64.zip
# If the gfx is not covered by lemonade (or no ROCm is detected) it falls back to
# the upstream CPU build (llama-<tag>-bin-ubuntu-x64.tar.gz), same as the
# installer. Then it runs real inference + tool calling and prints a PASS/FAIL
# report with measured tok/s (the surest GPU-vs-CPU signal).
#
# Nothing is installed system-wide; files go under $WORK and can be deleted.
#
# Usage:   bash amd_confirm.sh
# Env:     WORK PORT GGUF_URL KEEP=1 AMD_GFX=gfx1100 (force) LEMONADE_TAG=b1292
#          TAG=b9518 (upstream CPU-fallback tag) BUNDLE_URL=... (force any zip/tgz)
#          HSA_OVERRIDE_GFX_VERSION=11.0.0 (pass through for unsupported gfx)
#
set -uo pipefail

WORK="${WORK:-$HOME/llama_amd_test}"
PORT="${PORT:-8141}"
GGUF_URL="${GGUF_URL:-https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
KEEP="${KEEP:-0}"
TAG="${TAG:-b9518}"                 # upstream ggml-org tag for the CPU fallback
LEMONADE_REPO="lemonade-sdk/llamacpp-rocm"
UPSTREAM_REPO="ggml-org/llama.cpp"
AMD_GFX="${AMD_GFX:-}"
BUNDLE_URL="${BUNDLE_URL:-}"
ARCH="$(uname -m)"

PASS_N=0; FAIL_N=0; WARN_N=0; SERVER_PID=""; HAVE_GPU=0; CPU_BUILD=0
bold(){ printf '\033[1m%s\033[0m\n' "$*"; }
ok(){   printf '  [PASS] %s\n' "$*"; PASS_N=$((PASS_N+1)); }
bad(){  printf '  [FAIL] %s\n' "$*"; FAIL_N=$((FAIL_N+1)); }
warn(){ printf '  [WARN] %s\n' "$*"; WARN_N=$((WARN_N+1)); }
info(){ printf '         %s\n' "$*"; }
hr(){   printf -- '---------------------------------------------------------------\n'; }

cleanup(){ [ -n "$SERVER_PID" ] && kill "$SERVER_PID" >/dev/null 2>&1; [ "$KEEP" != "1" ] && rm -f "$WORK"/bundle.zip "$WORK"/bundle.tgz >/dev/null 2>&1; }
trap cleanup EXIT

# gfx -> lemonade family (mirrors install_llama_prebuilt.py _LEMONADE_GFX_FAMILIES;
# specific prefixes first).
gfx_family(){
  case "$1" in
    gfx1151) echo gfx1151 ;;
    gfx1150) echo gfx1150 ;;
    gfx120*) echo gfx120X ;;
    gfx110*) echo gfx110X ;;
    gfx103*) echo gfx103X ;;
    *)       echo "" ;;
  esac
}

mkdir -p "$WORK"; cd "$WORK" || { echo "cannot cd $WORK"; exit 1; }
echo; bold "=== Unsloth llama.cpp AMD ROCm prebuilt confirmation (Linux) ==="
echo "scratch dir : $WORK"; hr

# --------------------------------------------------------------------------- #
# 1. Host
# --------------------------------------------------------------------------- #
bold "1) Host detection"
info "uname     : $(uname -s) $ARCH ($(uname -r))"
[ -r /etc/os-release ] && { . /etc/os-release; info "distro    : ${PRETTY_NAME:-unknown}"; }
case "$ARCH" in
  x86_64) ok "x86_64 Linux" ;;
  *)      warn "lemonade ROCm builds are x64-only; arch '$ARCH' will use the CPU fallback" ;;
esac
if command -v rocminfo >/dev/null 2>&1 || command -v amd-smi >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1; then
  info "ROCm tools: $(command -v rocminfo amd-smi rocm-smi 2>/dev/null | tr '\n' ' ')"
fi
# Match the AMD vendor specifically: a bare "ATI" substring also hits
# "compATIble"/"CorporATIon" on non-AMD GPUs, so key on the real vendor strings.
AMD_RE='Radeon|Advanced Micro Devices|\[AMD'
if lspci 2>/dev/null | grep -iE "VGA|Display|3D controller" | grep -qiE "$AMD_RE"; then
  HAVE_GPU=1
  lspci 2>/dev/null | grep -iE "VGA|Display|3D controller" | grep -iE "$AMD_RE" | sed 's/^/           - /'
  ok "AMD GPU present (lspci)"
else
  warn "no AMD GPU on lspci (normal under WSL2 - GPU is paravirtual); selection falls to rocminfo/AMD_GFX"
fi
hr

# --------------------------------------------------------------------------- #
# 2. Detect gfx arch + lemonade family
# --------------------------------------------------------------------------- #
bold "2) Detect AMD gfx arch"
GFX="$AMD_GFX"
# rocminfo may be on PATH or only under /opt/rocm/bin (the ROCDXG profile.d PATH
# edit reaches login shells only). In WSL2 the system HSA runtime enumerates the
# GPU over /dev/dxg ONLY when HSA_ENABLE_DXG_DETECTION=1 (harmless no-op on bare
# metal), so set it for the probe or a ROCDXG box reports no GPU -> false CPU build.
ROCMINFO="$(command -v rocminfo 2>/dev/null)"
[ -z "$ROCMINFO" ] && [ -x /opt/rocm/bin/rocminfo ] && ROCMINFO=/opt/rocm/bin/rocminfo
if [ -z "$GFX" ] && [ -n "$ROCMINFO" ]; then
  GFX="$(HSA_ENABLE_DXG_DETECTION=1 "$ROCMINFO" 2>/dev/null | grep -oiE 'gfx[1-9][0-9a-z]{2,3}' | head -1 | tr 'A-Z' 'a-z')"
fi
if [ -n "$GFX" ]; then
  [ -n "$AMD_GFX" ] && info "gfx (forced): $GFX" || info "gfx (rocminfo): $GFX"
  HAVE_GPU=1
else
  info "no gfx detected (rocminfo missing or no ROCm GPU). Set AMD_GFX=gfxNNNN to force a ROCm build."
fi
FAMILY=""; [ -n "$GFX" ] && FAMILY="$(gfx_family "$GFX")"
if [ -n "$FAMILY" ]; then ok "lemonade family: $FAMILY"
elif [ -n "$GFX" ]; then warn "gfx '$GFX' is not covered by lemonade ROCm prebuilts - falling back to the CPU build"; fi
hr

# --------------------------------------------------------------------------- #
# 3. Select + download
# --------------------------------------------------------------------------- #
bold "3) Select + download the prebuilt"
EXT="zip"
if [ -n "$BUNDLE_URL" ]; then
  info "selection : forced via BUNDLE_URL"
  case "$BUNDLE_URL" in *.tar.gz|*.tgz) EXT="tgz";; esac
  case "$BUNDLE_URL" in *bin-ubuntu-x64*|*-cpu-*) CPU_BUILD=1;; esac
elif [ -n "$FAMILY" ]; then
  LEMO_TAG="${LEMONADE_TAG:-$(curl -fsSL "https://api.github.com/repos/$LEMONADE_REPO/releases/latest" 2>/dev/null | grep -oE '"tag_name"[^,]*' | grep -oE 'b[0-9]+' | head -1)}"
  [ -z "$LEMO_TAG" ] && LEMO_TAG="b1292"
  BUNDLE_URL="https://github.com/$LEMONADE_REPO/releases/download/$LEMO_TAG/llama-$LEMO_TAG-ubuntu-rocm-$FAMILY-x64.zip"
  info "selection : lemonade ROCm $FAMILY @ $LEMO_TAG"
else
  CPU_BUILD=1; EXT="tgz"
  BUNDLE_URL="https://github.com/$UPSTREAM_REPO/releases/download/$TAG/llama-$TAG-bin-ubuntu-x64.tar.gz"
  info "selection : no ROCm-capable GPU detected - upstream CPU build @ $TAG"
fi
info "bundle    : $BUNDLE_URL"
if curl -fL --retry 3 -o "bundle.$EXT" "$BUNDLE_URL"; then ok "downloaded ($(du -h "bundle.$EXT" | cut -f1))"
else bad "download failed"; echo; bold "Cannot continue."; exit 1; fi
hr

# --------------------------------------------------------------------------- #
# 4. Extract
# --------------------------------------------------------------------------- #
bold "4) Extract"
rm -rf bundle && mkdir -p bundle
if [ "$EXT" = "zip" ]; then
  if command -v unzip >/dev/null 2>&1; then unzip -q "bundle.zip" -d bundle; else python3 -m zipfile -e "bundle.zip" bundle; fi
else
  tar -xzf "bundle.tgz" -C bundle
fi
SRV="$(find bundle -type f -name llama-server | head -1)"
[ -n "$SRV" ] && { B="$(dirname "$SRV")"; chmod +x "$B"/llama-* 2>/dev/null; ok "extracted (llama-server in $B)"; } || { bad "llama-server not found in the archive"; echo; exit 1; }
hr

# --------------------------------------------------------------------------- #
# 5. Version
# --------------------------------------------------------------------------- #
bold "5) llama-server --version"
RUNPATH="$B:${LD_LIBRARY_PATH:-}"
VOUT="$(LD_LIBRARY_PATH="$RUNPATH" "$B/llama-server" --version 2>&1 | head -3)"; echo "$VOUT" | sed 's/^/         /'
echo "$VOUT" | grep -qiE "version:" && ok "binary runs" || bad "binary did not run"
hr

# --------------------------------------------------------------------------- #
# 6. Model
# --------------------------------------------------------------------------- #
bold "6) Download a small test GGUF"
GGUF="$WORK/$(basename "$GGUF_URL")"
if [ -s "$GGUF" ]; then ok "model present ($(du -h "$GGUF" | cut -f1))"
elif curl -fL --retry 3 -o "$GGUF" "$GGUF_URL"; then ok "downloaded model ($(du -h "$GGUF" | cut -f1))"
else bad "model download failed"; fi
hr

# --------------------------------------------------------------------------- #
# 7. Inference with full offload (-ngl 99)
# --------------------------------------------------------------------------- #
bold "7) Inference with full offload (-ngl 99)"
SRVLOG="$WORK/server.log"
LD_LIBRARY_PATH="$RUNPATH" ${HSA_OVERRIDE_GFX_VERSION:+HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION} \
  "$B/llama-server" -m "$GGUF" -ngl 99 --host 127.0.0.1 --port "$PORT" -c 2048 --jinja > "$SRVLOG" 2>&1 &
SERVER_PID=$!
READY=0
for i in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && { READY=1; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1
done
if [ "$READY" = "1" ]; then
  ok "server healthy on :$PORT"
  RESP="$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"In one short sentence, what is the capital of Japan?"}],"max_tokens":40,"temperature":0}')"
  CONTENT="$(printf '%s' "$RESP" | (python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || echo ""))"
  [ -z "$CONTENT" ] && CONTENT="$(printf '%s' "$RESP" | grep -oE '"content":"[^"]*"' | head -1)"
  TPS="$(printf '%s' "$RESP" | python3 -c 'import sys,json;d=json.load(sys.stdin);t=(d.get("timings") or {}).get("predicted_per_second") or 0;print(f"{t:.1f}" if t else "")' 2>/dev/null)"
  [ -z "$TPS" ] && TPS="$(grep -E "tokens per second" "$SRVLOG" | grep -vi "prompt eval" | grep -oE "[0-9.]+ tokens per second" | tail -1 | grep -oE "^[0-9.]+")"
  TPSTXT=""; [ -n "$TPS" ] && TPSTXT=" (generation ${TPS} tok/s)"
  if [ "$CPU_BUILD" = 1 ]; then
    info "CPU build - inference runs on CPU as expected${TPSTXT}"
  else
    GPU_FAIL="$(grep -iE "failed to initialize|no ROCm|no HIP|no usable GPU|hipError|ROCm error|no .*device" "$SRVLOG" | head -3)"
    GPU_DEV="$(grep -iE -- "-[[:space:]]*(ROCm|HIP|CUDA)[0-9]+[[:space:]]*:.*MiB|using device (ROCm|HIP)|found [0-9]+ (ROCm|HIP) device" "$SRVLOG" | head -4)"
    if [ -n "$GPU_DEV" ] && [ -z "$GPU_FAIL" ]; then
      echo "$GPU_DEV" | sed 's/^/         /'
      ok "ROCm GPU enumerated and active${TPSTXT}"
    elif [ -n "$GPU_FAIL" ]; then
      echo "$GPU_FAIL" | sed 's/^/         /'
      if [ "$HAVE_GPU" = 1 ]; then bad "an AMD GPU is present but ROCm failed to initialize - ran on CPU${TPSTXT} (try HSA_OVERRIDE_GFX_VERSION)"; else warn "ROCm did not initialize - running on CPU${TPSTXT}"; fi
    elif [ "$HAVE_GPU" = 1 ]; then
      bad "AMD GPU present but llama.cpp did not enumerate a ROCm device - ran on CPU${TPSTXT}"
      grep -iE "rocm|hip|device|fitting|buffer|backend|error" "$SRVLOG" | head -12 | sed 's/^/         /'
    else
      warn "no ROCm device enumerated - running on CPU${TPSTXT}"
    fi
  fi
  info "model reply: $CONTENT"
  printf '%s' "$CONTENT" | grep -qi "tokyo" && ok "coherent generation (mentions Tokyo)" || warn "answer unexpected"
else
  bad "server failed to become ready:"; tail -15 "$SRVLOG" | sed 's/^/         /'
fi
hr

# --------------------------------------------------------------------------- #
# 8. Tool calling
# --------------------------------------------------------------------------- #
bold "8) Tool calling"
if [ "$READY" = "1" ]; then
  TRESP="$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What is the weather in Paris? Use the get_weather tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"tool_choice":"auto","max_tokens":128,"temperature":0}')"
  printf '%s' "$TRESP" | grep -qi "get_weather" && ok "model emitted a get_weather tool call" || warn "no tool call (small model may decline)"
else warn "skipped (server not ready)"; fi
hr

bold "=== SUMMARY ==="
echo "host: $ARCH   build: $(basename "$BUNDLE_URL")"
echo "PASS: $PASS_N   WARN: $WARN_N   FAIL: $FAIL_N"; echo
if [ "$FAIL_N" = "0" ]; then
  [ "$CPU_BUILD" = 1 ] && bold "RESULT: CONFIRMED - CPU prebuilt runs on this box (no ROCm GPU used)." || bold "RESULT: CONFIRMED - AMD ROCm prebuilt runs on this box."
else bold "RESULT: $FAIL_N hard failure(s) - paste this whole output back."; fi
echo "(server log: $SRVLOG)"
exit 0
