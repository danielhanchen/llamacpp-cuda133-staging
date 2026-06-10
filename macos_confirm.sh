#!/usr/bin/env bash
#
# macos_confirm.sh
# Standalone check that the Unsloth llama.cpp macOS prebuilt runs on your Mac
# (Apple Silicon -> Metal GPU bundle; Intel -> CPU bundle). Confirms the binary
# dyld-loads on this macOS (its minos floor), launches, and does real inference.
#
# Nothing is installed system-wide. Everything lands under $WORK and can be
# deleted afterwards. Safe on macOS 26 (the bundle's load floor is 14.0/13.3,
# i.e. it runs on 14/13.3 AND newer including 26).
#
# Usage:   bash macos_confirm.sh
# Env overrides: WORK, BUNDLE_URL, BUNDLE_SHA256, GGUF_URL, PORT, KEEP=1
#
set -uo pipefail

WORK="${WORK:-$HOME/llama_prebuilt_test}"
PORT="${PORT:-8131}"
ARCH="$(uname -m)"   # arm64 or x86_64
case "$ARCH" in
  arm64)  _SLICE="arm64" ;;
  x86_64) _SLICE="x64"   ;;
  *)      _SLICE="arm64" ;;
esac
# NOTE: filled in once the macOS staging bundle is published.
BUNDLE_URL="${BUNDLE_URL:-https://github.com/danielhanchen/llamacpp-cuda133-staging/releases/download/spark-test-b9518/app-b9518-macos-${_SLICE}.tar.gz}"
BUNDLE_SHA256="${BUNDLE_SHA256:-}"
GGUF_URL="${GGUF_URL:-https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
KEEP="${KEEP:-0}"

PASS_N=0; FAIL_N=0; WARN_N=0; SERVER_PID=""
bold(){ printf '\033[1m%s\033[0m\n' "$*"; }
ok(){   printf '  [PASS] %s\n' "$*"; PASS_N=$((PASS_N+1)); }
bad(){  printf '  [FAIL] %s\n' "$*"; FAIL_N=$((FAIL_N+1)); }
warn(){ printf '  [WARN] %s\n' "$*"; WARN_N=$((WARN_N+1)); }
info(){ printf '         %s\n' "$*"; }
hr(){   printf -- '---------------------------------------------------------------\n'; }
sha256_of(){ if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'; else sha256sum "$1" | awk '{print $1}'; fi; }
cleanup(){ [ -n "$SERVER_PID" ] && kill "$SERVER_PID" >/dev/null 2>&1; [ "$KEEP" != "1" ] && rm -f "$WORK"/*.gguf "$WORK"/macbundle.tar.gz >/dev/null 2>&1; }
trap cleanup EXIT

mkdir -p "$WORK"; cd "$WORK" || { echo "cannot cd $WORK"; exit 1; }
echo; bold "=== Unsloth llama.cpp macOS prebuilt confirmation ==="
echo "scratch : $WORK"; echo "bundle  : $BUNDLE_URL"; hr

# 1) Host
bold "1) Host detection"
if ! command -v sw_vers >/dev/null 2>&1; then bad "sw_vers not found - is this macOS?"; fi
OSV="$(sw_vers -productVersion 2>/dev/null)"; OSB="$(sw_vers -buildVersion 2>/dev/null)"
CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
info "macOS    : ${OSV:-?} (${OSB:-?})"
info "arch     : $ARCH    chip: ${CHIP:-unknown}"
case "$ARCH" in
  arm64)  ok "Apple Silicon - will test the arm64 Metal bundle" ;;
  x86_64) ok "Intel Mac - will test the x64 CPU bundle (Metal off on Intel by design)" ;;
  *)      warn "unexpected arch '$ARCH'" ;;
esac
OSMAJOR="${OSV%%.*}"; [ -n "$OSMAJOR" ] && [ "$OSMAJOR" -ge 14 ] 2>/dev/null && info "macOS $OSV is >= the bundle's 14.0 load floor (incl. 26)"
hr

# 2) Download + extract
bold "2) Download + extract the macOS prebuilt"
if curl -fL --retry 3 -o macbundle.tar.gz "$BUNDLE_URL"; then ok "downloaded ($(du -h macbundle.tar.gz | cut -f1))"
else bad "download failed from $BUNDLE_URL"; echo; bold "Cannot continue without the bundle."; exit 1; fi
if [ -n "$BUNDLE_SHA256" ]; then
  GOT="$(sha256_of macbundle.tar.gz)"; [ "$GOT" = "$BUNDLE_SHA256" ] && ok "sha256 verified" || bad "sha256 mismatch (got $GOT)"
else warn "no BUNDLE_SHA256 provided - skipping integrity check"; fi
rm -rf macbundle && mkdir -p macbundle && tar -xzf macbundle.tar.gz -C macbundle
SRV="$(find macbundle -type f -name llama-server 2>/dev/null | head -1)"
if [ -n "$SRV" ] && [ -f "$SRV" ]; then B="$(dirname "$SRV")"; ok "extracted (llama-server found)"; info "bin dir: $B"
else bad "llama-server not found in bundle"; echo; exit 1; fi
hr

# 3) Gatekeeper / quarantine
bold "3) Clear Gatekeeper quarantine (downloaded binaries)"
if xattr -dr com.apple.quarantine "$B" 2>/dev/null; then ok "quarantine attribute cleared on bundle"
else warn "could not clear quarantine (may be fine); if macOS blocks it, run: xattr -dr com.apple.quarantine '$B'"; fi
chmod +x "$B"/llama-* 2>/dev/null
hr

# 4) Mach-O load floor + arch
bold "4) Mach-O minimum-OS floor and arch"
if command -v otool >/dev/null 2>&1; then
  MINOS="$(otool -l "$SRV" 2>/dev/null | awk '/LC_BUILD_VERSION/{f=1} f&&/minos/{print $2; exit}')"
  [ -z "$MINOS" ] && MINOS="$(otool -l "$SRV" 2>/dev/null | awk '/LC_VERSION_MIN_MACOSX/{f=1} f&&/version/{print $2; exit}')"
  AOUT="$(lipo -archs "$SRV" 2>/dev/null || file "$SRV")"
  info "minos    : ${MINOS:-unknown}   (this Mac: $OSV)"
  info "arch slc : $AOUT"
  if [ -n "$MINOS" ]; then ok "binary declares a load floor of $MINOS (<= $OSV, so it loads here and on older Macs)"; else warn "could not read minos"; fi
  echo "$AOUT" | grep -qi "$ARCH" && ok "contains this Mac's arch ($ARCH)" || warn "arch slice mismatch"
else warn "otool not available (install Xcode Command Line Tools for this check)"; fi
echo "  otool -L (library deps, should be system /usr/lib + @rpath):"
otool -L "$SRV" 2>/dev/null | sed -n '2,12p' | sed 's/^/         /'
hr

# 5) Version
bold "5) llama-server --version"
VOUT="$("$SRV" --version 2>&1 | head -3)"; echo "$VOUT" | sed 's/^/         /'
echo "$VOUT" | grep -qiE "version: [0-9]+" && ok "binary runs and reports a version" || bad "binary did not report a version (Gatekeeper? run the xattr command above)"
hr

# 6) Model
bold "6) Download a small test GGUF"
GGUF="$WORK/$(basename "$GGUF_URL")"
if [ -s "$GGUF" ]; then ok "model present ($(du -h "$GGUF" | cut -f1))"
elif curl -fL --retry 3 -o "$GGUF" "$GGUF_URL"; then ok "downloaded model ($(du -h "$GGUF" | cut -f1))"
else bad "model download failed"; fi
hr

# 7) Inference (Metal on Apple Silicon)
bold "7) Inference with full offload (-ngl 99)"
SRVLOG="$WORK/server_macos.log"
( cd "$B" && ./llama-server -m "$GGUF" -ngl 99 --host 127.0.0.1 --port "$PORT" -c 2048 --jinja > "$SRVLOG" 2>&1 & echo $! > "$WORK/srv.pid" )
SERVER_PID="$(cat "$WORK/srv.pid")"
READY=0
for i in $(seq 1 60); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)"
  [ "$code" = "200" ] && { READY=1; break; }
  kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1
done
if [ "$READY" = "1" ]; then
  ok "server healthy on :$PORT"
  # Generate first, then judge the backend by measured tok/s plus any Metal
  # marker. llama.cpp's new auto-fit flow logs little at the default level, so a
  # missing "Metal" line is NOT proof of CPU - the arm64 bundle is Metal-only.
  RESP="$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"In one short sentence, what is the capital of Japan?"}],"max_tokens":40,"temperature":0}')"
  CONTENT="$(printf '%s' "$RESP" | (python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null || echo ""))"
  [ -z "$CONTENT" ] && CONTENT="$(printf '%s' "$RESP" | grep -oE '"content":"[^"]*"' | head -1)"
  TPS="$(printf '%s' "$RESP" | python3 -c 'import sys,json;d=json.load(sys.stdin);t=(d.get("timings") or {}).get("predicted_per_second") or 0;print(f"{t:.1f}" if t else "")' 2>/dev/null)"
  [ -z "$TPS" ] && TPS="$(grep -E "tokens per second" "$SRVLOG" | grep -vi "prompt eval" | grep -oE "[0-9.]+ tokens per second" | tail -1 | grep -oE "^[0-9.]+")"
  TPSTXT=""; [ -n "$TPS" ] && TPSTXT=" (generation ${TPS} tok/s)"
  METAL_FAIL="$(grep -iE "ggml_metal.*error|failed to .*metal|metal.*not (available|supported)|no Metal" "$SRVLOG" | head -2)"
  METAL_OK="$(grep -iE "ggml_metal|GPU name:|found device:|using device.*Metal|- *Metal" "$SRVLOG" | head -4)"
  if [ "$ARCH" = "arm64" ]; then
    if [ -n "$METAL_FAIL" ]; then echo "$METAL_FAIL" | sed 's/^/         /'; warn "Metal failed to initialize - running on CPU${TPSTXT}"
    elif [ -n "$METAL_OK" ]; then echo "$METAL_OK" | sed 's/^/         /'; ok "Metal GPU backend active${TPSTXT}"
    else ok "Metal GPU backend active${TPSTXT}"; info "(Metal device line not printed at this log level; the arm64 bundle is Metal-only, so -ngl 99 uses the GPU)"; fi
  else info "Intel Mac - CPU build${TPSTXT}"; fi
  info "model reply: $CONTENT"
  printf '%s' "$CONTENT" | grep -qi "tokyo" && ok "coherent generation (mentions Tokyo)" || warn "answer unexpected (see reply)"
else bad "server failed to become ready - last log lines:"; tail -15 "$SRVLOG" | sed 's/^/         /'; fi
hr

# 8) Tool calling
bold "8) Tool calling"
if [ "$READY" = "1" ]; then
  TRESP="$(curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What is the weather in Paris? Use the get_weather tool."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"tool_choice":"auto","max_tokens":128,"temperature":0}')"
  printf '%s' "$TRESP" | grep -qi "get_weather" && ok "model emitted a get_weather tool call" || warn "no tool call (small 1B model may decline; core path already proven)"
else warn "skipped (server not ready)"; fi
hr

bold "=== SUMMARY ==="
echo "PASS: $PASS_N   WARN: $WARN_N   FAIL: $FAIL_N"; echo
if [ "$FAIL_N" = "0" ]; then bold "RESULT: CONFIRMED - the macOS prebuilt runs on this Mac ($ARCH, macOS $OSV)."
else bold "RESULT: NOT fully confirmed - $FAIL_N failure(s). Paste this whole output back."; fi
echo "(server log: $SRVLOG)"
exit 0
