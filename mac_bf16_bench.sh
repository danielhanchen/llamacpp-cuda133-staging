#!/usr/bin/env bash
#
# mac_bf16_bench.sh
# A/B the GGML_METAL_USE_BF16 build flag on THIS Mac. Both bundles are the same
# llama.cpp (b9518, macos-26 SDK, load floor 14.0); the ONLY difference is the
# BF16 Metal kernels:
#   app-b9518-m26      -> GGML_METAL_USE_BF16=ON
#   app-b9518-nobf16   -> GGML_METAL_USE_BF16=OFF
#
# Runs the same prompt-processing (pp) + generation (tg) benchmark for:
#   1. BF16 model + flag ON   (uses the bf16 kernels)
#   2. BF16 model + flag OFF   (bf16 kernels absent -> f16 path)
#   3. F16  model + flag ON    (reference baseline)
#
# On M1/M2 (no native bf16 GPU ALUs) #1 is expected to be SLOWER than #2/#3;
# on M3+ (native bf16) all three should be ~equal. Nothing installs
# system-wide; scratch lives under $WORK.
#
# Usage:   bash mac_bf16_bench.sh
# Env:     WORK PORT NGEN(=256) REPEATS(=3) KEEP=1 MODEL_BF16_URL MODEL_F16_URL
#
set -uo pipefail

WORK="${WORK:-$HOME/llama_bf16_test}"
PORT="${PORT:-8160}"
NGEN="${NGEN:-256}"
REPEATS="${REPEATS:-3}"
KEEP="${KEEP:-0}"
REL="https://github.com/danielhanchen/llamacpp-cuda133-staging/releases/download/m5-tensor-test"
BUNDLE_ON="$REL/app-b9518-m26-macos-arm64.tar.gz"
BUNDLE_OFF="$REL/app-b9518-nobf16-macos-arm64.tar.gz"
HF="https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main"
MODEL_BF16_URL="${MODEL_BF16_URL:-$HF/Llama-3.2-1B-Instruct-BF16.gguf}"
MODEL_F16_URL="${MODEL_F16_URL:-$HF/Llama-3.2-1B-Instruct-F16.gguf}"

bold(){ printf '\033[1m%s\033[0m\n' "$*"; }
hr(){ printf -- '---------------------------------------------------------------\n'; }
SERVER_PID=""
cleanup(){ [ -n "$SERVER_PID" ] && kill "$SERVER_PID" >/dev/null 2>&1; }
trap cleanup EXIT

mkdir -p "$WORK"; cd "$WORK" || { echo "cannot cd $WORK"; exit 1; }
ARCH="$(uname -m)"; CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"; OSV="$(sw_vers -productVersion 2>/dev/null)"
echo; bold "=== GGML_METAL_USE_BF16 A/B bench ==="
echo "host : $CHIP   macOS $OSV   ($ARCH)   gen=$NGEN repeats=$REPEATS"
hr

# ---- fetch helpers (cache by filename) ----
fetch(){ # url -> path (download once)
  local url="$1" out="$WORK/$(basename "$1")"
  if [ -s "$out" ]; then echo "$out"; return; fi
  curl -fL --retry 3 -o "$out" "$url" >/dev/null 2>&1 || { echo ""; return; }
  echo "$out"
}
extract_bundle(){ # url -> bin dir (download+extract once)
  local url="$1" tag; tag="$(basename "$url" .tar.gz)"
  local dir="$WORK/x_$tag"
  if [ ! -x "$dir/.ok" ] 2>/dev/null; then :; fi
  if [ ! -f "$dir/.done" ]; then
    rm -rf "$dir"; mkdir -p "$dir"
    if ! curl -fL --retry 3 -o "$dir/b.tgz" "$url" >/dev/null 2>&1; then echo ""; return; fi
    tar -xzf "$dir/b.tgz" -C "$dir" 2>/dev/null && touch "$dir/.done"
  fi
  local srv; srv="$(find "$dir" -type f -name llama-server | head -1)"
  [ -z "$srv" ] && { echo ""; return; }
  local b; b="$(dirname "$srv")"; xattr -dr com.apple.quarantine "$b" 2>/dev/null; chmod +x "$b"/llama-* 2>/dev/null
  echo "$b"
}

bench_one(){ # url n promptfile -> "pp tg"
  python3 - "$1" "$2" "$3" <<'PY'
import json,urllib.request,sys
url,n,pf=sys.argv[1],int(sys.argv[2]),sys.argv[3]
body=json.dumps({"prompt":open(pf).read(),"n_predict":n,"cache_prompt":False,"temperature":0,"top_k":1}).encode()
try:
    r=json.load(urllib.request.urlopen(urllib.request.Request(url+"/completion",data=body,headers={"Content-Type":"application/json"}),timeout=900))
except Exception:
    print("ERR ERR"); sys.exit(0)
t=r.get("timings",{}); print(f"{t.get('prompt_per_second',0):.1f} {t.get('predicted_per_second',0):.1f}")
PY
}

PFILE="$WORK/prompt_long.txt"
yes 'The quick brown fox writes a detailed technical report about GPU kernels, matrix multiplication throughput, and memory bandwidth on Apple Silicon. ' 2>/dev/null | head -50 | tr -d '\n' > "$PFILE"

RESULTS="$WORK/results.tsv"; : > "$RESULTS"
# label|bundle_url|model_url
RUNS=(
  "BF16 model + flag ON|$BUNDLE_ON|$MODEL_BF16_URL"
  "BF16 model + flag OFF|$BUNDLE_OFF|$MODEL_BF16_URL"
  "F16 model (reference)|$BUNDLE_ON|$MODEL_F16_URL"
)

for entry in "${RUNS[@]}"; do
  label="${entry%%|*}"; rest="${entry#*|}"; burl="${rest%%|*}"; murl="${rest#*|}"
  bold ">>> $label"
  bdir="$(extract_bundle "$burl")"
  [ -z "$bdir" ] && { echo "  bundle unavailable: $(basename "$burl") (built yet?)"; printf '%s\tERR\tERR\n' "$label" >>"$RESULTS"; hr; continue; }
  model="$(fetch "$murl")"
  [ -z "$model" ] && { echo "  model download failed"; printf '%s\tERR\tERR\n' "$label" >>"$RESULTS"; hr; continue; }
  log="$WORK/srv_$(echo "$label" | tr -dc 'A-Za-z0-9').log"
  ( cd "$bdir" && ./llama-server -m "$model" -ngl 99 --host 127.0.0.1 --port "$PORT" -c 4096 --cache-ram 0 >"$log" 2>&1 & echo $! >"$WORK/pid" )
  SERVER_PID="$(cat "$WORK/pid")"; ready=0
  for i in $(seq 1 180); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && { ready=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1
  done
  [ "$ready" != 1 ] && { echo "  server failed:"; tail -4 "$log" | sed 's/^/    /'; kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""; printf '%s\tERR\tERR\n' "$label" >>"$RESULTS"; hr; continue; }
  bf="$(grep -oiE "use bfloat = (true|false)|has bfloat = (true|false)" "$log" | tr '\n' ' ')"; [ -n "$bf" ] && echo "  metal: $bf"
  bench_one "http://127.0.0.1:$PORT" 8 "$PFILE" >/dev/null  # warmup
  bestpp=0 besttg=0
  for r in $(seq 1 "$REPEATS"); do
    out="$(bench_one "http://127.0.0.1:$PORT" "$NGEN" "$PFILE")"; pp="${out%% *}"; tg="${out##* }"
    [ "$pp" = "ERR" ] && { echo "  request error"; continue; }
    awk -v a="$pp" -v b="$bestpp" 'BEGIN{exit !(a>b)}' && bestpp="$pp"
    awk -v a="$tg" -v b="$besttg" 'BEGIN{exit !(a>b)}' && besttg="$tg"
    printf '    run %s: pp=%s  tg=%s tok/s\n' "$r" "$pp" "$tg"
  done
  kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""; sleep 1
  printf '%s\t%s\t%s\n' "$label" "$bestpp" "$besttg" >>"$RESULTS"
  hr
done

bold "=== RESULTS (best of $REPEATS, tokens/sec) ==="
printf '%-26s %12s %10s\n' "CONFIG" "PROMPT(pp)" "GEN(tg)"
awk -F'\t' '{printf "%-26s %12s %10s\n",$1,$2,$3}' "$RESULTS"
echo
bold "Read:"
echo "  Compare row 1 (BF16+ON) vs row 2 (BF16+OFF), same model -> isolates the BF16 flag."
echo "  M1/M2: expect ON slower than OFF. M3+: expect ~equal (native bf16)."
[ "$KEEP" = 1 ] || { rm -rf "$WORK"/x_* 2>/dev/null; echo "(removed bundles; kept models. KEEP=1 to keep all; rm -rf $WORK to wipe)"; }
