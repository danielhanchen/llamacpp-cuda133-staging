#!/usr/bin/env bash
#
# mac_tensor_bench.sh
# A/B the Metal 4 tensor API on THIS Mac. Both bundles are the same llama.cpp
# (b9518, macos-26 SDK, USE_BF16=ON); the ONLY difference is the deployment
# floor, which gates whether the runtime can request Metal 4 / the tensor API:
#   app-b9518-m26  -> deploy floor 14.0  (tensor API OFF: error compiling source)
#   app-b9518-f26  -> deploy floor 26.0  (tensor API ON, M5/A19 + macOS 26 only)
#
# Best run on an M5 (Apple9+/A19, the only chip with the tensor API). On M1-M4
# both should be ~equal (no tensor hardware). f26 only loads on macOS 26+.
#
# Usage:   bash mac_tensor_bench.sh
# Env:     WORK PORT NGEN(=256) REPEATS(=3) KEEP=1 GGUF_URL
#
set -uo pipefail

WORK="${WORK:-$HOME/llama_tensor_test}"
PORT="${PORT:-8170}"
NGEN="${NGEN:-256}"
REPEATS="${REPEATS:-3}"
KEEP="${KEEP:-0}"
REL="https://github.com/danielhanchen/llamacpp-cuda133-staging/releases/download/m5-tensor-test"
# 8B Q4_K_M: prompt-processing is a real matmul workload where the tensor API
# (if engaged) should show. Override with GGUF_URL for a smaller model.
GGUF_URL="${GGUF_URL:-https://huggingface.co/unsloth/Llama-3.1-8B-Instruct-GGUF/resolve/main/Llama-3.1-8B-Instruct-Q4_K_M.gguf}"

bold(){ printf '\033[1m%s\033[0m\n' "$*"; }
hr(){ printf -- '---------------------------------------------------------------\n'; }
SERVER_PID=""
cleanup(){ [ -n "$SERVER_PID" ] && kill "$SERVER_PID" >/dev/null 2>&1; }
trap cleanup EXIT

mkdir -p "$WORK"; cd "$WORK" || { echo "cannot cd $WORK"; exit 1; }
ARCH="$(uname -m)"; CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"; OSV="$(sw_vers -productVersion 2>/dev/null)"
echo; bold "=== Metal 4 tensor-API A/B bench ==="
echo "host : $CHIP   macOS $OSV   ($ARCH)   gen=$NGEN repeats=$REPEATS"
[ "$ARCH" != "arm64" ] && { echo "needs Apple Silicon"; exit 1; }
hr

GGUF="$WORK/$(basename "$GGUF_URL")"
if [ -s "$GGUF" ]; then echo "model present: $(du -h "$GGUF" | cut -f1)"
else echo "downloading model ..."; curl -fL --retry 3 -o "$GGUF" "$GGUF_URL" || { echo "model download failed"; exit 1; }; fi
PFILE="$WORK/prompt_long.txt"
yes 'The quick brown fox writes a detailed technical report about GPU kernels, matrix multiplication throughput, and memory bandwidth on Apple Silicon. ' 2>/dev/null | head -50 | tr -d '\n' > "$PFILE"

bench_one(){
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

RESULTS="$WORK/results.tsv"; : > "$RESULTS"
# label|asset
RUNS=(
  "floor14 (tensor OFF)|app-b9518-m26-macos-arm64.tar.gz"
  "floor26 (tensor ON)|app-b9518-f26-macos-arm64.tar.gz"
)
for entry in "${RUNS[@]}"; do
  label="${entry%%|*}"; asset="${entry#*|}"; dir="$WORK/x_$(echo "$asset" | tr -dc 'a-z0-9')"
  bold ">>> $label  ($asset)"
  rm -rf "$dir"; mkdir -p "$dir"
  if ! curl -fL --retry 3 -o "$dir/b.tgz" "$REL/$asset" >/dev/null 2>&1; then echo "  bundle unavailable (built yet?)"; printf '%s\tERR\tERR\tERR\n' "$label" >>"$RESULTS"; hr; continue; fi
  tar -xzf "$dir/b.tgz" -C "$dir" 2>/dev/null
  srv="$(find "$dir" -type f -name llama-server | head -1)"; [ -z "$srv" ] && { echo "  no llama-server"; printf '%s\tERR\tERR\tERR\n' "$label" >>"$RESULTS"; hr; continue; }
  b="$(dirname "$srv")"; xattr -dr com.apple.quarantine "$b" 2>/dev/null; chmod +x "$b"/llama-* 2>/dev/null
  log="$dir/server.log"
  ( cd "$b" && ./llama-server -m "$GGUF" -ngl 99 --host 127.0.0.1 --port "$PORT" -c 4096 --cache-ram 0 >"$log" 2>&1 & echo $! >"$dir/pid" )
  SERVER_PID="$(cat "$dir/pid")"; ready=0
  for i in $(seq 1 180); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && { ready=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1
  done
  [ "$ready" != 1 ] && { echo "  server failed:"; tail -4 "$log" | sed 's/^/    /'; kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""; printf '%s\tERR\tERR\tERR\n' "$label" >>"$RESULTS"; hr; continue; }
  if grep -qiE "error compiling source|tensor API.*disabling|not supported in this environment" "$log"; then tapi="off"; else tapi="on"; fi
  echo "  tensor API: $tapi"
  bench_one "http://127.0.0.1:$PORT" 8 "$PFILE" >/dev/null
  bestpp=0 besttg=0
  for r in $(seq 1 "$REPEATS"); do
    out="$(bench_one "http://127.0.0.1:$PORT" "$NGEN" "$PFILE")"; pp="${out%% *}"; tg="${out##* }"
    [ "$pp" = "ERR" ] && { echo "  request error"; continue; }
    awk -v a="$pp" -v b="$bestpp" 'BEGIN{exit !(a>b)}' && bestpp="$pp"
    awk -v a="$tg" -v b="$besttg" 'BEGIN{exit !(a>b)}' && besttg="$tg"
    printf '    run %s: pp=%s  tg=%s tok/s\n' "$r" "$pp" "$tg"
  done
  kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""; sleep 1
  printf '%s\t%s\t%s\t%s\n' "$label" "$tapi" "$bestpp" "$besttg" >>"$RESULTS"
  hr
done

bold "=== RESULTS (best of $REPEATS, tokens/sec) ==="
printf '%-22s %-11s %12s %10s\n' "BUILD" "TENSOR_API" "PROMPT(pp)" "GEN(tg)"
awk -F'\t' '{printf "%-22s %-11s %12s %10s\n",$1,$2,$3,$4}' "$RESULTS"
echo
pp14=$(awk -F'\t' '/floor14/{print $3}' "$RESULTS"); pp26=$(awk -F'\t' '/floor26/{print $3}' "$RESULTS")
if [ -n "${pp14:-}" ] && [ -n "${pp26:-}" ] && [ "$pp14" != "ERR" ] && [ "$pp26" != "ERR" ]; then
  awk -v a="$pp14" -v b="$pp26" 'BEGIN{ if(a>0) printf "Tensor-API prompt-processing delta: %+.1f%% (floor26 vs floor14)\n", (b-a)/a*100 }'
fi
bold "Read: a clear floor26 pp win = the Metal 4 tensor API helps -> worth a separate macOS-26 slice. ~equal = not worth it."
[ "$KEEP" = 1 ] || { rm -rf "$WORK"/x_* 2>/dev/null; echo "(removed bundles; kept model. rm -rf $WORK to wipe)"; }
