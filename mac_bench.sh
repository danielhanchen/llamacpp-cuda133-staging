#!/usr/bin/env bash
#
# mac_bench.sh
# Head-to-head throughput of the three arm64 macOS llama.cpp builds on THIS Mac:
#   1. unsloth-cur   - built on macos-14 (old SDK),   load floor 14.0
#   2. unsloth-m26   - built on macos-26 (Metal 4 SDK), load floor 14.0  (the experiment)
#   3. official-ggml - built on macos-26 (Metal 4 SDK), load floor 26
#
# For each: download + extract the bundle, start llama-server with full GPU
# offload, run the SAME prompt-processing (pp) and generation (tg) benchmark,
# and report tokens/sec. Also reports whether the Metal 4 tensor API engaged
# (looks for the "error compiling source" / "tensor API ... disabling" lines).
# The model is downloaded once and reused. Nothing installs system-wide; all
# scratch lives under $WORK and can be deleted afterward.
#
# Usage:   bash mac_bench.sh
# Env:     WORK, PORT, GGUF_URL, NGEN (gen tokens, default 256),
#          REPEATS (default 3), KEEP=1 (keep model+bundles)
#
set -uo pipefail

WORK="${WORK:-$HOME/llama_bench_test}"
PORT="${PORT:-8150}"
NGEN="${NGEN:-256}"
REPEATS="${REPEATS:-3}"
KEEP="${KEEP:-0}"
# 8B Q4_K_M: big enough that the tensor-API matmul speedup (if any) is visible.
# Override with a smaller model via GGUF_URL if bandwidth is tight.
GGUF_URL="${GGUF_URL:-https://huggingface.co/unsloth/Llama-3.1-8B-Instruct-GGUF/resolve/main/Llama-3.1-8B-Instruct-Q4_K_M.gguf}"

S="spark-test-b9518"; M="m5-tensor-test"
BASE="https://github.com/danielhanchen/llamacpp-cuda133-staging/releases/download"
GGML="https://github.com/ggml-org/llama.cpp/releases/download/b9518"
# label|url   (kept in one place so it is easy to add/remove a build)
BUNDLES=(
  "unsloth-cur(macos14,floor14)|$BASE/$S/app-b9518-macos-arm64.tar.gz"
  "unsloth-m26(macos26,floor14)|$BASE/$M/app-b9518-m26-macos-arm64.tar.gz"
  "official-ggml(macos26,floor26)|$GGML/llama-b9518-bin-macos-arm64.tar.gz"
)

bold(){ printf '\033[1m%s\033[0m\n' "$*"; }
hr(){ printf -- '---------------------------------------------------------------\n'; }
SERVER_PID=""
cleanup(){ [ -n "$SERVER_PID" ] && kill "$SERVER_PID" >/dev/null 2>&1; }
trap cleanup EXIT

mkdir -p "$WORK"; cd "$WORK" || { echo "cannot cd $WORK"; exit 1; }
ARCH="$(uname -m)"; CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
OSV="$(sw_vers -productVersion 2>/dev/null)"
echo; bold "=== Unsloth llama.cpp macOS throughput bench ==="
echo "host : $CHIP   macOS $OSV   ($ARCH)"
echo "model: $(basename "$GGUF_URL")   gen=$NGEN  repeats=$REPEATS"
if [ "$ARCH" != "arm64" ]; then echo "NOTE: not arm64 - Metal/tensor-API comparison only meaningful on Apple Silicon"; fi
hr

# ---- model (download once, reuse) ----
GGUF="$WORK/$(basename "$GGUF_URL")"
if [ -s "$GGUF" ]; then echo "model present: $(du -h "$GGUF" | cut -f1)"
else echo "downloading model ..."; curl -fL --retry 3 -o "$GGUF" "$GGUF_URL" || { echo "model download failed"; exit 1; }; fi

# ---- a long prompt (~1k tokens) so prompt-processing is a real matmul workload ----
PFILE="$WORK/prompt_long.txt"
yes 'The quick brown fox writes a detailed technical report about distributed systems, GPU kernels, matrix multiplication throughput, and memory bandwidth on Apple Silicon. ' 2>/dev/null | head -50 | tr -d '\n' > "$PFILE"

# bench_one <server_url> <n_predict> <prompt_file>  -> "pp_tps tg_tps"  (via /completion timings)
bench_one(){
  python3 - "$1" "$2" "$3" <<'PY'
import json,urllib.request,sys
url,n,pf=sys.argv[1],int(sys.argv[2]),sys.argv[3]
prompt=open(pf).read()
body=json.dumps({"prompt":prompt,"n_predict":n,"cache_prompt":False,"temperature":0,"top_k":1}).encode()
req=urllib.request.Request(url+"/completion",data=body,headers={"Content-Type":"application/json"})
try:
    r=json.load(urllib.request.urlopen(req,timeout=900))
except Exception as e:
    print("ERR ERR"); sys.exit(0)
t=r.get("timings",{})
print(f"{t.get('prompt_per_second',0):.1f} {t.get('predicted_per_second',0):.1f}")
PY
}

RESULTS="$WORK/results.tsv"; : > "$RESULTS"

run_bundle(){
  local label="$1" url="$2" dir="$WORK/b_$(echo "$label" | tr -dc 'a-z0-9')"
  bold ">>> $label"
  rm -rf "$dir"; mkdir -p "$dir"
  if ! curl -fL --retry 3 -o "$dir/b.tgz" "$url"; then echo "  download FAILED"; printf '%s\tERR\tERR\tERR\n' "$label" >>"$RESULTS"; return; fi
  tar -xzf "$dir/b.tgz" -C "$dir" || { echo "  extract FAILED"; printf '%s\tERR\tERR\tERR\n' "$label" >>"$RESULTS"; return; }
  local srv; srv="$(find "$dir" -type f -name llama-server | head -1)"
  [ -z "$srv" ] && { echo "  llama-server not found"; printf '%s\tERR\tERR\tERR\n' "$label" >>"$RESULTS"; return; }
  local bdir; bdir="$(dirname "$srv")"
  xattr -dr com.apple.quarantine "$bdir" 2>/dev/null; chmod +x "$bdir"/llama-* 2>/dev/null

  local log="$dir/server.log"
  ( cd "$bdir" && ./llama-server -m "$GGUF" -ngl 99 --host 127.0.0.1 --port "$PORT" -c 4096 --cache-ram 0 >"$log" 2>&1 & echo $! >"$dir/pid" )
  SERVER_PID="$(cat "$dir/pid")"
  local ready=0 i
  for i in $(seq 1 180); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)" = "200" ] && { ready=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1
  done
  if [ "$ready" != 1 ]; then echo "  server failed to start"; tail -5 "$log" | sed 's/^/    /'; kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""; printf '%s\tERR\tERR\tERR\n' "$label" >>"$RESULTS"; return; fi

  # tensor-API status from the load log
  local tapi="on"
  if grep -qiE "error compiling source|tensor API.*disabling|not supported in this environment" "$log"; then tapi="off"; fi
  echo "  tensor API: $tapi    (Metal device: $(grep -oE 'MTL[0-9].*' "$log" | head -1))"

  bench_one "http://127.0.0.1:$PORT" 8 "$PFILE" >/dev/null   # warmup
  local bestpp=0 besttg=0 r out pp tg
  for r in $(seq 1 "$REPEATS"); do
    out="$(bench_one "http://127.0.0.1:$PORT" "$NGEN" "$PFILE")"
    pp="$(echo "$out" | awk '{print $1}')"; tg="$(echo "$out" | awk '{print $2}')"
    [ "$pp" = "ERR" ] && { echo "  request error"; continue; }
    awk -v a="$pp" -v b="$bestpp" 'BEGIN{exit !(a>b)}' && bestpp="$pp"
    awk -v a="$tg" -v b="$besttg" 'BEGIN{exit !(a>b)}' && besttg="$tg"
    printf '    run %s: pp=%s tok/s   tg=%s tok/s\n' "$r" "$pp" "$tg"
  done
  kill "$SERVER_PID" 2>/dev/null; SERVER_PID=""; sleep 1
  printf '%s\t%s\t%s\t%s\n' "$label" "$tapi" "$bestpp" "$besttg" >>"$RESULTS"
}

for entry in "${BUNDLES[@]}"; do run_bundle "${entry%%|*}" "${entry#*|}"; hr; done

bold "=== RESULTS (best of $REPEATS, tokens/sec) ==="
printf '%-34s %-10s %14s %12s\n' "BUNDLE" "TENSOR_API" "PROMPT(pp)" "GEN(tg)"
awk -F'\t' '{printf "%-34s %-10s %14s %12s\n",$1,$2,$3,$4}' "$RESULTS"
echo
bold "Fastest prompt-processing:"; sort -t$'\t' -k3 -gr "$RESULTS" | head -1 | awk -F'\t' '{printf "  %s  (%s pp tok/s)\n",$1,$3}'
bold "Fastest generation:";        sort -t$'\t' -k4 -gr "$RESULTS" | head -1 | awk -F'\t' '{printf "  %s  (%s tg tok/s)\n",$1,$4}'
echo
[ "$KEEP" = 1 ] || { rm -rf "$WORK"/b_* "$WORK"/b.tgz 2>/dev/null; echo "(removed extracted bundles; kept model. set KEEP=1 to keep everything; rm -rf $WORK to wipe)"; }
echo "logs: $WORK/b_*/server.log"
