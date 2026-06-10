<#
amd_confirm.ps1  (Windows AMD ROCm/HIP prebuilt confirmation)

Confirms the llama.cpp build the Unsloth installer would fetch for an AMD GPU on
Windows actually runs, on the GPU. AMD does NOT use the unslothai bundles - the
installer selects, in order:
  1. lemonade-sdk/llamacpp-rocm  llama-<lemo_tag>-windows-rocm-<gfx_family>-x64.zip
     (when the GPU's gfx arch maps to a lemonade family)
  2. upstream ggml-org           llama-<tag>-bin-win-hip-radeon-x64.zip
     (generic Radeon HIP build, when an AMD GPU is present but gfx is unknown)
  3. upstream CPU                llama-<tag>-bin-win-cpu-x64.zip   (no AMD GPU)
Then it runs real inference + tool calling and reports PASS/FAIL with measured
tok/s (the surest GPU-vs-CPU signal).

Nothing is installed system-wide; files go under $WORK and can be deleted.

Usage:    powershell -ExecutionPolicy Bypass -File amd_confirm.ps1
Env overrides (set first, e.g. $env:AMD_GFX='gfx1100'):
  WORK PORT GGUF_URL KEEP=1
  AMD_GFX        force a gfx arch (skip hipinfo/amd-smi detection)
  LEMONADE_TAG   lemonade release tag (default: latest, ~b1292)
  WIN_TAG        upstream tag for the HIP/CPU fallback (default b9518)
  BUNDLE_URL     force any zip URL
#>

$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'

$WORK     = if ($env:WORK)     { $env:WORK }     else { Join-Path $HOME 'llama_amd_test' }
$PORT     = if ($env:PORT)     { [int]$env:PORT } else { 8147 }
$GGUF_URL = if ($env:GGUF_URL) { $env:GGUF_URL } else { 'https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf' }
$KEEP     = ($env:KEEP -eq '1')
$WIN_TAG  = if ($env:WIN_TAG)  { $env:WIN_TAG }  else { 'b9518' }
$AMD_GFX  = if ($env:AMD_GFX)  { $env:AMD_GFX }  else { '' }
$BUNDLE_URL = if ($env:BUNDLE_URL) { $env:BUNDLE_URL } else { '' }
$LEMONADE_REPO = 'lemonade-sdk/llamacpp-rocm'
$UPSTREAM_REPO = 'ggml-org/llama.cpp'

$script:PASS_N = 0; $script:FAIL_N = 0; $script:WARN_N = 0
function Bold($m){ Write-Host $m -ForegroundColor White }
function Ok($m)  { Write-Host "  [PASS] $m" -ForegroundColor Green;  $script:PASS_N++ }
function Bad($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red;    $script:FAIL_N++ }
function Warn($m){ Write-Host "  [WARN] $m" -ForegroundColor Yellow; $script:WARN_N++ }
function Info($m){ Write-Host "         $m" }
function Hr      { Write-Host '---------------------------------------------------------------' }
$server = $null
function Stop-Server { if ($server -and -not $server.HasExited) { try { $server.Kill() } catch {} } }

function Get-File($url, $out) {
  $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
  if ($curl) { & curl.exe -fL --retry 3 -o $out $url 2>$null; if ($LASTEXITCODE -eq 0 -and (Test-Path $out)) { return $true } }
  try { Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing -ErrorAction Stop; return (Test-Path $out) } catch { return $false }
}
function HSize($p){ if (Test-Path $p){ '{0:N1} MB' -f ((Get-Item $p).Length/1MB) } else { '?' } }
# gfx -> lemonade family (mirrors install_llama_prebuilt.py); specific prefixes first.
function Gfx-Family($g){
  switch -Regex ($g) { '^gfx1151' {'gfx1151';break} '^gfx1150' {'gfx1150';break} '^gfx120' {'gfx120X';break} '^gfx110' {'gfx110X';break} '^gfx103' {'gfx103X';break} default {''} }
}
function Find-Exe($name){
  $c = Get-Command "$name.exe" -ErrorAction SilentlyContinue
  if ($c) { return $c.Source }
  foreach ($e in 'HIP_PATH','ROCM_PATH') { $r = [Environment]::GetEnvironmentVariable($e); if ($r) { $p = Join-Path $r "bin\$name.exe"; if (Test-Path $p) { return $p } } }
  return $null
}

New-Item -ItemType Directory -Force -Path $WORK | Out-Null
Write-Host ''
Bold '=== Unsloth llama.cpp AMD ROCm/HIP prebuilt confirmation (Windows) ==='
Write-Host "scratch dir : $WORK"; Hr

# --------------------------------------------------------------------------- #
# 1. Host + AMD GPU
# --------------------------------------------------------------------------- #
Bold '1) Host detection'
$arch = $env:PROCESSOR_ARCHITECTURE
$os   = try { (Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).Caption } catch { 'Windows' }
Info "OS        : $os"
Info "arch      : $arch"
$haveGpu = $false
try {
  $vc = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -match '(?i)AMD|Radeon' })
  if ($vc.Count -gt 0) { $haveGpu = $true; $vc | ForEach-Object { Info "  - $($_.Name)" }; Ok 'AMD GPU present (Win32_VideoController)' }
  else { Warn 'no AMD GPU listed by Windows - will confirm the CPU build (or set AMD_GFX to force ROCm)' }
} catch { Warn 'could not query video controllers' }
Hr

# --------------------------------------------------------------------------- #
# 2. Detect gfx arch + lemonade family
# --------------------------------------------------------------------------- #
Bold '2) Detect AMD gfx arch'
$gfx = $AMD_GFX
if (-not $gfx) {
  $hip = Find-Exe 'hipinfo'
  if ($hip) { $o = & $hip 2>$null | Out-String; $m = [regex]::Match($o, '(?i)gcnArchName:\s*(gfx[0-9a-z]+)'); if ($m.Success) { $gfx = $m.Groups[1].Value } }
  if (-not $gfx) { $smi = Find-Exe 'amd-smi'; if ($smi) { $o = & $smi list 2>$null | Out-String; $m = [regex]::Match($o, '(?i)(gfx[0-9a-z]{3,})'); if ($m.Success) { $gfx = $m.Groups[1].Value } } }
}
if ($gfx) { $gfx = $gfx.ToLower(); $haveGpu = $true; if ($AMD_GFX) { Info "gfx (forced): $gfx" } else { Info "gfx (hipinfo/amd-smi): $gfx" } }
else { Info 'no gfx detected (hipinfo / amd-smi not found - HIP SDK not installed). Set AMD_GFX=gfxNNNN to force a ROCm build.' }
$family = if ($gfx) { Gfx-Family $gfx } else { '' }
if ($family) { Ok "lemonade family: $family" }
elseif ($gfx) { Warn "gfx '$gfx' is not covered by lemonade ROCm prebuilts - will use the generic win-hip-radeon build" }
Hr

# --------------------------------------------------------------------------- #
# 3. Select + download
# --------------------------------------------------------------------------- #
Bold '3) Select + download the prebuilt'
$cpuBuild = $false; $reason = ''
if ($BUNDLE_URL) {
  $reason = 'forced via BUNDLE_URL'; if ($BUNDLE_URL -match 'win-cpu') { $cpuBuild = $true }
} elseif ($family) {
  $lemoTag = $env:LEMONADE_TAG
  if (-not $lemoTag) { try { $lemoTag = (Invoke-RestMethod "https://api.github.com/repos/$LEMONADE_REPO/releases/latest" -UseBasicParsing).tag_name } catch { $lemoTag = 'b1292' } }
  $BUNDLE_URL = "https://github.com/$LEMONADE_REPO/releases/download/$lemoTag/llama-$lemoTag-windows-rocm-$family-x64.zip"
  $reason = "lemonade ROCm $family @ $lemoTag"
} elseif ($haveGpu) {
  $BUNDLE_URL = "https://github.com/$UPSTREAM_REPO/releases/download/$WIN_TAG/llama-$WIN_TAG-bin-win-hip-radeon-x64.zip"
  $reason = "AMD GPU present, gfx unknown to lemonade - upstream win-hip-radeon @ $WIN_TAG"
} else {
  $cpuBuild = $true
  $BUNDLE_URL = "https://github.com/$UPSTREAM_REPO/releases/download/$WIN_TAG/llama-$WIN_TAG-bin-win-cpu-x64.zip"
  $reason = "no AMD GPU detected - upstream CPU build @ $WIN_TAG"
}
Info "selection : $reason"
Info "bundle    : $BUNDLE_URL"
if (Get-File $BUNDLE_URL (Join-Path $WORK 'bundle.zip')) { Ok "downloaded ($(HSize (Join-Path $WORK 'bundle.zip')))" }
else { Bad 'download failed'; Write-Host ''; Bold 'Cannot continue.'; exit 0 }
Hr

# --------------------------------------------------------------------------- #
# 4. Extract
# --------------------------------------------------------------------------- #
Bold '4) Extract'
$B = Join-Path $WORK 'bundle'
Remove-Item -Recurse -Force $B -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $B | Out-Null
try { Expand-Archive -Path (Join-Path $WORK 'bundle.zip') -DestinationPath $B -Force } catch { Bad "extract failed: $_" }
$server_exe = (Get-ChildItem -Path $B -Recurse -Filter 'llama-server.exe' -ErrorAction SilentlyContinue | Select-Object -First 1)
if ($server_exe) { Ok "extracted (llama-server.exe in $($server_exe.Directory.FullName))" } else { Bad 'llama-server.exe not found in the archive'; Hr; exit 0 }
$bindir = $server_exe.Directory.FullName
Hr

# --------------------------------------------------------------------------- #
# 5. Version
# --------------------------------------------------------------------------- #
Bold '5) llama-server.exe --version'
$vout = & $server_exe.FullName --version 2>&1 | Select-Object -First 3
$vout | ForEach-Object { Info $_ }
if ($vout -match 'version:') { Ok 'binary runs' } else { Bad 'binary did not run' }
Hr

# --------------------------------------------------------------------------- #
# 6. Model
# --------------------------------------------------------------------------- #
Bold '6) Download a small test GGUF'
$gguf = Join-Path $WORK ([IO.Path]::GetFileName(([uri]$GGUF_URL).AbsolutePath))
if ((Test-Path $gguf) -and (Get-Item $gguf).Length -gt 0) { Ok "model present ($(HSize $gguf))" }
elseif (Get-File $GGUF_URL $gguf) { Ok "downloaded model ($(HSize $gguf))" }
else { Bad 'model download failed' }
Hr

# --------------------------------------------------------------------------- #
# 7. Inference with full offload (-ngl 99)
# --------------------------------------------------------------------------- #
Bold '7) Inference with full offload (-ngl 99)'
$ready = $false; $outLog = Join-Path $WORK 'server.out.log'; $errLog = Join-Path $WORK 'server.err.log'
if ($server_exe -and (Test-Path $gguf)) {
  $srvArgs = @('-m', $gguf, '-ngl', '99', '--host', '127.0.0.1', '--port', "$PORT", '-c', '2048', '--jinja')
  $server = Start-Process -FilePath $server_exe.FullName -ArgumentList $srvArgs -PassThru -NoNewWindow -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  for ($i = 0; $i -lt 120; $i++) {
    try { if ((Invoke-WebRequest "http://127.0.0.1:$PORT/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop).StatusCode -eq 200) { $ready = $true; break } } catch {}
    if ($server.HasExited) { break }; Start-Sleep -Seconds 1
  }
  if ($ready) {
    Ok "server healthy on :$PORT"
    $content = ''; $tps = $null
    try {
      $body = @{ messages=@(@{role='user';content='In one short sentence, what is the capital of Japan?'}); max_tokens=40; temperature=0 } | ConvertTo-Json -Depth 6
      $r = Invoke-RestMethod "http://127.0.0.1:$PORT/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 180
      $content = $r.choices[0].message.content
      if (($r.PSObject.Properties.Name -contains 'timings') -and $r.timings.predicted_per_second) { $tps = [math]::Round([double]$r.timings.predicted_per_second, 1) }
    } catch { Warn "generation request failed: $_" }
    $log = @(); if (Test-Path $errLog) { $log += Get-Content $errLog }; if (Test-Path $outLog) { $log += Get-Content $outLog }
    if (-not $tps) { $m = ($log | Select-String -Pattern 'eval time =.*?([\d.]+) tokens per second' | Select-Object -Last 1); if ($m) { $tps = [math]::Round([double]$m.Matches[0].Groups[1].Value, 1) } }
    $tpsTxt = if ($tps) { " (generation $tps tok/s)" } else { "" }
    if ($cpuBuild) {
      Info "CPU build - inference runs on CPU as expected$tpsTxt"
    } else {
      $gpuFail = $log | Select-String -Pattern 'failed to initialize|no ROCm|no HIP|no usable GPU|hipError|ROCm error|no .*device' | Select-Object -First 3
      $gpuDev  = $log | Select-String -Pattern '-\s*(ROCm|HIP|CUDA)\d+\s*:.*MiB|using device (ROCm|HIP)|found \d+ (ROCm|HIP) device' | Select-Object -First 4
      if ($gpuDev -and -not $gpuFail) {
        $gpuDev | ForEach-Object { Info $_.Line }; Ok "ROCm/HIP GPU enumerated and active$tpsTxt"
      } elseif ($gpuFail) {
        $gpuFail | ForEach-Object { Info $_.Line }
        if ($haveGpu) { Bad "an AMD GPU is present but ROCm/HIP failed to initialize - ran on CPU$tpsTxt" }
        else { Warn "ROCm/HIP did not initialize - running on CPU$tpsTxt" }
      } elseif ($haveGpu) {
        Bad "AMD GPU present but llama.cpp did not enumerate a ROCm/HIP device - ran on CPU$tpsTxt"
        ($log | Select-String -Pattern 'rocm|hip|device|fitting|buffer|backend|error' | Select-Object -First 12) | ForEach-Object { Info $_.Line }
      } else {
        Warn "no ROCm/HIP device enumerated - running on CPU$tpsTxt"
      }
    }
    if ($content) { Info "model reply: $content"; if ($content -match '(?i)tokyo') { Ok 'coherent generation (mentions Tokyo)' } else { Warn 'answer unexpected' } }
    else { Warn 'no generation content returned' }
  } else {
    $log = @(); if (Test-Path $errLog) { $log += Get-Content $errLog }; if (Test-Path $outLog) { $log += Get-Content $outLog }
    Bad 'server failed to become ready:'; ($log | Select-Object -Last 15) | ForEach-Object { Info $_ }
  }
} else { Bad 'skipped (no exe or model)' }
Hr

# --------------------------------------------------------------------------- #
# 8. Tool calling
# --------------------------------------------------------------------------- #
Bold '8) Tool calling'
if ($ready) {
  try {
    $tool = @{ messages=@(@{role='user';content='What is the weather in Paris? Use the get_weather tool.'}); tools=@(@{ type='function'; function=@{ name='get_weather'; description='Get weather'; parameters=@{ type='object'; properties=@{ location=@{ type='string' } }; required=@('location') } } }); tool_choice='auto'; max_tokens=128; temperature=0 } | ConvertTo-Json -Depth 12
    $tr = Invoke-RestMethod "http://127.0.0.1:$PORT/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $tool -TimeoutSec 180
    if (($tr | ConvertTo-Json -Depth 12) -match 'get_weather') { Ok 'model emitted a get_weather tool call' } else { Warn 'no tool call (small model may decline)' }
  } catch { Warn "tool-call request failed: $_" }
} else { Warn 'skipped (server not ready)' }
Hr

Stop-Server
if (-not $KEEP) { Remove-Item -Force (Join-Path $WORK 'bundle.zip') -ErrorAction SilentlyContinue }
Bold '=== SUMMARY ==='
Write-Host "host: $arch   build: $(if($BUNDLE_URL){[IO.Path]::GetFileName(([uri]$BUNDLE_URL).AbsolutePath)}else{'none'})"
Write-Host "PASS: $script:PASS_N   WARN: $script:WARN_N   FAIL: $script:FAIL_N"
Write-Host ''
if ($script:FAIL_N -eq 0) {
  if ($cpuBuild) { Bold 'RESULT: CONFIRMED - CPU prebuilt runs on this box (no AMD GPU used).' }
  else { Bold 'RESULT: CONFIRMED - AMD ROCm/HIP prebuilt runs on this box.' }
} else { Bold "RESULT: $script:FAIL_N hard failure(s) - paste this whole output back." }
Write-Host "(server logs: $errLog ; $outLog)"
exit 0
