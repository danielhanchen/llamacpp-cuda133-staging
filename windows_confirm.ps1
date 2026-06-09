<#
windows_confirm.ps1  (native Windows NVIDIA CUDA prebuilt confirmation)

Confirms that the llama.cpp Windows CUDA prebuilt the Unsloth installer would
fetch actually runs on this box, on the GPU. On Windows the installer pulls the
UPSTREAM ggml-org build (not the unslothai Linux bundles): the binaries archive
  llama-<tag>-bin-win-cuda-<rt>-x64.zip
plus the paired CUDA runtime archive
  cudart-llama-bin-win-cuda-<rt>-x64.zip
extracted side by side, so no system CUDA toolkit is needed on PATH. This script
detects your installed CUDA runtime / driver, selects the matching upstream
build the same way the installer does (cuda-13.3 or cuda-12.4, with the b9360
cuda-13.1 pin for Blackwell on a 13.0-13.2 driver), then runs real GPU inference
plus tool calling and prints a PASS/FAIL report.

Nothing is installed system-wide and the GPU driver is never touched. Files go
under $WORK (default $HOME\llama_prebuilt_test) and can be deleted afterwards.

Usage:    powershell -ExecutionPolicy Bypass -File windows_confirm.ps1
          pwsh -File windows_confirm.ps1

Env overrides (set before running, e.g. $env:WIN_TAG='b9510'):
  WORK         scratch dir (default $HOME\llama_prebuilt_test)
  WIN_REPO     upstream release repo (default ggml-org/llama.cpp)
  WIN_TAG      release tag           (default b9518)
  WIN_RUNTIME  force runtime version (e.g. 13.3 / 12.4 / 13.1) - skips detection
  LLAMA_URL    force the binaries zip URL (BUNDLE_URL is accepted as an alias)
  CUDART_URL   force the paired cudart zip URL (auto-derived from LLAMA_URL if unset)
  GGUF_URL     test model (default Llama-3.2-1B Q4_K_M)
  PORT         server port (default 8137)
  KEEP         '1' to keep the downloaded zips/model on exit
#>

$ErrorActionPreference = 'Continue'
$ProgressPreference    = 'SilentlyContinue'   # keep Invoke-WebRequest fast/quiet

# --------------------------------------------------------------------------- #
# Config (env-overridable)
# --------------------------------------------------------------------------- #
$WORK     = if ($env:WORK)     { $env:WORK }     else { Join-Path $HOME 'llama_prebuilt_test' }
$PORT     = if ($env:PORT)     { [int]$env:PORT } else { 8137 }
$REPO     = if ($env:WIN_REPO) { $env:WIN_REPO } else { 'ggml-org/llama.cpp' }
$TAG      = if ($env:WIN_TAG)  { $env:WIN_TAG }  else { 'b9518' }
$GGUF_URL = if ($env:GGUF_URL) { $env:GGUF_URL } else { 'https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf' }
$KEEP     = ($env:KEEP -eq '1')
$LLAMA_URL  = if ($env:LLAMA_URL) { $env:LLAMA_URL } elseif ($env:BUNDLE_URL) { $env:BUNDLE_URL } else { '' }
$CUDART_URL = if ($env:CUDART_URL) { $env:CUDART_URL } else { '' }

# Pinned Blackwell fallback (matches the installer): upstream's cuda-13.3 needs a
# 13.3 driver, so a Blackwell card on a 13.0-13.2 driver is gated off it; b9360
# ships a native sm_120 cuda-13.1 build that runs on a 13.0+ r580 driver.
$BL_TAG = 'b9360'; $BL_RT = '13.1'

# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
$script:PASS_N = 0; $script:FAIL_N = 0; $script:WARN_N = 0
function Bold($m){ Write-Host $m -ForegroundColor White }
function Ok($m)  { Write-Host "  [PASS] $m" -ForegroundColor Green; $script:PASS_N++ }
function Bad($m) { Write-Host "  [FAIL] $m" -ForegroundColor Red;   $script:FAIL_N++ }
function Warn($m){ Write-Host "  [WARN] $m" -ForegroundColor Yellow;$script:WARN_N++ }
function Info($m){ Write-Host "         $m" }
function Hr      { Write-Host '---------------------------------------------------------------' }

$server = $null
function Stop-Server { if ($server -and -not $server.HasExited) { try { $server.Kill() } catch {} } }

# Download $url -> $out. Prefer curl.exe (fast); fall back to Invoke-WebRequest.
function Get-File($url, $out) {
  $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
  if ($curl) {
    & curl.exe -fL --retry 3 -o $out $url 2>$null
    if ($LASTEXITCODE -eq 0 -and (Test-Path $out)) { return $true }
  }
  try { Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing -ErrorAction Stop; return (Test-Path $out) }
  catch { return $false }
}

function HSize($path){ if (Test-Path $path){ '{0:N1} MB' -f ((Get-Item $path).Length/1MB) } else { '?' } }

New-Item -ItemType Directory -Force -Path $WORK | Out-Null
# All file paths below are absolute (Join-Path $WORK ...), so we deliberately do
# NOT Set-Location into $WORK - that would leak the directory change back to the
# caller's shell.
Write-Host ''
Bold '=== Unsloth llama.cpp CUDA prebuilt confirmation (native Windows) ==='
Write-Host "scratch dir : $WORK"; Hr

# --------------------------------------------------------------------------- #
# 1. Host
# --------------------------------------------------------------------------- #
Bold '1) Host detection'
$arch = $env:PROCESSOR_ARCHITECTURE
$os   = try { (Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).Caption } catch { 'Windows' }
Info "OS        : $os"
Info "arch      : $arch"
if ($arch -match 'ARM64') { Warn 'Windows on ARM64 - upstream ships no win-cuda ARM64 build (CPU only); this script targets x64 CUDA' }
elseif ($arch -match 'AMD64|x86_64') { Ok 'x64 Windows (RTX 30xx / 40xx / 50xx / data-center NVIDIA)' }
else { Warn "unexpected arch '$arch' - continuing" }

# nvidia-smi: GPUs, driver, and the driver's max CUDA version (NOT the runtime).
$driverMaj = $null; $driverMin = $null; $caps = @()
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) { $sysSmi = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'; if (Test-Path $sysSmi) { $smi = $sysSmi } }
if ($smi) {
  $drv = (& $smi --query-gpu=driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
  $hdr = (& $smi 2>$null | Select-String -Pattern 'CUDA Version:\s*([0-9.]+)' | Select-Object -First 1)
  $cudaVer = if ($hdr) { $hdr.Matches[0].Groups[1].Value } else { '' }
  if ($cudaVer -match '^(\d+)\.(\d+)') { $driverMaj = [int]$Matches[1]; $driverMin = [int]$Matches[2] }
  Info "driver    : $drv   (driver max CUDA: $cudaVer)"
  $caps = @(& $smi --query-gpu=compute_cap --format=csv,noheader 2>$null | ForEach-Object { $_.Trim() } | Where-Object { $_ })
  $names = @(& $smi --query-gpu=index,name,compute_cap --format=csv,noheader 2>$null)
  if ($names) { Info 'GPU(s)    :'; $names | ForEach-Object { Info "  - $_" } }
  if ($caps) { Ok 'GPU(s) visible via nvidia-smi' } else { Warn 'nvidia-smi ran but listed no GPUs' }
  Info 'note: the driver CUDA version is the max it supports, not the installed runtime - the build is chosen below'
} else {
  Warn 'nvidia-smi not found - no NVIDIA GPU detected; will confirm the CPU build (install the driver if you do have a GPU)'
}
$blackwell = @($caps | Where-Object { [double]($_ -replace '[^0-9.]','') -ge 12.0 }).Count -gt 0
Hr

# --------------------------------------------------------------------------- #
# 2. Detect installed CUDA runtime (cudart64_NN.dll)
# --------------------------------------------------------------------------- #
Bold '2) Detect installed CUDA runtime (cudart64_<major>.dll)'
# Same search set the installer uses: PATH, CUDA_PATH/HOME/ROOT, the toolkit dir,
# and PyTorch / pip nvidia-* wheels (which ship cudart/cublas).
$dllDirs = New-Object System.Collections.Generic.List[string]
foreach ($p in @($env:CUDA_RUNTIME_DLL_DIR, $env:PATH) -join ';' -split ';') { if ($p) { $dllDirs.Add($p) } }
foreach ($n in 'CUDA_PATH','CUDA_HOME','CUDA_ROOT') {
  $v = [Environment]::GetEnvironmentVariable($n); if ($v) { $dllDirs.Add((Join-Path $v 'bin')); $dllDirs.Add((Join-Path $v 'lib\x64')) }
}
$tk = Join-Path ${env:ProgramFiles} 'NVIDIA GPU Computing Toolkit\CUDA'
if (Test-Path $tk) { Get-ChildItem $tk -Directory -Filter 'v*' -ErrorAction SilentlyContinue | ForEach-Object { $dllDirs.Add((Join-Path $_.FullName 'bin')) } }
# torch / pip nvidia wheels
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python3 -ErrorAction SilentlyContinue }
if ($py) {
  $pyout = & $py.Source -c @'
import importlib.util, os
seen=set()
for mod in ("torch","nvidia.cuda_runtime","nvidia.cublas"):
    try:
        s=importlib.util.find_spec(mod)
        if s and s.submodule_search_locations:
            b=list(s.submodule_search_locations)[0]
            for sub in ("lib","bin","..\\cuda_runtime\\bin","..\\..\\nvidia\\cuda_runtime\\bin","..\\..\\nvidia\\cublas\\bin"):
                p=os.path.normpath(os.path.join(b,sub))
                if p not in seen: seen.add(p); print(p)
    except Exception: pass
'@ 2>$null
  foreach ($line in $pyout) { if ($line) { $dllDirs.Add($line) } }
}
function Find-Cudart($major) {
  foreach ($d in $dllDirs) { if ($d -and (Test-Path $d)) { if (Get-ChildItem -Path $d -Filter "cudart64_$major*.dll" -ErrorAction SilentlyContinue | Select-Object -First 1) { return $d } } }
  return $null
}
$rt13 = Find-Cudart 13; $rt12 = Find-Cudart 12
if ($rt13) { Info "cudart64_13*.dll : $rt13" }
if ($rt12) { Info "cudart64_12*.dll : $rt12" }
if ($rt13 -or $rt12) { Ok ("installed CUDA runtime DLL major: " + (@(if($rt13){'13'}; if($rt12){'12'}) -join ', ')) }
else { Info 'no cudart64_*.dll found locally - selection will fall back to the driver CUDA version (the paired cudart zip supplies the runtime anyway)' }
Hr

# --------------------------------------------------------------------------- #
# 3. Select + download the upstream build (binaries + paired cudart)
# --------------------------------------------------------------------------- #
Bold '3) Select + download the Windows CUDA prebuilt'
$useTag = $TAG; $rt = $env:WIN_RUNTIME; $reason = ''; $cpuBuild = $false
if ($LLAMA_URL) {
  $reason = 'forced via LLAMA_URL/BUNDLE_URL'
} elseif ($rt) {
  $reason = "forced via WIN_RUNTIME=$rt"
} else {
  # Mirror the installer's driver-gated choice.
  $has13 = $rt13 -or ($driverMaj -ge 13)
  $has12 = $rt12 -or ($driverMaj -gt 12) -or ($driverMaj -eq 12 -and $driverMin -ge 4)
  if ($has13 -and $driverMaj -ge 13 -and (($driverMaj -gt 13) -or ($driverMin -ge 3))) {
    $rt = '13.3'; $reason = "driver CUDA $driverMaj.$driverMin supports cuda-13.3"
  } elseif ($has13 -and $blackwell -and $driverMaj -eq 13) {
    $useTag = $BL_TAG; $rt = $BL_RT; $reason = "Blackwell on a 13.0-13.2 driver: pinned $BL_TAG cuda-$BL_RT (13.3 needs a 13.3 driver)"
  } elseif ($has12) {
    $rt = '12.4'; $reason = "selecting cuda-12.4 (driver CUDA $driverMaj.$driverMin)"
  } else {
    # No CUDA runtime/driver: fall back to the upstream CPU build, like the installer.
    $cpuBuild = $true; $reason = 'no NVIDIA CUDA runtime/driver detected - selecting the CPU build'
  }
}
if (-not $LLAMA_URL) {
  if ($cpuBuild) {
    $LLAMA_URL = "https://github.com/$REPO/releases/download/$TAG/llama-$TAG-bin-win-cpu-x64.zip"
  } elseif ($rt) {
    $LLAMA_URL = "https://github.com/$REPO/releases/download/$useTag/llama-$useTag-bin-win-cuda-$rt-x64.zip"
    if (-not $CUDART_URL) { $CUDART_URL = "https://github.com/$REPO/releases/download/$useTag/cudart-llama-bin-win-cuda-$rt-x64.zip" }
  }
}
# Auto-derive the paired (tag-less) cudart URL from an explicit CUDA LLAMA_URL.
if ($LLAMA_URL -and -not $CUDART_URL -and $LLAMA_URL -match 'llama-[^/]*-bin-win-cuda-\d+\.\d+-x64\.zip') {
  $CUDART_URL = ($LLAMA_URL -replace 'llama-[^/]*-bin-win-cuda-', 'cudart-llama-bin-win-cuda-')
}
# A forced CPU LLAMA_URL is a CPU build too (no cudart pairing).
if ($LLAMA_URL -match 'bin-win-cpu') { $cpuBuild = $true }
if ($reason) { Info "selection : $reason" }
if (-not $LLAMA_URL) { Bad 'could not determine a build to download'; }
else {
  Info "binaries  : $LLAMA_URL"
  if ($CUDART_URL) { Info "cudart    : $CUDART_URL" } else { Warn 'no paired cudart zip - llama-server may need a CUDA toolkit on PATH' }
  if (Get-File $LLAMA_URL (Join-Path $WORK 'llama.zip')) { Ok "downloaded binaries ($(HSize (Join-Path $WORK 'llama.zip')))" }
  else { Bad 'binaries download failed'; }
  if ($CUDART_URL) {
    if (Get-File $CUDART_URL (Join-Path $WORK 'cudart.zip')) { Ok "downloaded cudart ($(HSize (Join-Path $WORK 'cudart.zip')))" }
    else { Warn 'cudart download failed' }
  }
}
Hr

# --------------------------------------------------------------------------- #
# 4. Extract (binaries + cudart into one dir)
# --------------------------------------------------------------------------- #
Bold '4) Extract'
$B = Join-Path $WORK 'bundle'
Remove-Item -Recurse -Force $B -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $B | Out-Null
$server_exe = $null
if (Test-Path (Join-Path $WORK 'llama.zip')) {
  try { Expand-Archive -Path (Join-Path $WORK 'llama.zip') -DestinationPath $B -Force } catch { Bad "binaries extract failed: $_" }
  if (Test-Path (Join-Path $WORK 'cudart.zip')) { try { Expand-Archive -Path (Join-Path $WORK 'cudart.zip') -DestinationPath $B -Force } catch { Warn "cudart extract failed: $_" } }
  $server_exe = (Get-ChildItem -Path $B -Recurse -Filter 'llama-server.exe' -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($server_exe) {
    $bindir = $server_exe.Directory.FullName
    # If the cudart DLLs landed at the zip root but the exe is in a subdir, copy them next to the exe.
    if ($bindir -ne $B) { Get-ChildItem -Path $B -Filter 'cudart64_*.dll' -ErrorAction SilentlyContinue | Copy-Item -Destination $bindir -Force -ErrorAction SilentlyContinue; Get-ChildItem -Path $B -Filter 'cublas*64_*.dll' -ErrorAction SilentlyContinue | Copy-Item -Destination $bindir -Force -ErrorAction SilentlyContinue }
    Ok "extracted (llama-server.exe in $bindir)"
  } else { Bad 'llama-server.exe not found in the archive' }
}
Hr

# --------------------------------------------------------------------------- #
# 5. DLL presence (CUDA backend + runtime sit next to the exe)
# --------------------------------------------------------------------------- #
Bold '5) Backend + runtime DLLs'
if ($cpuBuild) {
  Info 'CPU build - no CUDA backend or runtime DLLs expected'
} elseif ($server_exe) {
  $bindir = $server_exe.Directory.FullName
  $ggml   = Get-ChildItem -Path $bindir -Filter 'ggml-cuda.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
  $cudart = Get-ChildItem -Path $bindir -Filter 'cudart64_*.dll' -ErrorAction SilentlyContinue | Select-Object -First 1
  $cublas = Get-ChildItem -Path $bindir -Filter 'cublas64_*.dll'  -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($ggml)   { Info "ggml-cuda.dll : $($ggml.Name)" }   else { Warn 'ggml-cuda.dll not next to the exe (CUDA backend may be a plugin or missing)' }
  if ($cudart) { Info "cudart        : $($cudart.Name)" } else { Warn 'cudart64_*.dll not next to the exe - needs a CUDA toolkit on PATH' }
  if ($cublas) { Info "cublas        : $($cublas.Name)" }
  if ($ggml -and $cudart) { Ok 'CUDA backend + runtime DLLs present beside llama-server.exe' }
} else { Warn 'skipped (no exe)' }
Hr

# --------------------------------------------------------------------------- #
# 6. Version
# --------------------------------------------------------------------------- #
Bold '6) llama-server.exe --version'
if ($server_exe) {
  $vout = & $server_exe.FullName --version 2>&1 | Select-Object -First 3
  $vout | ForEach-Object { Info $_ }
  if ($vout -match 'version:') { Ok 'binary runs' } else { Bad 'binary did not run' }
} else { Bad 'skipped (no exe)' }
Hr

# --------------------------------------------------------------------------- #
# 7. Model
# --------------------------------------------------------------------------- #
Bold '7) Download a small test GGUF'
$gguf = Join-Path $WORK ([IO.Path]::GetFileName(([uri]$GGUF_URL).AbsolutePath))
if ((Test-Path $gguf) -and (Get-Item $gguf).Length -gt 0) { Ok "model present ($(HSize $gguf))" }
elseif (Get-File $GGUF_URL $gguf) { Ok "downloaded model ($(HSize $gguf))" }
else { Bad 'model download failed' }
Hr

# --------------------------------------------------------------------------- #
# 8. GPU inference with full offload (-ngl 99)
# --------------------------------------------------------------------------- #
Bold '8) GPU inference with full offload (-ngl 99)'
$ready = $false; $outLog = Join-Path $WORK 'server.out.log'; $errLog = Join-Path $WORK 'server.err.log'
if ($server_exe -and (Test-Path $gguf)) {
  $srvArgs = @('-m', $gguf, '-ngl', '99', '--host', '127.0.0.1', '--port', "$PORT", '-c', '2048', '--jinja')
  $server = Start-Process -FilePath $server_exe.FullName -ArgumentList $srvArgs -PassThru -NoNewWindow `
              -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  for ($i = 0; $i -lt 90; $i++) {
    try { $h = Invoke-WebRequest "http://127.0.0.1:$PORT/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
          if ($h.StatusCode -eq 200) { $ready = $true; break } } catch {}
    if ($server.HasExited) { break }
    Start-Sleep -Seconds 1
  }
  $log = @(); if (Test-Path $errLog) { $log += Get-Content $errLog }; if (Test-Path $outLog) { $log += Get-Content $outLog }
  if ($ready) {
    Ok "server healthy on :$PORT"
    if ($cpuBuild) {
      Info 'CPU build (no CUDA) - inference runs on CPU as expected'
    } else {
    # 'ggml_cuda_init' alone is not proof of offload - it is logged even when init
    # fails and llama.cpp drops to CPU. Require a real offload marker, and treat
    # "GPU visible but no offload" as a FAIL (the build silently ran on CPU).
    $gpuVisible = ($caps.Count -gt 0)
    $gpuFail = $log | Select-String -Pattern 'failed to initialize CUDA|no CUDA devices|no usable GPU|CUDA error' | Select-Object -First 3
    $gpuOk   = $log | Select-String -Pattern 'offloaded .* layers to GPU|offloading .* layers to GPU|CUDA0 .*buffer size|found \d+ CUDA device|using device CUDA' | Select-Object -First 4
    if ($gpuOk -and -not $gpuFail) {
      $gpuOk | ForEach-Object { Info $_.Line }; Ok 'CUDA GPU backend is ACTIVE (real GPU offload)'
    } elseif ($gpuVisible) {
      Bad 'GPU is visible via nvidia-smi but the prebuilt ran on CPU (no GPU offload) - this build does not use your GPU'
      ($log | Select-String -Pattern 'cuda|device|offload|buffer|backend|error|tensor' | Select-Object -First 16) | ForEach-Object { Info $_.Line }
    } elseif ($gpuFail) {
      $gpuFail | ForEach-Object { Info $_.Line }; Warn 'CUDA failed to initialize - running on CPU (no GPU on this host)'
    } else {
      Warn 'no CUDA offload lines in the log - running on CPU (no GPU on this host)'
      ($log | Select-String -Pattern 'buffer size|backend|CPU' | Select-Object -First 3) | ForEach-Object { Info $_.Line }
    }
    }
  } else {
    Bad 'server failed to become ready:'; ($log | Select-Object -Last 15) | ForEach-Object { Info $_ }
  }
  if ($ready) {
    try {
      $body = @{ messages=@(@{role='user';content='In one short sentence, what is the capital of Japan?'}); max_tokens=40; temperature=0 } | ConvertTo-Json -Depth 6
      $r = Invoke-RestMethod "http://127.0.0.1:$PORT/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 120
      $content = $r.choices[0].message.content
      Info "model reply: $content"
      if ($content -match '(?i)tokyo') { Ok 'coherent generation (mentions Tokyo)' } else { Warn 'answer unexpected' }
    } catch { Warn "generation request failed: $_" }
  }
} else { Bad 'skipped (no exe or model)' }
Hr

# --------------------------------------------------------------------------- #
# 9. Tool calling
# --------------------------------------------------------------------------- #
Bold '9) Tool calling'
if ($ready) {
  try {
    $tool = @{
      messages=@(@{role='user';content='What is the weather in Paris? Use the get_weather tool.'})
      tools=@(@{ type='function'; function=@{ name='get_weather'; description='Get weather'; parameters=@{ type='object'; properties=@{ location=@{ type='string' } }; required=@('location') } } })
      tool_choice='auto'; max_tokens=128; temperature=0
    } | ConvertTo-Json -Depth 12
    $tr = Invoke-RestMethod "http://127.0.0.1:$PORT/v1/chat/completions" -Method Post -ContentType 'application/json' -Body $tool -TimeoutSec 120
    $traw = $tr | ConvertTo-Json -Depth 12
    if ($traw -match 'get_weather') { Ok 'model emitted a get_weather tool call' } else { Warn 'no tool call (small model may decline)' }
  } catch { Warn "tool-call request failed: $_" }
} else { Warn 'skipped (server not ready)' }
Hr

# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
Stop-Server
if (-not $KEEP) { Remove-Item -Force (Join-Path $WORK 'llama.zip'),(Join-Path $WORK 'cudart.zip') -ErrorAction SilentlyContinue }
Bold '=== SUMMARY ==='
Write-Host "host: $arch   build: $(if($LLAMA_URL){[IO.Path]::GetFileName(([uri]$LLAMA_URL).AbsolutePath)}else{'none'})"
Write-Host "PASS: $script:PASS_N   WARN: $script:WARN_N   FAIL: $script:FAIL_N"
Write-Host ''
if ($script:FAIL_N -eq 0) {
  if ($cpuBuild) { Bold 'RESULT: CONFIRMED - CPU prebuilt runs on this box (no NVIDIA GPU detected).' }
  else { Bold 'RESULT: CONFIRMED - prebuilt runs on this box.' }
} else { Bold "RESULT: $script:FAIL_N hard failure(s) - paste this whole output back." }
Write-Host "(server logs: $errLog ; $outLog)"
exit 0
