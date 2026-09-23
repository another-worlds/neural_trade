<#
.SYNOPSIS
    Build the `nt` training environment from nothing on a Windows box.

.DESCRIPTION
    Installs Miniforge (conda-forge only, BSD-licensed, no Anaconda Terms-of-Service
    obligations) if no conda is on PATH, then creates the `nt` environment:
    Python 3.10 + CUDA 11.2.2 + cuDNN 8.1.0.77 from conda-forge, and the pinned pip
    stack from requirements.txt (TensorFlow 2.10.0).

    TensorFlow 2.10 is the last release with native Windows GPU support, which is why
    the whole stack is pinned around it. The CUDA/cuDNN pair is not a free choice: it
    is what the official TF 2.10 Windows wheel links against.

.PARAMETER Prefix
    Where to install Miniforge if it has to be installed. Default: $HOME\miniforge3.

.PARAMETER EnvName
    Conda environment name. Default: nt.

.PARAMETER CpuOnly
    Skip cudatoolkit/cudnn; install the CPU wheel set from requirements-ci.txt instead.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -CpuOnly
#>
[CmdletBinding()]
param(
    [string]$Prefix  = (Join-Path $HOME 'miniforge3'),
    [string]$EnvName = 'nt',
    [switch]$CpuOnly
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

# --------------------------------------------------------------- 1. conda
$conda = $null
$onPath = Get-Command conda -ErrorAction SilentlyContinue
if ($onPath) {
    $conda = $onPath.Source
    Write-Step "Using conda already on PATH: $conda"
} elseif (Test-Path (Join-Path $Prefix 'Scripts\conda.exe')) {
    $conda = Join-Path $Prefix 'Scripts\conda.exe'
    Write-Step "Using existing Miniforge at $Prefix"
} else {
    Write-Step "Installing Miniforge to $Prefix"
    $url = 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe'
    $exe = Join-Path $env:TEMP 'Miniforge3-Windows-x86_64.exe'
    Invoke-WebRequest -Uri $url -OutFile $exe -UseBasicParsing
    # NSIS: /D must be the LAST argument and must not be quoted.
    $p = Start-Process -FilePath $exe `
        -ArgumentList '/InstallationType=JustMe', '/RegisterPython=0', '/AddToPath=0', '/S', "/D=$Prefix" `
        -Wait -PassThru -NoNewWindow
    if ($p.ExitCode -ne 0) { throw "Miniforge installer exited with $($p.ExitCode)" }
    Remove-Item $exe -Force -ErrorAction SilentlyContinue
    $conda = Join-Path $Prefix 'Scripts\conda.exe'
}
if (-not (Test-Path $conda)) { throw "conda not found at $conda" }

# --------------------------------------------------------------- 2. environment
Write-Step "Creating the '$EnvName' environment (python 3.10$(if (-not $CpuOnly) { ' + CUDA 11.2 / cuDNN 8.1' }))"
$createArgs = @('create', '-y', '-n', $EnvName, '-c', 'conda-forge', '--override-channels', 'python=3.10', 'pip')
if (-not $CpuOnly) { $createArgs += @('cudatoolkit=11.2.2', 'cudnn=8.1.0.77') }
& $conda @createArgs
if ($LASTEXITCODE -ne 0) { throw "conda create failed ($LASTEXITCODE)" }

$envPython = & $conda run -n $EnvName python -c "import sys; print(sys.executable)"
if ($LASTEXITCODE -ne 0) { throw "could not resolve the interpreter for '$EnvName'" }
$envPython = $envPython.Trim()
Write-Host "interpreter: $envPython"

# --------------------------------------------------------------- 3. pip stack
$req = if ($CpuOnly) { 'requirements-ci.txt' } else { 'requirements.txt' }
Write-Step "Installing $req (this pulls ~500 MB of TensorFlow; expect several minutes)"
& $envPython -m pip install --upgrade pip
& $envPython -m pip install --no-input -r (Join-Path $RepoRoot $req)
if ($LASTEXITCODE -ne 0) { throw "pip install -r $req failed ($LASTEXITCODE)" }

# --------------------------------------------------------------- 4. verify
Write-Step 'Verifying'
& $envPython -c @'
import numpy, pandas, sklearn, scipy, tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print(f"tensorflow {tf.__version__}   keras {tf.keras.__version__}")
print(f"numpy {numpy.__version__}   pandas {pandas.__version__}   sklearn {sklearn.__version__}   scipy {scipy.__version__}")
print(f"built with CUDA: {tf.test.is_built_with_cuda()}   GPUs visible: {len(gpus)}")
for g in gpus:
    print("   ", g)
if not gpus:
    print("   (CPU only - set NEURAL_TRADE_FORCE_CPU=1 to make that explicit in tests)")
'@

Write-Step "Done. Use it with:  conda activate $EnvName   (or: $conda run -n $EnvName python ...)"
Write-Host "Next:  pytest -q      then      python -c `"from model import train_and_evaluate; train_and_evaluate(force=True, epochs=2)`""
