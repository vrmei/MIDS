# mids-bench Windows setup (PowerShell).
#
# Run in an *elevated* PowerShell prompt:
#
#     Set-ExecutionPolicy -Scope Process Bypass
#     .\scripts\setup_windows.ps1
#
# Idempotent: re-runs safely. Creates a conda env named `mids-bench`
# with PyTorch CUDA 12.1, minimal Python deps, and (transparently)
# the pure-PyTorch Mamba fallback. mamba-ssm / causal-conv1d are NOT
# installed — those are Linux-only without serious pain. The codebase
# already handles their absence (see src/models/mids.py).

$ErrorActionPreference = "Stop"

# -----------------------------------------------------------------
# 1) Conda check.
# -----------------------------------------------------------------
$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not $conda) {
    Write-Host "Conda not found on PATH." -ForegroundColor Yellow
    Write-Host "Install Miniconda first:"
    Write-Host "  https://docs.conda.io/projects/miniconda/en/latest/"
    Write-Host "Then re-open PowerShell and re-run this script."
    exit 1
}
Write-Host "[1/5] Conda detected at $($conda.Source)"

# -----------------------------------------------------------------
# 2) Env create / activate.
# -----------------------------------------------------------------
$envName = "mids-bench"
$existing = conda env list | Select-String -Pattern "^\s*$envName\s"
if ($existing) {
    Write-Host "[2/5] Env '$envName' already exists; reusing." -ForegroundColor Green
} else {
    Write-Host "[2/5] Creating env '$envName' with Python 3.10..."
    conda create -y -n $envName python=3.10
}

# -----------------------------------------------------------------
# 3) PyTorch with CUDA 12.1.
# -----------------------------------------------------------------
Write-Host "[3/5] Installing PyTorch 2.3.1 + CUDA 12.1..."
conda run -n $envName --no-capture-output `
    pip install torch==2.3.1 torchvision==0.18.1 `
        --index-url https://download.pytorch.org/whl/cu121

# -----------------------------------------------------------------
# 4) Other Python deps from requirements-minimal.txt.
# -----------------------------------------------------------------
Write-Host "[4/5] Installing minimal Python deps..."
$req = Join-Path $PSScriptRoot "..\requirements-minimal.txt"
conda run -n $envName --no-capture-output `
    pip install -r $req

# -----------------------------------------------------------------
# 5) Smoke test.
# -----------------------------------------------------------------
Write-Host "[5/5] Smoke-test: import torch + check CUDA..."
conda run -n $envName --no-capture-output python -c @"
import torch
print(f'  torch       : {torch.__version__}')
print(f'  CUDA avail  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  device      : {torch.cuda.get_device_name(0)}')
    print(f'  compute cap : {torch.cuda.get_device_capability(0)}')

# Verify the pure-PyTorch Mamba fallback works.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__) if '__file__' in dir() else '.', '.'))
os.environ['MIDS_FORCE_PURE_MAMBA'] = '1'
from src.models._mamba_torch import Mamba
m = Mamba(d_model=64, d_state=8, d_conv=4, expand=2).cuda() if torch.cuda.is_available() else Mamba(d_model=64)
x = torch.randn(2, 50, 64).to(next(m.parameters()).device)
y = m(x)
assert y.shape == x.shape
print(f'  Mamba smoke : OK (in {tuple(x.shape)} -> out {tuple(y.shape)})')
"@

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Green
Write-Host "Done. Activate with:" -ForegroundColor Green
Write-Host "    conda activate mids-bench"
Write-Host ""
Write-Host "Then run a single fold to verify end-to-end:"
Write-Host "    python scripts/run_one.py --model mids --dataset tesla --fold 0"
Write-Host ""
Write-Host "Notes:" -ForegroundColor Yellow
Write-Host "  - mamba-ssm / causal-conv1d are NOT installed (Windows-unfriendly)."
Write-Host "  - The codebase uses pure-PyTorch Mamba, ~3-5x slower but works."
Write-Host "  - For fast kernels later: switch to WSL2 + pip install mamba-ssm."
