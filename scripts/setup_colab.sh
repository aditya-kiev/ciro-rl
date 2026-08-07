#!/usr/bin/env bash
# One-shot environment setup for this repo on a Colab / Linux GPU box.
#   bash scripts/setup_colab.sh
#
# Installs the core (torch, torchvision, numpy, scipy, scikit-learn, PyYAML,
# pytest) into the active Python environment, plus credentialed DCS installs
# ONLY if you accept the MuJoCo + Distracting Control licenses. See README ->
# "dcs_wrapper.py / dm_control / distracting_control license note".
#
# Nothing here is conditional on being on a T4; the code falls back to CPU and
# disables AMP if no CUDA is present (see ciro_rl.utils.seeding.resolve_device).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Installing core dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "==> Installing dev/test deps"
python -m pip install "pytest>=7.0"

# Optional DCS environments. Enable by setting INSTALL_DCS=1 before running, or
# uncomment below. Requires WMLaunch/MuJoCo license acceptance.
if [[ "${INSTALL_DCS:-0}" == "1" ]]; then
  echo "==> Installing DCS dependencies (MuJoCo + distracting_control)"
  python -m pip install "dm-control>=1.0.29" "distracting-control>=0.0.1"
else
  echo "==> Skipping DCS installs (set INSTALL_DCS=1 to enable), DCS datasets will be skipped at runtime"
fi

echo "==> Sanity check"
python -c "import torch, torchvision, numpy, scipy, sklearn, yaml; torch.manual_seed(0); print('imports OK; devices:', getattr(torch, 'cuda_is_available')() if hasattr(torch,'cuda_is_available') else torch.cuda.is_available())"

echo "==> Unit tests"
python -m pytest -q tests

echo "==> Quick smoke test"
python experiments/quick_test.py --out ./outputs/quick_test

echo "Done. See README.md for the full run instructions."