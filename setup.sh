#!/usr/bin/env bash
# systematic_setup.sh  (with expecttest)
set -euo pipefail

ENV_NAME="pytorch_env"
PYTHON_VER="3.12"
CUDA_VER="12.1" # change to 11.8 if you need the older stack

echo ">>> 1/7  Updating system packages (sudo apt update && upgrade)"
sudo apt update -y
sudo apt upgrade -y

echo ">>> 2/7  Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VER}"
conda create -y -n "${ENV_NAME}" python="${PYTHON_VER}"

echo ">>> 3/7  Activating environment and upgrading pip"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
python -m pip install --upgrade pip

echo ">>> 4/7  Installing PyTorch (CUDA ${CUDA_VER})"
conda install -y pytorch torchvision torchaudio pytorch-cuda="${CUDA_VER}" -c pytorch -c nvidia

echo ">>> 5/7  Installing sglang[all]==0.4.6.post5"
python -m pip install "sglang[all]==0.4.6.post5"

echo ">>> 6/7  Installing expecttest"
python -m pip install expecttest

echo ">>> 7/7  Adding environment to Jupyter notebook"
conda install -y ipykernel
python -m ipykernel install --user --name="${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo ">>> Verifying installation"
python - <<'PY'
import torch, sys, importlib.util
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

for pkg in ["sglang", "expecttest"]:
    spec = importlib.util.find_spec(pkg)
    print(f"{pkg}: {spec.loader.get_filename() if spec else 'NOT FOUND'}")
PY

echo ""
echo ">>> All steps completed!"
echo "   • Activate the environment:  conda activate ${ENV_NAME}"
echo "   • Launch Jupyter and choose the kernel 'Python (${ENV_NAME})'"
