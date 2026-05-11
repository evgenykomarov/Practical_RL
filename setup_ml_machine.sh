#!/usr/bin/env bash
set -euo pipefail

# reverse tunnel on the machine with internet access:
# ssh -N -R 127.0.0.1:1080 gpu2

export ALL_PROXY=socks5h://127.0.0.1:1080
export HTTP_PROXY=socks5h://127.0.0.1:1080
export HTTPS_PROXY=socks5h://127.0.0.1:1080

# ============================
# User-configurable variables
# ============================

ENV_NAME="qed137_ml"
PYTHON_VERSION="3.11.9"

# PyTorch CUDA wheel index.
# Your NVIDIA driver supports CUDA 12.4, and PyTorch cu121 wheels should work fine.
# PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"

# ============================
# System dependencies
# ============================

echo "Installing system dependencies..."

# sudo apt update

sudo apt install -y \
    build-essential \
    make \
    gcc \
    g++ \
    git \
    curl \
    wget \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    python3-openssl \
    python3-venv \
    unzip \
    htop \
    tmux \
    tree

# ============================
# Install pyenv if missing
# ============================

if [ ! -d "$HOME/.pyenv" ]; then
    echo "Installing pyenv..."
    curl https://pyenv.run | bash
else
    echo "pyenv already installed."
fi

# ============================
# Configure shell startup
# ============================

SHELL_RC="$HOME/.bashrc"

if ! grep -q 'PYENV_ROOT' "$SHELL_RC"; then
    echo "Adding pyenv config to $SHELL_RC..."

    cat <<'EOF' >> "$SHELL_RC"

# pyenv configuration
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
EOF
fi

# Load pyenv in this script session
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# # # ============================
# # # Install Python
# # # ============================

if ! pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
    echo "Installing Python $PYTHON_VERSION..."
    pyenv install "$PYTHON_VERSION"
else
    echo "Python $PYTHON_VERSION already installed."
fi

# # ============================
# # Create virtualenv
# # ============================

if ! pyenv virtualenvs --bare | grep -qx "$ENV_NAME"; then
    echo "Creating pyenv virtualenv $ENV_NAME..."
    pyenv virtualenv "$PYTHON_VERSION" "$ENV_NAME"
else
    echo "Virtualenv $ENV_NAME already exists."
fi

pyenv activate "$ENV_NAME"

# ============================
# Upgrade packaging tools
# ============================

python -m pip install --upgrade pip setuptools wheel

# ============================
# Install PyTorch with GPU support
# ============================

echo "Installing PyTorch with CUDA support..."

python -m pip install \
    torch \
    torchvision \
    torchaudio \
    # --index-url "$PYTORCH_INDEX_URL"

# ============================
# Write requirements file
# ============================

cat > requirements-ml-rl.txt <<'EOF'
# Core scientific Python
numpy==1.26.4
scipy==1.13.1
pandas==2.2.2
scikit-learn==1.5.1
statsmodels==0.14.2

# Plotting / notebooks
matplotlib>=3.9,<3.11
seaborn==0.13.2
plotly==5.23.0
jupyterlab==4.2.4
notebook==7.2.1
ipywidgets==8.1.3
tqdm==4.66.5

# ML / DL utilities
einops==0.8.0
tensorboard==2.17.0
torchmetrics==1.4.1
lightning==2.3.3
transformers==4.43.3
datasets==2.20.0
accelerate==0.33.0
huggingface-hub==0.24.5
safetensors==0.4.3

# RL libraries
gymnasium==0.29.1
stable-baselines3==2.3.2
pygame==2.6.0
ale-py==0.8.1

# Config / experiment management
hydra-core==1.3.2
omegaconf==2.3.0
wandb==0.17.5
rich==13.7.1
typer==0.12.3
python-dotenv==1.0.1

# Images / vision / video
pillow==10.4.0
opencv-python==4.10.0.84
imageio==2.34.2
imageio-ffmpeg==0.5.1

# Dev quality of life
black==24.4.2
ruff==0.5.5
pytest==8.3.2
mypy==1.11.0
ipykernel==6.29.5

# Useful miscellany
requests==2.32.3
beautifulsoup4==4.12.3
pyyaml==6.0.2
joblib==1.4.2
EOF

# ============================
# Install requirements
# ============================

echo "Installing ML/RL requirements..."

python -m pip install -r requirements-ml-rl.txt

# ============================
# Add Jupyter kernel
# ============================

python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# ============================
# Test GPU availability
# ============================

echo ""
echo "Testing PyTorch GPU access..."
python - <<'PY'
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version used by PyTorch:", torch.version.cuda)
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    print("GPU tensor test OK:", y.shape, y.device)
else:
    print("WARNING: CUDA is not available to PyTorch.")
PY

echo ""
echo "Done."
echo ""
echo "To use the environment in a new terminal:"
echo "    pyenv activate $ENV_NAME"
echo ""
echo "To start JupyterLab:"
echo "    jupyter lab"
echo ""
echo "If pyenv is not found in a new terminal, run:"
echo "    source ~/.bashrc"
