param(
    [string]$EnvName = "chb-mit-gpu"
)

$ErrorActionPreference = "Stop"

$existingEnv = conda env list | Select-String "^\s*$EnvName\s"

if (-not $existingEnv) {
    conda create -n $EnvName python=3.11 -y
}

conda run -n $EnvName python -m pip install -r requirements.txt
conda run -n $EnvName python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
conda run -n $EnvName python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
