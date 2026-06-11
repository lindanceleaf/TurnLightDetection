$ErrorActionPreference = "Stop"

Write-Host "Python:"
python -c "import sys; print(sys.executable); print(sys.version)"

Write-Host ""
Write-Host "Removing possible CPU-only torch packages..."
python -m pip uninstall -y torch torchvision torchaudio

Write-Host ""
Write-Host "Installing PyTorch CUDA 12.8 wheels..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Write-Host ""
Write-Host "Installing project dependencies..."
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "CUDA check:"
python check_cuda.py
