#!/bin/bash
set -e

echo "========================================"
echo "VOXA RunPod Environment Setup"
echo "========================================"

echo ""
echo "[1/6] Checking GPU visibility..."
nvidia-smi
echo "GPU check passed."

echo ""
echo "[2/6] Updating pip..."
pip install --upgrade pip

echo ""
echo "[3/6] Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "[4/6] Installing vLLM (CUDA 12.1 build — this may take 3–5 minutes)..."
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

echo ""
echo "[5/6] Creating output directories..."
mkdir -p results/tts_audio
mkdir -p data/audio_samples

echo ""
echo "[6/6] Verifying key imports..."
python -c "import yaml; import rich; import openai; import httpx; print('Core imports OK')"

echo ""
echo "========================================"
echo "VOXA Pod Setup Complete"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
echo "Python: $(python --version)"
echo "Next step: run scripts/01_test_whisper.py"
echo "========================================"
