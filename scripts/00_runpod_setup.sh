#!/bin/bash
# VOXA Gate 1 — RunPod bootstrap
# Run once per pod session, from the repo root: bash scripts/00_runpod_setup.sh
set -e

echo "========================================"
echo "VOXA RunPod Environment Setup"
echo "========================================"

# ── Step 1: GPU + driver check ────────────────────────────────────────────────
echo ""
echo "[1/7] Checking GPU and driver..."
nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

# Highest CUDA version the installed driver can serve
DRIVER_CUDA=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)
echo ""
echo "GPU   : $GPU_NAME"
echo "VRAM  : $VRAM_MIB"
echo "Max CUDA supported by driver: $DRIVER_CUDA"

# ── Step 2: Upgrade pip ───────────────────────────────────────────────────────
echo ""
echo "[2/7] Updating pip..."
pip install --upgrade pip --quiet

# ── Step 3: Upgrade PyTorch to a CUDA 12.4 build ─────────────────────────────
# The RunPod PyTorch 2.1 template ships CUDA 12.1 — modern vLLM (0.6+) requires
# CUDA 12.4+. RTX 4090 drivers (525+) forward-support CUDA 12.4 wheels even
# when the container toolkit is 12.1.  This upgrade fixes the vLLM + CUDA
# version mismatch that causes RuntimeError / undefined-symbol crashes.
echo ""
echo "[3/7] Upgrading PyTorch to 2.4 (CUDA 12.4 wheel — forward-compatible)..."
pip install \
    "torch==2.4.0" \
    "torchvision==0.19.0" \
    "torchaudio==2.4.0" \
    --index-url https://download.pytorch.org/whl/cu124 \
    --quiet

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available after torch upgrade!'
print(f'  torch {torch.__version__}  |  CUDA available: {torch.cuda.is_available()}')
print(f'  Device: {torch.cuda.get_device_name(0)}')
print(f'  VRAM  : {torch.cuda.get_device_properties(0).total_memory // 1024**2} MiB')
"

# ── Step 4: Install vLLM ──────────────────────────────────────────────────────
# Install from PyPI — no extra-index-url needed.
# vLLM wheels auto-match your CUDA toolkit; we installed torch cu124 above.
# Do NOT pass --extra-index-url https://download.pytorch.org/whl/cu121 here;
# that is a PyTorch wheel index and does nothing useful for vLLM.
echo ""
echo "[4/7] Installing vLLM (this takes 3–5 minutes)..."
pip install vllm --quiet

python -c "import vllm; print(f'  vLLM {vllm.__version__} installed OK')"

# ── Step 5: Install openai-whisper + ffmpeg ───────────────────────────────────
# openai-whisper is the standard library (pip install openai-whisper).
# whisper-live is a different package — a streaming server.  We do NOT use it.
# ffmpeg is required by whisper for audio decoding.
echo ""
echo "[5/7] Installing openai-whisper + ffmpeg..."
apt-get install -y -q ffmpeg
pip install "openai-whisper" "numpy<2.0" --quiet
python -c "import whisper; print(f'  openai-whisper OK (turbo model supported)')"

# ── Step 6: Install remaining project deps ────────────────────────────────────
echo ""
echo "[6/7] Installing remaining dependencies..."
pip install \
    "openai>=1.0.0" \
    "httpx>=0.24.0" \
    "pyyaml>=6.0" \
    "rich>=13.0.0" \
    "soundfile" \
    "voxcpm" \
    --quiet

python -c "import yaml, rich, openai, httpx, soundfile; print('  All utility imports OK')"

# ── Step 7: Create output directories ────────────────────────────────────────
echo ""
echo "[7/7] Creating output directories..."
mkdir -p results/tts_audio data/audio_samples

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "VOXA Pod Setup Complete"
echo "GPU    : $GPU_NAME"
echo "VRAM   : $VRAM_MIB"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.version.cuda)')"
echo "vLLM   : $(python -c 'import vllm; print(vllm.__version__)')"
echo ""
echo "════════════════════════════════════════"
echo " SEQUENTIAL TEST ORDER (one at a time):"
echo "════════════════════════════════════════"
echo ""
echo " ① STT — single terminal, runs fast (~5 min)"
echo "   python scripts/01_test_whisper.py"
echo ""
echo " ② LLM — requires TWO terminals simultaneously:"
echo "   [Terminal 1]  bash scripts/02_start_vllm_server.sh"
echo "                 Wait for: 'Application startup complete'"
echo "   [Terminal 2]  python scripts/02_test_qwen.py"
echo "   [Terminal 1]  Ctrl+C to stop vLLM when done"
echo ""
echo " ③ TTS — single terminal (run after vLLM is stopped)"
echo "   python scripts/03_test_voxcpm2.py"
echo ""
echo " ④ Report"
echo "   python scripts/04_gate1_report.py"
echo ""
echo " ⚠  STOP THE POD when done — billing is per-second!"
echo "========================================"
