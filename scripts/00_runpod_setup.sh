#!/bin/bash
# VOXA Gate 1 — RunPod bootstrap
# Run once per pod session, from the repo root: bash scripts/00_runpod_setup.sh
set -e

echo "========================================"
echo "VOXA RunPod Environment Setup"
echo "========================================"

# ── Step 1: GPU + driver check ────────────────────────────────────────────────
echo ""
echo "[1/6] Checking GPU and driver..."
nvidia-smi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
DRIVER_CUDA=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" | head -1)

# Build wheel tag: "12.8" → "cu128", "13.0" → "cu130"
CUDA_TAG=$(echo "$DRIVER_CUDA" | tr -d '.')
CUDA_TAG="cu${CUDA_TAG}"

echo ""
echo "GPU        : $GPU_NAME"
echo "VRAM       : $VRAM_MIB"
echo "Driver CUDA: $DRIVER_CUDA  (wheel tag: $CUDA_TAG)"

# ── Step 2: Upgrade pip + install uv ─────────────────────────────────────────
# uv is vLLM's recommended installer — it supports --torch-backend which
# lets us pin the CUDA wheel variant (e.g. cu128) independently of vLLM version.
echo ""
echo "[2/6] Updating pip and installing uv..."
pip install --upgrade pip uv --quiet

# ── Step 3: Install vLLM with the correct torch CUDA backend ──────────────────
# Using uv + --torch-backend avoids the mismatch between vLLM's default
# torch wheel (e.g. cu130) and the pod's actual driver CUDA support.
# This installs the latest vLLM with a torch build matching the driver.
echo ""
echo "[3/6] Installing vLLM with torch backend $CUDA_TAG..."
# vLLM 0.20.0 was the first release compiled against CUDA 13.0 (libcudart.so.13).
# Pods with driver CUDA <= 12.8 must use < 0.20.0 which is compiled against CUDA 12.x.
uv pip install "vllm<0.20.0" --torch-backend="$CUDA_TAG"

VLLM_VER=$(python -c "import vllm; print(vllm.__version__)")
echo "  vLLM $VLLM_VER installed"

# ── Step 4: Verify torch + CUDA ───────────────────────────────────────────────
echo ""
echo "[4/6] Verifying PyTorch + CUDA..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — driver/wheel mismatch'
print(f'  torch  {torch.__version__}')
print(f'  CUDA   {torch.version.cuda}')
print(f'  Device: {torch.cuda.get_device_name(0)}')
print(f'  VRAM  : {torch.cuda.get_device_properties(0).total_memory // 1024**2} MiB')
"

# ── Step 5: Install openai-whisper + ffmpeg ───────────────────────────────────
# openai-whisper = direct GPU inference library (import whisper).
# whisper-live   = streaming server package — NOT used here.
# ffmpeg binary  = required by whisper for audio decoding.
# apt-get update is required first — package lists are stale on fresh pods.
echo ""
echo "[5/6] Installing openai-whisper + ffmpeg..."
apt-get update -qq && apt-get install -y -q ffmpeg || \
    conda install -c conda-forge ffmpeg -y 2>/dev/null || \
    echo "  WARNING: ffmpeg not installed — audio transcription will fail"

pip install "openai-whisper" "numpy<2.0" --quiet
python -c "import whisper; print('  openai-whisper OK')"

# ── Step 6: Remaining deps + final check ─────────────────────────────────────
echo ""
echo "[6/6] Installing remaining dependencies and verifying..."
pip install \
    "openai>=1.0.0" \
    "httpx>=0.24.0" \
    "pyyaml>=6.0" \
    "rich>=13.0.0" \
    "soundfile" \
    "voxcpm" \
    --quiet

mkdir -p results/tts_audio data/audio_samples

python -c "
import torch, whisper, vllm, yaml, rich, openai, httpx, soundfile
print(f'  torch  {torch.__version__} | CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}')
print(f'  vLLM   {vllm.__version__}')
print('  All imports OK — ready to test')
"

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
echo " ① STT — single terminal (~5 min)"
echo "   python scripts/01_test_whisper.py"
echo ""
echo " ② LLM — two terminals simultaneously:"
echo "   [Terminal 1]  bash scripts/02_start_vllm_server.sh"
echo "                 Wait for: 'Application startup complete'"
echo "   [Terminal 2]  python scripts/02_test_qwen.py"
echo "   [Terminal 1]  Ctrl+C when done"
echo ""
echo " ③ TTS — single terminal (after stopping vLLM)"
echo "   python scripts/03_test_voxcpm2.py"
echo ""
echo " ④ Report"
echo "   python scripts/04_gate1_report.py"
echo ""
echo " ⚠  STOP THE POD when done — billing is per-second!"
echo "========================================"
