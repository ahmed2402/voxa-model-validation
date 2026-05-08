#!/bin/bash
# VOXA Gate 1 — Sequential test runner guide
#
# This script prints the exact sequence and checks VRAM between stages.
# It does NOT launch background processes — you still need two terminals
# for the LLM step (vLLM server + test client simultaneously).
#
# Usage: bash scripts/run_sequential_tests.sh

set -e
cd "$(dirname "$0")/.."

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${CYAN}[VOXA]${NC} $1"; }
ok()  { echo -e "${GREEN}  ✓${NC} $1"; }
warn(){ echo -e "${YELLOW}  ⚠${NC}  $1"; }
err() { echo -e "${RED}  ✗${NC} $1"; }

check_gpu_free() {
    local threshold_mib="${1:-2000}"
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | tr -d ' MiB')
    if [ "${used:-0}" -gt "$threshold_mib" ]; then
        warn "GPU is using ${used} MiB. Stop previous model before continuing."
        warn "  pkill -f whisper   # if Whisper server is running"
        warn "  pkill -f vllm      # if vLLM server is running"
        return 1
    fi
    ok "GPU memory clear (${used} MiB used)"
    return 0
}

echo ""
echo "════════════════════════════════════════════════"
echo "   VOXA Gate 1 — Sequential Model Validation"
echo "════════════════════════════════════════════════"
echo ""
echo " Models tested in order (each clears GPU before next):"
echo "   1. Whisper Turbo   STT   ~5 min    2 GB VRAM"
echo "   2. Qwen3.5-9B      LLM   ~30 min  18 GB VRAM"
echo "   3. VoxCPM2         TTS   ~20 min   4 GB VRAM"
echo ""

# ── PRE-FLIGHT ────────────────────────────────────────────────────────────────
log "Pre-flight checks..."

if ! command -v python &>/dev/null; then
    err "python not found. Run setup first: bash scripts/00_runpod_setup.sh"; exit 1
fi

python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null && \
    ok "CUDA available ($(python -c 'import torch; print(torch.cuda.get_device_name(0))'))" || \
    { err "CUDA not available — run bash scripts/00_runpod_setup.sh first"; exit 1; }

python -c "import whisper" 2>/dev/null && ok "openai-whisper installed" || \
    { err "openai-whisper missing — run bash scripts/00_runpod_setup.sh"; exit 1; }

python -c "import vllm" 2>/dev/null && ok "vLLM installed" || \
    { err "vLLM missing — run bash scripts/00_runpod_setup.sh"; exit 1; }

AUDIO_COUNT=$(ls data/audio_samples/*.wav 2>/dev/null | wc -l)
if [ "$AUDIO_COUNT" -eq 0 ]; then
    err "No .wav files in data/audio_samples/"
    err "Upload 10 Urdu audio clips before starting."
    exit 1
else
    ok "$AUDIO_COUNT audio file(s) found in data/audio_samples/"
fi

echo ""
echo "════════════════════════════════════════════════"
echo " STAGE 1 of 3 — STT (Whisper Turbo)"
echo "════════════════════════════════════════════════"
log "Checking GPU is clear before loading Whisper..."
check_gpu_free 500

log "Running Whisper STT test..."
python scripts/01_test_whisper.py
ok "STT test complete → results/stt_results.json"

echo ""
echo "════════════════════════════════════════════════"
echo " STAGE 2 of 3 — LLM (Qwen3.5-9B via vLLM)"
echo "════════════════════════════════════════════════"
log "Checking GPU is clear after Whisper (model should have unloaded)..."
check_gpu_free 2000

echo ""
warn "This stage needs TWO terminals:"
echo ""
echo "  ┌─ Terminal 1 (vLLM server) ────────────────────────────┐"
echo "  │  bash scripts/02_start_vllm_server.sh                 │"
echo "  │  Wait for: 'Application startup complete'             │"
echo "  │  First run downloads ~16 GB — takes 8–15 minutes      │"
echo "  └───────────────────────────────────────────────────────┘"
echo ""
echo "  ┌─ Terminal 2 (this terminal, after server is ready) ───┐"
echo "  │  python scripts/02_test_qwen.py                       │"
echo "  └───────────────────────────────────────────────────────┘"
echo ""

read -r -p "  Press ENTER when the vLLM server shows 'Application startup complete'..."
echo ""

log "Running Qwen LLM test..."
python scripts/02_test_qwen.py
ok "LLM test complete → results/llm_results.json"

echo ""
warn "ACTION REQUIRED: Open results/llm_results.json and set manual_pass for each entry."
warn "Then press ENTER to continue (or Ctrl+C to pause and review first)."
warn "Also stop the vLLM server in Terminal 1 with Ctrl+C before continuing."
echo ""
read -r -p "  Press ENTER when vLLM is stopped and manual review is done..."

echo ""
echo "════════════════════════════════════════════════"
echo " STAGE 3 of 3 — TTS (VoxCPM2)"
echo "════════════════════════════════════════════════"
log "Checking GPU is clear after stopping vLLM..."
check_gpu_free 2000

log "Generating TTS audio clips..."
python scripts/03_test_voxcpm2.py
ok "TTS audio generated → results/tts_audio/"

echo ""
warn "ACTION REQUIRED: Share results/tts_audio/ with 3 native Urdu speakers."
warn "Collect their 1–5 scores, then run:"
echo ""
echo "  python scripts/03_test_voxcpm2.py --score"
echo ""
read -r -p "  Press ENTER when TTS scoring is complete..."

echo ""
echo "════════════════════════════════════════════════"
echo " FINAL REPORT"
echo "════════════════════════════════════════════════"
python scripts/04_gate1_report.py
GATE_EXIT=$?

echo ""
if [ $GATE_EXIT -eq 0 ]; then
    ok "Gate 1 PASSED — results/gate1_report.md"
else
    warn "Gate 1 FAILED — see results/gate1_report.md for details"
fi

echo ""
echo "════════════════════════════════════════════════"
warn "STOP THE POD NOW to avoid billing!"
echo "  RunPod dashboard → Pods → ⋮ → Stop Pod"
echo "  Download results first if you haven't already."
echo "════════════════════════════════════════════════"
echo ""

exit $GATE_EXIT
