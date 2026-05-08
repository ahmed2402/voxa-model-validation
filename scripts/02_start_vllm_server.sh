#!/bin/bash
# Start vLLM inference server for Qwen3.5-9B
# Run from the repo root in Terminal 1.
# Wait for "Application startup complete" before running 02_test_qwen.py in Terminal 2.

echo "════════════════════════════════════════════"
echo " vLLM Server — Qwen3.5-9B"
echo "════════════════════════════════════════════"
echo " Model       : Qwen/Qwen3.5-9B"
echo " Dtype       : float16  (fp16 — fits 24 GB VRAM)"
echo " Port        : 8000"
echo " VRAM budget : 85%  (~20 GB for weights + KV cache)"
echo " Context     : 8192 tokens"
echo ""
echo " Qwen3 thinking mode is NOT enabled here."
echo " /no_think in the system prompt keeps responses"
echo " fast and direct for customer service use."
echo ""
echo " First run downloads ~16 GB weights (~8–15 min)."
echo " Subsequent runs load from disk (~60–90 sec)."
echo "════════════════════════════════════════════"
echo ""

# Verify GPU is free before loading a 16 GB model
USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1 | tr -d ' MiB')
if [ "${USED_MIB:-0}" -gt 2000 ]; then
    echo "WARNING: GPU already using ${USED_MIB} MiB."
    echo "Stop any running Python processes first (e.g. Whisper server)."
    echo "  pkill -f whisper"
    echo "  pkill -f vllm"
    echo ""
fi

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B \
    --dtype half \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --served-model-name qwen3 \
    --host 0.0.0.0 \
    --enable-prefix-caching \
    --trust-remote-code \
    --no-enable-log-requests
