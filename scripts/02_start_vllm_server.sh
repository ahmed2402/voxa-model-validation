#!/bin/bash
set -e

echo "Starting vLLM server for Qwen3.5-9B..."
echo ""
echo "Model       : Qwen/Qwen3.5-9B"
echo "Quantization: none (fp16 — fits in 24 GB VRAM)"
echo "Port        : 8000"
echo "VRAM budget : 85% (~20 GB for weights + KV cache)"
echo "Context     : 8192 tokens"
echo ""
echo "NOTE: Qwen3 has a built-in thinking mode. We disable it via the"
echo "      chat template so responses are fast and direct (non-reasoning)."
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --served-model-name qwen3 \
    --host 0.0.0.0 \
    --enable-prefix-caching \
    --trust-remote-code

echo ""
echo "vLLM server starting... wait 60-90 seconds for model to load before running 02_test_qwen.py"
