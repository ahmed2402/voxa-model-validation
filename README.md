# VOXA Model Validation — Gate 1

## Overview

This repository contains the Gate 1 validation suite for VOXA, a multilingual (Urdu + English) voice agent system. Gate 1 is a **blocking quality gate** — all three component models (STT, LLM, TTS) must meet defined thresholds before any pipeline code is written. Testing is performed on a RunPod RTX 4090 pod and follows a strict sequential order: Whisper Turbo → Qwen3.5-9B → VoxCPM2. Results are persisted as JSON and aggregated into a final `gate1_report.md`.

---

## Prerequisites

- RunPod account with credits loaded (budget: ~$2.20 for full test run)
- RTX 4090 pod (24 GB VRAM) — Community Cloud
- Urdu `.wav` audio files in `data/audio_samples/` (16kHz mono recommended)
  - Download from: https://commonvoice.mozilla.org/ur/datasets
- Python 3.10+
- HuggingFace account token (to download gated models) — set as `HF_TOKEN` in `.env`

---

## RunPod Setup — Step by Step

1. Go to [runpod.io](https://runpod.io) → **Pods** → **Deploy**
2. Select template: **RunPod PyTorch 2.1** (has CUDA 12.1 pre-installed)
3. Select GPU: **RTX 4090 · Community Cloud · 24 GB VRAM**
4. Set container disk: **50 GB** (model weights need space)
5. Click **Deploy** — wait ~2 minutes for pod to start
6. Click **Connect** → **"Start Web Terminal"** (JupyterLab also works)
7. In the terminal, clone this repo and run setup:

```bash
git clone <your-repo-url>
cd voxa-model-validation
bash scripts/00_runpod_setup.sh
```

---

## Running Tests — In Order

```bash
# ── MODEL 1: Whisper Turbo (STT) ──────────────────────────
# Terminal 1: Start the STT server
python -m whisper_live.server --port 9090 --backend faster_whisper

# Terminal 2: Run tests (open via JupyterLab → New Terminal)
python scripts/01_test_whisper.py

# ── MODEL 2: Qwen3.5-9B (LLM) ──────────────────────────────
# Terminal 1: Start vLLM server (wait 60–90s for model to load)
bash scripts/02_start_vllm_server.sh

# Terminal 2: Run tests
python scripts/02_test_qwen.py

# ── MODEL 3: VoxCPM2 (TTS) ───────────────────────────────
# Single terminal (no server needed)
python scripts/03_test_voxcpm2.py

# After collecting scores from native speakers:
python scripts/03_test_voxcpm2.py --score

# ── GATE 1 REPORT ────────────────────────────────────────
python scripts/04_gate1_report.py
```

---

> **CRITICAL — Terminate your pod immediately after testing.**
> RunPod bills per second. An RTX 4090 costs ~$0.34/hr.
> Total test time: ~4–5 hours maximum = ~$1.70.
> Go to RunPod dashboard → Your Pod → **Stop Pod** when done.
> **Do NOT leave the pod running overnight.**

---

## Results

| File | Contents |
|------|----------|
| `results/stt_results.json` | Per-clip transcript, latency, language detected, p95 aggregate |
| `results/llm_results.json` | Per-prompt response, latency, manual_pass flag (you fill in) |
| `results/tts_results.json` | Per-clip generation time + naturalness scores after `--score` |
| `results/tts_audio/` | Generated `.wav` files for human review |
| `results/gate1_report.md` | Final Gate 1 pass/fail report with all metrics |

---

## Gate 1 Decision

| Model | Metric | Threshold | Action if Fail |
|-------|--------|-----------|----------------|
| Whisper Turbo | p95 latency | < 500 ms | Investigate audio quality / chunk size |
| Qwen3.5-9B | Manual pass rate | ≥ 80% (16/20) | Consider Qwen3-14B or prompt tuning |
| VoxCPM2 | Urdu naturalness avg | ≥ 3.0 / 5.0 | Switch to ElevenLabs Multilingual v2 |

The report script exits with code `0` (PASSED) or `1` (FAILED), making it suitable for CI/CD gating.
