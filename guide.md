# VoxCPM2 — RunPod Test Guide (Urdu TTS)

Step-by-step guide to test **openbmb/VoxCPM2** on a fresh RunPod Pod using
the project's Urdu sentences ([data/urdu_test_sentences.txt](data/urdu_test_sentences.txt)).

- **Official repo:** https://github.com/OpenBMB/VoxCPM
- **HF model:** https://huggingface.co/openbmb/VoxCPM2
- **Test script:** [scripts/test_voxcpm2_only.py](scripts/test_voxcpm2_only.py)

> ⚠️ **Urdu is NOT in VoxCPM2's official supported language list.**
> The model officially supports 30 languages including Arabic and Hindi (closest
> to Urdu phonetically). We're running this to measure real-world quality
> on Urdu — expect imperfect pronunciation. Output will still be useful to
> evaluate intelligibility and decide whether to fine-tune or use a different model.

---

## 1. Create the RunPod Pod

### Recommended GPU
- **RTX 4090 (24 GB)** — official reference, ~$0.34/hr (community cloud)
- **A40 (48 GB)** — also fine
- **L4 / L40S** — works
- **Minimum:** any GPU with **≥ 8 GB VRAM** + CUDA 12.0+

### Pod settings
| Setting | Value |
|---|---|
| **Template** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or any PyTorch ≥ 2.5 / CUDA ≥ 12.0 image) |
| **GPU** | RTX 4090 (or any ≥ 8 GB) |
| **Container Disk** | 30 GB |
| **Volume Disk** | 30 GB (model weights ~ 4–6 GB; safer with headroom) |
| **Volume Mount Path** | `/workspace` |
| **Expose Ports** | 22 (SSH), 8888 (Jupyter optional) |

### Why these requirements
- **PyTorch ≥ 2.5.0**, **CUDA ≥ 12.0**, **Python 3.10–3.12** are hard requirements from the [VoxCPM repo](https://github.com/OpenBMB/VoxCPM).
- VoxCPM2 uses `bfloat16` and needs an Ampere or newer GPU (RTX 30xx/40xx, A-series, L-series). Avoid V100 / T4.

### Connect
```bash
# from local machine
ssh root@<pod-ip> -p <pod-port> -i ~/.ssh/<key>
# OR use the RunPod web terminal
```

---

## 2. Set up the project on the Pod

```bash
cd /workspace

# Option A: clone your own repo
# git clone <your-repo-url> voxa-model-validation
# cd voxa-model-validation

# Option B: copy just the two files we need (recommended for a quick test)
mkdir -p voxa-model-validation/{scripts,data,results/voxcpm2}
cd voxa-model-validation
# Then upload via RunPod web UI / scp:
#   scripts/test_voxcpm2_only.py
#   data/urdu_test_sentences.txt
```

`scp` example from your local Windows box (PowerShell):
```powershell
scp -P <pod-port> `
  "c:\Users\ahmed\OneDrive\Desktop\local_LLMs\voxa-model-validation\scripts\test_voxcpm2_only.py" `
  "c:\Users\ahmed\OneDrive\Desktop\local_LLMs\voxa-model-validation\data\urdu_test_sentences.txt" `
  root@<pod-ip>:/workspace/voxa-model-validation/
```

---

## 3. Create a clean Python venv

```bash
cd /workspace/voxa-model-validation

# Verify Python version is 3.10, 3.11, or 3.12
python3 --version

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Sanity
python -c "import sys; print(sys.version)"
```

> ❗ If `python3 --version` shows 3.13, install 3.11 first:
> ```bash
> apt-get update && apt-get install -y python3.11 python3.11-venv
> python3.11 -m venv .venv
> ```

---

## 4. Install dependencies

```bash
# Inside the activated venv
pip install --upgrade pip wheel

# Install PyTorch matching the pod's CUDA (12.4 example below)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Verify torch sees the GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Install VoxCPM (official package — pulls VoxCPM2 weights at runtime from HF)
pip install voxcpm

# Extras the test script needs
pip install soundfile
```

> If `voxcpm` install fails on missing build tools:
> ```bash
> apt-get install -y build-essential ffmpeg libsndfile1
> ```

---

## 5. (Optional) Hugging Face login

VoxCPM2 weights on HF are public, but logging in avoids rate-limit issues
and is required if you later use gated models.

```bash
pip install huggingface_hub
huggingface-cli login   # paste your HF token
```

Get a token at: https://huggingface.co/settings/tokens

---

## 6. Run the test

Place the files so the script can find them:
```
voxa-model-validation/
├── scripts/test_voxcpm2_only.py
├── data/urdu_test_sentences.txt
└── results/voxcpm2/        # auto-created
```

Run:
```bash
cd /workspace/voxa-model-validation
source .venv/bin/activate

python scripts/test_voxcpm2_only.py \
  --input data/urdu_test_sentences.txt \
  --out_dir results/voxcpm2
```

First run will download VoxCPM2 weights (~4–6 GB) from Hugging Face. Subsequent
runs reuse the cache at `~/.cache/huggingface/`.

### Expected output
```
results/voxcpm2/
├── urdu_01.wav        # 48 kHz, mono
├── urdu_02.wav
├── ...
├── urdu_10.wav
└── report.json        # latency + RTF per sentence + averages
```

Console will print per-sentence:
- `gen` — generation time in seconds
- `audio` — output audio length in seconds
- `RTF` — real-time factor (lower = faster; ~0.30 on RTX 4090 is the official benchmark)

---

## 7. Optional: Voice cloning

If you have a reference Urdu voice sample (3–10 seconds of clean speech, 16 kHz+):

```bash
python scripts/test_voxcpm2_only.py \
  --input data/urdu_test_sentences.txt \
  --out_dir results/voxcpm2_cloned \
  --reference_wav /workspace/reference_voice.wav
```

For "ultimate cloning" (better quality, needs the transcript of the reference):
```bash
python scripts/test_voxcpm2_only.py \
  --input data/urdu_test_sentences.txt \
  --out_dir results/voxcpm2_ultimate \
  --reference_wav /workspace/reference_voice.wav \
  --prompt_text "اردو میں ریفرنس آڈیو کا متن یہاں لکھیں"
```

---

## 8. Pull results back to your local machine

```powershell
# from local Windows PowerShell
scp -P <pod-port> -r `
  root@<pod-ip>:/workspace/voxa-model-validation/results/voxcpm2 `
  "c:\Users\ahmed\OneDrive\Desktop\local_LLMs\voxa-model-validation\results\"
```

Or zip first on the pod for faster transfer:
```bash
cd /workspace/voxa-model-validation
tar -czf voxcpm2_results.tar.gz results/voxcpm2
```

---

## 9. Tear down the Pod

After you've downloaded the results, **stop or terminate** the pod from the
RunPod dashboard to stop billing. If you'll keep iterating, **stop** (keeps
volume) is cheaper than re-creating.

---

## Tuning knobs (script flags)

| Flag | Default | Notes |
|---|---|---|
| `--cfg_value` | `2.0` | Higher = stronger adherence to text/style. 1.5–3.0 typical. |
| `--inference_timesteps` | `10` | Diffusion steps. Higher = better quality, slower. Try 8–20. |
| `--reference_wav` | `None` | Path to reference voice for cloning. |
| `--prompt_text` | `None` | Reference transcript for ultimate cloning. |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: voxcpm` | `pip install voxcpm` inside the activated venv |
| `CUDA out of memory` | Use a GPU with ≥ 8 GB free; close other processes |
| `bfloat16 not supported` | GPU too old (V100/T4). Use RTX 30xx+, A-series, or L-series |
| Garbled / robotic Urdu | Expected — Urdu is unsupported. Try cloning with an Urdu reference voice |
| HF download stalls | `huggingface-cli login` and re-run; or set `HF_HUB_ENABLE_HF_TRANSFER=1` |
| `python3.13` venv | Install 3.11: `apt-get install -y python3.11 python3.11-venv` |
