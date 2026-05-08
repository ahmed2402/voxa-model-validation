# VOXA Gate 1 — Detailed Setup Guide
### RunPod + vLLM for Complete Beginners

> **Reading time:** 15 minutes  
> **Setup time:** 30–45 minutes (most of it is waiting for downloads)  
> **Cost:** ~$2–3 total if you terminate the pod when done

---

## Table of Contents

1. [What You're Setting Up and Why](#1-what-youre-setting-up-and-why)
2. [Before You Start — Accounts & Tokens](#2-before-you-start--accounts--tokens)
3. [RunPod — Create and Deploy a Pod](#3-runpod--create-and-deploy-a-pod)
4. [Connect to Your Pod](#4-connect-to-your-pod)
5. [Get the Repo onto the Pod](#5-get-the-repo-onto-the-pod)
6. [Run the Bootstrap Script](#6-run-the-bootstrap-script)
7. [Understanding vLLM — What It Is and How It Works](#7-understanding-vllm--what-it-is-and-how-it-works)
8. [Test Execution Order — Step by Step](#8-test-execution-order--step-by-step)
   - [Option A: Guided sequential runner (recommended)](#option-a--guided-sequential-runner-recommended)
   - [Option B: Manual step-by-step](#option-b--manual-step-by-step)
   - [Model 1: Whisper Turbo (STT)](#model-1-whisper-turbo-stt)
   - [Model 2: Qwen3.5-9B (LLM)](#model-2-qwen35-9b-llm)
   - [Model 3: VoxCPM2 (TTS)](#model-3-voxcpm2-tts)
   - [Gate 1 Report](#gate-1-report)
9. [Monitoring GPU During Tests](#9-monitoring-gpu-during-tests)
10. [Terminating the Pod — CRITICAL](#10-terminating-the-pod--critical)
11. [Troubleshooting](#11-troubleshooting)
12. [Steps Requiring Manual Assistance](#12-steps-requiring-manual-assistance)

---

## 1. What You're Setting Up and Why

**RunPod** is a cloud GPU rental service. You rent a machine with an NVIDIA GPU by the second, run your workload, then terminate — you only pay for time used. You will rent one **RTX 4090 (24 GB VRAM)** pod for a few hours.

**vLLM** is an inference engine for large language models. Think of it as a web server that loads a model into GPU memory and exposes an API endpoint. Your Python test script then calls that API exactly like it would call the OpenAI API — same format, local URL.

**Why separate server + test script for LLM only?**  
Loading a 16 GB model into VRAM takes 60–90 seconds. Keeping vLLM as a persistent server means you load once and run all 20 test prompts against it instantly, rather than reloading for every single inference. Whisper and VoxCPM2 are much smaller and load inline in the test scripts directly — no server needed for them.

**Three models, tested sequentially — each fully clears the GPU before the next loads:**

| # | Model | Role | VRAM | How it runs |
|---|-------|------|------|-------------|
| 1 | Whisper Turbo | STT | ~2 GB | Inline in test script, single terminal |
| 2 | Qwen3.5-9B | LLM | ~18 GB | vLLM server (Terminal 1) + test client (Terminal 2) |
| 3 | VoxCPM2 | TTS | ~4 GB | Inline in test script, single terminal |

---

## 2. Before You Start — Accounts & Tokens

### 2a. RunPod Account

> **[MANUAL]** You must do this before anything else.

1. Go to [runpod.io](https://runpod.io) and click **Sign Up**
2. Verify your email
3. Go to **Billing** → **Add Credits** → add **$10** minimum (actual spend will be ~$2–3)
4. Save your login credentials — you will need to access the dashboard mid-session

### 2b. HuggingFace Token

> **[MANUAL]** Required to download Qwen3.5-9B weights.

1. Go to [huggingface.co](https://huggingface.co) → Settings → **Access Tokens**
2. Click **New token** → name it `voxa-runpod` → select **Read** role → click **Generate**
3. Copy the token (starts with `hf_...`) — you will only see it once
4. Save it somewhere safe (password manager or notepad)

You will paste this token into the pod terminal as an environment variable during setup.

### 2c. Urdu Audio Files

> **[MANUAL]** Required for the Whisper STT test.

1. Go to [commonvoice.mozilla.org/ur/datasets](https://commonvoice.mozilla.org/ur/datasets)
2. Download the Urdu validated clips dataset (the smallest validated set is fine)
3. Extract and pick **10 short clips** (5–15 seconds each, male and female voices ideally)
4. Any filename is fine — the script picks up all `.wav` files in `data/audio_samples/`
5. Keep them ready — you will upload them to the pod after connecting

> **Audio format:** Whisper works best with 16 kHz mono WAV. If your clips are stereo or at a different sample rate, convert them locally before uploading:
> ```bash
> # Using ffmpeg (install: brew install ffmpeg  or  apt install ffmpeg)
> ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
> ```

---

## 3. RunPod — Create and Deploy a Pod

> **[MANUAL]** Follow these steps exactly.

### Step 1 — Go to Pods

Log in → click **Pods** in the left sidebar → click the blue **+ Deploy** button.

### Step 2 — Select a Template

Search for and select:

```
RunPod PyTorch 2.4
```

> **Why 2.4 and not 2.1?** Modern vLLM (0.6+) requires CUDA 12.4+. The PyTorch 2.1 template ships with CUDA 12.1 which causes vLLM install failures and `RuntimeError: CUDA error` crashes. The bootstrap script handles the upgrade automatically if you're on 2.1, but starting on 2.4 avoids the problem entirely.

### Step 3 — Select GPU

In the GPU picker, look for **RTX 4090** under the **Community Cloud** tab.

- VRAM: 24 GB
- Approximate cost: **$0.34/hr**
- If no RTX 4090 is available, **RTX 3090 (24 GB)** is an acceptable substitute

> Avoid the Secure Cloud tab — it costs significantly more.

### Step 4 — Configure Storage

Before clicking Deploy, set these storage values:

| Setting | Value | Why |
|---------|-------|-----|
| Container Disk | **50 GB** | Qwen3.5-9B weights are ~16 GB; Whisper + VoxCPM2 add ~10 GB more |
| Volume Disk | 0 GB | Not needed — results will be downloaded before pod termination |

### Step 5 — Set Expose Ports (Optional)

Click **Customize Deployment** → in the **Expose TCP Ports** field add:

```
8000
```

This makes the vLLM server (port 8000) accessible for health checks from your local machine if needed. Not strictly required since tests run inside the pod.

> Port 9090 is no longer needed — Whisper now runs inline, not as a server.

### Step 6 — Deploy

Click **Deploy**. The pod will take 1–3 minutes to start. You will see the status change from `Starting` → `Running`.

---

## 4. Connect to Your Pod

Once the pod status shows **Running**, click the **Connect** button next to your pod.

### Option A — JupyterLab (Recommended)

1. Click **Connect to JupyterLab**
2. A browser tab opens with a file browser and notebook interface
3. Click **Terminal** (the black square icon) to open a terminal
4. To open a **second terminal** later (needed for the LLM step only): click the `+` button → Terminal

### Option B — SSH

1. Click **Connect via SSH**
2. Copy the SSH command shown (looks like `ssh root@[ip] -p [port] -i ~/.ssh/id_rsa`)
3. Paste it in your local terminal
4. If you get a permission error, add your SSH public key under RunPod → Settings → SSH Public Keys

---

## 5. Get the Repo onto the Pod

In the pod terminal, run these commands one by one:

```bash
# Confirm you have internet and GPU access
nvidia-smi
curl -s https://huggingface.co | head -5

# Clone the repo
git clone https://github.com/YOUR_USERNAME/voxa-model-validation.git
cd voxa-model-validation

# Set your HuggingFace token so model downloads work
export HF_TOKEN=hf_your_token_here
huggingface-cli login --token $HF_TOKEN
```

> If you have not pushed the repo to GitHub yet: in JupyterLab, use the file upload button (↑) to upload the entire `voxa-model-validation/` folder as a zip, then `unzip` it in the terminal.

---

## 6. Run the Bootstrap Script

This is the single setup script. Run it once per pod session from the repo root.

```bash
bash scripts/00_runpod_setup.sh
```

**What it does (7 steps):**

| Step | Action | Time |
|------|--------|------|
| 1 | GPU check (`nvidia-smi`) | instant |
| 2 | Upgrade pip | ~10 s |
| 3 | Upgrade PyTorch to 2.4 (CUDA 12.4 wheel) | ~2–3 min |
| 4 | Install vLLM from PyPI | ~3–5 min |
| 5 | Install `openai-whisper` + `ffmpeg` + `numpy<2.0` | ~1 min |
| 6 | Install remaining deps (`openai`, `httpx`, `rich`, `soundfile`, `voxcpm`) | ~30 s |
| 7 | Create output directories, verify all imports | instant |

**Expected final output:**

```
========================================
VOXA Pod Setup Complete
GPU    : NVIDIA GeForce RTX 4090
VRAM   : 24564 MiB
PyTorch: 2.4.0+cu124
CUDA   : 12.4
vLLM   : 0.x.x

════════════════════════════════════════
 SEQUENTIAL TEST ORDER (one at a time):
════════════════════════════════════════

 ① STT — single terminal, runs fast (~5 min)
   python scripts/01_test_whisper.py
 ...
========================================
```

> **Why the PyTorch upgrade?** The RunPod PyTorch 2.1 template ships CUDA 12.1. vLLM 0.6+ requires CUDA 12.4+. The bootstrap script upgrades PyTorch to a CUDA 12.4 wheel — this works because RTX 4090 drivers (525+) forward-support CUDA 12.4 binaries even with a CUDA 12.1 container toolkit. Without this step, `pip install vllm` either crashes or installs an old incompatible version.

---

## 7. Understanding vLLM — What It Is and How It Works

You don't need to understand vLLM deeply, but this mental model will help you when things go wrong.

### The Architecture (LLM step only)

```
Terminal 1 (server)          Terminal 2 (test script)
┌────────────────────┐       ┌──────────────────────────┐
│  vLLM process      │       │  02_test_qwen.py         │
│                    │       │                          │
│  Loads Qwen3.5-9B  │       │  wait_for_vllm()         │
│  into GPU VRAM     │◄──────│  → pings /health         │
│                    │       │                          │
│  Listens on        │       │  For each of 20 prompts: │
│  localhost:8000    │◄──────│  → POST /v1/chat/        │
│                    │       │     completions          │
│  Responds with     │──────►│                          │
│  completions       │       │  Logs result             │
└────────────────────┘       └──────────────────────────┘
```

Whisper (STT) and VoxCPM2 (TTS) load directly in their test scripts — no separate server process.

### Key vLLM Flags

| Flag | What it means |
|------|---------------|
| `--model` | HuggingFace model ID — vLLM downloads it automatically |
| `--dtype half` | Explicit float16 — Qwen3.5-9B fits in 24 GB at fp16 |
| `--max-model-len 8192` | Maximum token context window (prompt + response) |
| `--gpu-memory-utilization 0.85` | Reserve 85% of VRAM for weights + KV cache. Do NOT set to 1.0 — the pod will crash with OOM |
| `--served-model-name qwen3` | The alias the test script uses when calling the API |
| `--enable-prefix-caching` | Caches repeated parts of the system prompt — speeds up sequential calls |
| `--trust-remote-code` | Required for Qwen3's tokenizer |
| `--disable-log-requests` | Suppresses per-request log noise; makes startup message easier to spot |

### What "Application startup complete" Means

```
Loading model weights...          ← downloading from HuggingFace (~10 min first time)
Loading model into GPU...         ← takes ~30–60 seconds
Warming up model...               ← a few test inferences
INFO:     Application startup complete.   ← SERVER IS READY
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Do not run the test script until you see "Application startup complete".**

### First Run vs. Subsequent Runs

The first time you start vLLM with `Qwen/Qwen3.5-9B`, it downloads ~16 GB of weights from HuggingFace to the pod disk. This takes **8–15 minutes** depending on network speed. On subsequent starts in the same pod session, the model loads from disk in ~60–90 seconds.

---

## 8. Test Execution Order — Step by Step

**Run models in this exact order. Always stop the current model before starting the next — each needs the full 24 GB VRAM.**

### Option A — Guided sequential runner (recommended)

A single script that walks you through all three stages with VRAM checks, pauses at manual steps, and generates the final report:

```bash
bash scripts/run_sequential_tests.sh
```

It will pause and wait for your input at the two manual steps (LLM review and TTS scoring). The LLM stage still requires a second terminal for the vLLM server — the script tells you exactly when and what to run.

---

### Option B — Manual step-by-step

Follow the sections below if you prefer to run each piece yourself.

---

### Model 1: Whisper Turbo (STT)

**Before you begin:** Upload your Urdu `.wav` files to `data/audio_samples/`.

In JupyterLab, use the upload button in the file browser sidebar to navigate to `data/audio_samples/` and upload the files. Or via terminal:
```bash
# From your local machine:
scp -P [pod-port] *.wav root@[pod-ip]:~/voxa-model-validation/data/audio_samples/
```

**Single terminal — run directly:**
```bash
cd voxa-model-validation
python scripts/01_test_whisper.py
```

Whisper Turbo loads directly onto the GPU (~1.5 GB download on first run), transcribes each `.wav` file, and saves results to `results/stt_results.json`. No server process needed.

**What you will see:**
```
Device : cuda — NVIDIA GeForce RTX 4090
Loading Whisper turbo (downloads ~1.5 GB on first run)...
Model loaded on CUDA.

Transcribing 10 file(s)...

[1/10] urdu_01.wav ...
  Transcript : السلام علیکم...
  Language   : ur  |  Latency : 312 ms

...

p95 Latency : 410 ms  (target: < 500 ms)
Gate passed : YES
```

When the script finishes, the Whisper model is unloaded automatically — GPU memory is free for the next step.

---

### Model 2: Qwen3.5-9B (LLM)

This step uses **two terminals simultaneously**.

**Terminal 1 — Start the vLLM server:**
```bash
cd voxa-model-validation
bash scripts/02_start_vllm_server.sh
```

Watch the logs. The first run downloads ~16 GB of weights — this takes time. Wait for:
```
INFO:     Application startup complete.
```

**Terminal 2 — Run LLM tests:**
```bash
cd voxa-model-validation
python scripts/02_test_qwen.py
```

The script first pings `localhost:8000/health` (printing dots) until the server is ready, then runs all 20 prompts.

**What you will see:**
```
Waiting for vLLM server at http://localhost:8000/health .......... Ready.

Running 20 LLM test prompts...

[01/20] [urdu_faq] آپ کے دفتر کے اوقات کیا ہیں؟
  → ہمارا دفتر پیر سے جمعہ، صبح 9 بجے سے شام 5 بجے تک کھلا رہتا ہے۔
  Latency: 842 ms   Pass criteria: responds in Urdu, mentions time/hours
...
```

**After the test completes:**

> **[MANUAL]** Open `results/llm_results.json` and review each response against the `pass_criteria` field. Set `"manual_pass": true` or `"manual_pass": false` for each of the 20 entries.

Quick terminal review:
```bash
python -c "
import json
records = [json.loads(l) for l in open('results/llm_results.json') if l.strip()]
for r in records:
    if r.get('type') == 'aggregate': continue
    print(f\"\n[{r['test_id']}] {r['category']}\")
    print(f\"  IN : {r['input'][:80]}\")
    print(f\"  OUT: {r['output'][:120]}\")
    print(f\"  CRITERIA: {r['pass_criteria']}\")
"
```

**When done:** Press `Ctrl+C` in Terminal 1 to stop vLLM. This frees ~18 GB VRAM for the TTS model. Confirm with `nvidia-smi` that memory usage drops before continuing.

---

### Model 3: VoxCPM2 (TTS)

No server needed — the model loads inline.

```bash
cd voxa-model-validation
python scripts/03_test_voxcpm2.py
```

This generates 10 `.wav` files in `results/tts_audio/` and prints instructions for human scoring.

> **[MANUAL]** Share the `results/tts_audio/` folder with 3 native Urdu speakers. Ask them to score each clip 1–5 (see scoring rubric in the script output). Collect the scores.

Once you have the scores:
```bash
python scripts/03_test_voxcpm2.py --score
```

The script walks you through entering scores clip by clip.

> **Note:** VoxCPM2 officially supports 30 languages — Urdu is not listed (Hindi is, and shares some phonemes). Expect average scores of 1.5–2.5 / 5.0. If Gate 1 fails for TTS, ElevenLabs Multilingual v2 becomes the mandatory TTS for VOXA.

---

### Gate 1 Report

After all three models are tested and LLM results are manually reviewed:

```bash
python scripts/04_gate1_report.py
```

This prints and saves `results/gate1_report.md`. Exit code `0` = PASSED, `1` = FAILED.

**Download your results before terminating the pod:**
```bash
# From your LOCAL machine terminal:
scp -P [pod-port] -r root@[pod-ip]:~/voxa-model-validation/results/ ./voxa-results/
```

Or in JupyterLab: right-click the `results/` folder → Download as zip.

---

## 9. Monitoring GPU During Tests

Open an extra terminal and run:

```bash
watch -n 2 nvidia-smi
```

This refreshes GPU stats every 2 seconds. Key things to watch:

| Metric | What to look for |
|--------|-----------------|
| `Memory-Usage` | Should be < 22,000 MiB at peak. If it hits 24,000 MiB the process will crash with OOM |
| `GPU-Util` | Should spike to 80–100% during inference, drop to near 0% when idle |
| `Temp` | Normal range 70–85°C under load. Above 90°C is a warning |

If you see an OOM error in vLLM:
1. Stop the server (`Ctrl+C`)
2. Lower `--gpu-memory-utilization` to `0.70` in `scripts/02_start_vllm_server.sh`
3. Restart

---

## 10. Terminating the Pod — CRITICAL

> **This is the most important step for cost control.**

After downloading your results:

1. Go to [runpod.io](https://runpod.io) → **Pods**
2. Find your pod — it should show status **Running**
3. Click the **three-dot menu (⋮)** next to your pod
4. Click **Stop Pod**
5. Confirm

The billing **stops within seconds** of the pod stopping.

**What NOT to do:**
- Do not just close the browser tab — the pod keeps running
- Do not click "Pause" if that option appears — some templates still charge for paused pods
- Do not leave it running while you sleep / review results — even 8 idle hours = ~$2.72 wasted

**How to verify it stopped:**
Refresh the RunPod dashboard. The pod status should show **Stopped** (grey, not green). Check your billing page — the per-second charge should have stopped.

---

## 11. Troubleshooting

### `ModuleNotFoundError: No module named 'whisper'`

The bootstrap script may have failed partway through. Re-run:

```bash
pip install openai-whisper "numpy<2.0"
```

> Do **not** install `whisper-live` — that is a streaming server package, not the inference library. The test script uses `import whisper` which comes from the `openai-whisper` package.

---

### Whisper runs on CPU instead of GPU

The test script checks `torch.cuda.is_available()`. If CUDA is unavailable, it falls back to CPU. Fix:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

If this prints `False`, the PyTorch install doesn't have CUDA support. Re-run the bootstrap script — specifically step 3 which upgrades PyTorch to the CUDA 12.4 wheel:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

---

### vLLM install fails / `RuntimeError: CUDA error` when starting vLLM

Root cause: PyTorch version in the container doesn't match what vLLM expects.

Check:
```bash
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda)"
python -c "import vllm; print('vllm:', vllm.__version__)"
```

Fix — upgrade PyTorch first, then reinstall vLLM:
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install vllm
```

> Do **not** add `--extra-index-url https://download.pytorch.org/whl/cu121` when installing vLLM. That is a PyTorch wheel index and has no effect on vLLM — it only confuses the dependency resolver.

---

### `CUDA out of memory` when starting vLLM

A previous model process is still holding GPU memory.

```bash
# See what is using GPU memory
nvidia-smi

# Kill lingering Python processes
pkill -f whisper   # if Whisper didn't unload
pkill -f vllm      # if a previous vLLM instance crashed without cleanup
```

Then restart the vLLM server. If OOM persists, lower the memory utilization:
```bash
# Edit scripts/02_start_vllm_server.sh
# Change --gpu-memory-utilization 0.85 to 0.70
```

---

### Whisper returns empty transcripts for Urdu audio

Check:
1. Audio is 16 kHz mono (not stereo, not 44.1 kHz):
   ```bash
   python -c "import soundfile as sf; d,sr=sf.read('data/audio_samples/urdu_01.wav'); print(sr, d.shape)"
   ```
2. File is not silent or corrupt (shape should be non-zero)
3. Try letting Whisper auto-detect language: open `01_test_whisper.py` and change `language="ur"` to `language=None`

---

### HuggingFace download fails / 401 Unauthorized

```bash
huggingface-cli login --token hf_your_token_here
```

Qwen3.5-9B is not a gated model so authentication is not strictly required, but a token avoids rate limiting on large downloads.

---

### `Connection refused` when the test script tries to reach vLLM

The server is either not started or still loading. Check Terminal 1 for `Application startup complete`. The test script waits up to 120 seconds — if it times out, the server likely crashed (read the error in Terminal 1).

---

### JupyterLab terminal disconnects mid-test

Use `tmux` to run the vLLM server in a persistent session:
```bash
tmux new -s vllm
bash scripts/02_start_vllm_server.sh
# Detach: Ctrl+B then D
# Reattach later: tmux attach -t vllm
```

---

## 12. Steps Requiring Manual Assistance

The following steps **cannot be automated** and require your direct action:

---

### MANUAL 1 — RunPod account creation and credit top-up
**When:** Before anything else  
**What:** Create a RunPod account at runpod.io, add at least $10 in credits  
**Why:** Requires payment information and account creation on an external website

---

### MANUAL 2 — HuggingFace token generation
**When:** Before pod setup  
**What:** Generate a Read token at huggingface.co → Settings → Access Tokens  
**Why:** Requires logging into your HuggingFace account

---

### MANUAL 3 — Pod deployment on RunPod
**When:** Start of session  
**What:** Click through the RunPod UI to create the pod (Section 3 above)  
**Why:** Requires navigating the RunPod web dashboard

---

### MANUAL 4 — Upload Urdu audio files
**When:** Before running `01_test_whisper.py`  
**What:** Upload your `.wav` files to `data/audio_samples/` on the pod  
**Why:** Audio files must come from you  
**Audio conversion if needed (run locally):**
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

---

### MANUAL 5 — LLM response review
**When:** After `02_test_qwen.py` completes  
**What:** Open `results/llm_results.json`, read each of the 20 model responses, set `"manual_pass": true` or `"manual_pass": false` based on the `pass_criteria` field  
**Why:** Subjective judgement — did the model respond in Urdu? Did it de-escalate correctly? Only a human can judge.  
**Time estimate:** 15–20 minutes

---

### MANUAL 6 — TTS naturalness scoring
**When:** After `03_test_voxcpm2.py` generates audio  
**What:** Share `results/tts_audio/` with 3 native Urdu speakers, collect their 1–5 scores for each of the 10 clips  
**Why:** Requires human native-speaker judgement on Urdu phoneme naturalness  
**Time estimate:** 30–60 minutes  
**Download the audio folder to your local machine first:**
```bash
# From your local machine:
scp -P [pod-port] -r root@[pod-ip]:~/voxa-model-validation/results/tts_audio/ ./tts_audio/
```

---

### MANUAL 7 — Pod termination
**When:** Immediately after downloading results  
**What:** Go to RunPod dashboard → stop the pod  
**Why:** Requires clicking in the RunPod web dashboard  
**This is the highest-stakes manual step — forgetting it costs money every minute**

---

### Summary Table

| # | Step | When | Estimated Time |
|---|------|------|----------------|
| 1 | Create RunPod account + add credits | Before start | 10 min |
| 2 | Generate HuggingFace token | Before start | 5 min |
| 3 | Deploy RunPod pod | Start of session | 5 min |
| 4 | Upload Urdu audio files | Before STT test | 15–30 min |
| 5 | Review 20 LLM responses, set pass/fail | After LLM test | 15–20 min |
| 6 | Collect TTS scores from native speakers | After TTS generation | 30–60 min |
| 7 | Terminate the pod | End of session | 2 min |
