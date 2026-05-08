"""STT validation — Whisper Turbo via faster-whisper (GPU, direct inference)."""

# ── CUDA library path fix ────────────────────────────────────────────────────
# vLLM installs CUDA 13 pip packages whose .so files sit outside the default
# LD_LIBRARY_PATH. We collect every nvidia/*/lib path + system CUDA paths and
# re-exec once so the dynamic linker picks them up before ctranslate2 loads.
import os, sys

if os.environ.get("_CUDA_PATHS_SET") != "1":
    import site
    cuda_paths = []
    nvidia_base = os.path.join(site.getsitepackages()[0], "nvidia")
    if os.path.isdir(nvidia_base):
        for pkg in sorted(os.listdir(nvidia_base)):
            lib = os.path.join(nvidia_base, pkg, "lib")
            if os.path.isdir(lib):
                cuda_paths.append(lib)
    for p in ["/usr/local/cuda/lib64", "/usr/local/cuda-12.1/lib64",
              "/usr/local/cuda-12.4/lib64", "/usr/lib/x86_64-linux-gnu"]:
        if os.path.isdir(p):
            cuda_paths.append(p)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(cuda_paths) + (":" + existing if existing else "")
    os.environ["_CUDA_PATHS_SET"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)
# ────────────────────────────────────────────────────────────────────────────

import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils.logger import ResultLogger
from utils.scorer import Gate1Scorer


def load_config() -> dict:
    with open(ROOT / "configs" / "test_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    load_config()

    audio_dir = ROOT / "data" / "audio_samples"
    wav_files = sorted(audio_dir.glob("*.wav"))

    if not wav_files:
        print(
            "\nNo audio files found in data/audio_samples/\n"
            "Add .wav files (16kHz mono recommended) then re-run.\n"
        )
        sys.exit(0)

    print("Loading Whisper Turbo on GPU (downloads ~1.5 GB on first run)...")
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("turbo", device="cuda", compute_type="float16")
        print("Model loaded on GPU.\n")
    except Exception as exc:
        print(f"GPU load failed ({exc})\nFalling back to CPU / int8 ...")
        from faster_whisper import WhisperModel
        model = WhisperModel("turbo", device="cpu", compute_type="int8")
        print("Model loaded on CPU.\n")

    logger = ResultLogger(
        model_name="Whisper Turbo",
        output_file=str(ROOT / "results" / "stt_results.json"),
    )

    print(f"Transcribing {len(wav_files)} file(s)...\n")

    for i, wav_path in enumerate(wav_files, start=1):
        print(f"[{i}/{len(wav_files)}] {wav_path.name} ...")
        start = time.monotonic()
        try:
            segments, info = model.transcribe(
                str(wav_path),
                language="ur",
                beam_size=5,
                vad_filter=True,
            )
            transcript = " ".join(seg.text for seg in segments).strip()
            latency_ms = int((time.monotonic() - start) * 1000)
            lang_detected = info.language
            status = "pass" if transcript else "empty"
        except Exception as exc:
            transcript = ""
            latency_ms = 9999
            lang_detected = "error"
            status = f"error: {exc}"

        print(f"  Transcript : {transcript[:100] or '(empty)'}")
        print(f"  Language   : {lang_detected}  |  Latency: {latency_ms} ms\n")

        logger.log(
            test_id=f"stt_{i:02d}",
            category="urdu_audio",
            input_text=wav_path.name,
            output=transcript,
            metadata={
                "latency_ms": latency_ms,
                "language_detected": lang_detected,
                "status": status,
            },
        )

    print("── Summary ─────────────────────────────────")
    logger.summary()

    scorer = Gate1Scorer()
    stt_score = scorer.score_stt(logger.records)
    print(f"\np95 Latency : {stt_score['p95_latency_ms']} ms  (target: < {stt_score['threshold_ms']} ms)")
    print(f"Gate passed : {'YES' if stt_score['gate_passed'] else 'NO'}")

    with open(ROOT / "results" / "stt_results.json", "a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "aggregate", "stt_score": stt_score}, ensure_ascii=False) + "\n")

    print("\nResults saved → results/stt_results.json")
    print("Next step   : bash scripts/02_start_vllm_server.sh  (Terminal 1)")


if __name__ == "__main__":
    main()
