"""
VoxCPM2 Standalone Test Script
Tests openbmb/VoxCPM2 TTS on Urdu sentences.

Official references:
  - GitHub: https://github.com/OpenBMB/VoxCPM
  - HF:     https://huggingface.co/openbmb/VoxCPM2
"""

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch
from voxcpm import VoxCPM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="data/urdu_test_sentences.txt",
        help="Path to text file (one sentence per line)",
    )
    p.add_argument(
        "--out_dir",
        default="results/voxcpm2",
        help="Directory to save .wav files and report.json",
    )
    p.add_argument("--model_id", default="openbmb/VoxCPM2")
    p.add_argument("--cfg_value", type=float, default=2.0)
    p.add_argument("--inference_timesteps", type=int, default=10)
    p.add_argument(
        "--reference_wav",
        default=None,
        help="Optional reference .wav for voice cloning",
    )
    p.add_argument(
        "--prompt_text",
        default=None,
        help="Optional transcript of reference wav (ultimate cloning)",
    )
    return p.parse_args()


def load_sentences(path: Path):
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s:
            lines.append(s)
    return lines


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences(in_path)
    print(f"[info] Loaded {len(sentences)} sentences from {in_path}")

    print(f"[info] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[info] GPU: {torch.cuda.get_device_name(0)}")

    print(f"[info] Loading {args.model_id} ...")
    t0 = time.time()
    model = VoxCPM.from_pretrained(args.model_id, load_denoiser=False)
    load_secs = time.time() - t0
    sr = model.tts_model.sample_rate
    print(f"[info] Loaded in {load_secs:.1f}s | sample_rate={sr}")

    results = []
    for i, text in enumerate(sentences, start=1):
        print(f"\n[{i}/{len(sentences)}] {text}")
        gen_kwargs = dict(
            text=text,
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
        )
        if args.reference_wav:
            gen_kwargs["reference_wav_path"] = args.reference_wav
            if args.prompt_text:
                gen_kwargs["prompt_wav_path"] = args.reference_wav
                gen_kwargs["prompt_text"] = args.prompt_text

        t0 = time.time()
        wav = model.generate(**gen_kwargs)
        gen_secs = time.time() - t0

        wav_path = out_dir / f"urdu_{i:02d}.wav"
        sf.write(str(wav_path), wav, sr)

        audio_secs = len(wav) / sr
        rtf = gen_secs / audio_secs if audio_secs > 0 else None
        print(
            f"    -> {wav_path.name} | gen={gen_secs:.2f}s | "
            f"audio={audio_secs:.2f}s | RTF={rtf:.3f}"
        )

        results.append(
            {
                "index": i,
                "text": text,
                "wav_file": wav_path.name,
                "gen_seconds": round(gen_secs, 3),
                "audio_seconds": round(audio_secs, 3),
                "rtf": round(rtf, 3) if rtf is not None else None,
            }
        )

    report = {
        "model_id": args.model_id,
        "sample_rate": sr,
        "model_load_seconds": round(load_secs, 2),
        "cfg_value": args.cfg_value,
        "inference_timesteps": args.inference_timesteps,
        "reference_wav": args.reference_wav,
        "num_sentences": len(sentences),
        "avg_rtf": round(
            sum(r["rtf"] for r in results) / max(len(results), 1), 3
        ),
        "results": results,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] Wrote {len(results)} wav files + {report_path}")
    print(f"[done] Average RTF: {report['avg_rtf']}")


if __name__ == "__main__":
    main()
