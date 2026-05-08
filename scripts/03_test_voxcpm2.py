"""TTS validation — VoxCPM2 Urdu audio generation + Gate 1 naturalness scoring."""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils.scorer import Gate1Scorer


def load_config() -> dict:
    with open(ROOT / "configs" / "test_config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sentences() -> list[str]:
    path = ROOT / "data" / "urdu_test_sentences.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate_audio(cfg: dict) -> list[dict]:
    tts_cfg = cfg["models"]["tts"]
    output_dir = ROOT / tts_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences()
    results = []

    print("Loading VoxCPM2 model...")
    try:
        from voxcpm import VoxCPM2
        model = VoxCPM2(load_denoiser=tts_cfg["load_denoiser"])
    except ImportError:
        print("ERROR: voxcpm package not installed. Run: pip install voxcpm")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR loading VoxCPM2: {exc}")
        sys.exit(1)

    print(f"Generating {len(sentences)} audio clips...\n")

    for i, sentence in enumerate(sentences, start=1):
        out_path = output_dir / f"urdu_{i:02d}.wav"
        start = time.monotonic()
        try:
            audio = model.synthesize(
                text=sentence,
                cfg_value=tts_cfg["cfg_value"],
                inference_timesteps=tts_cfg["inference_timesteps"],
            )
            import soundfile as sf
            sf.write(str(out_path), audio, samplerate=22050)
            gen_ms = int((time.monotonic() - start) * 1000)
            status = "ok"
        except Exception as exc:
            gen_ms = int((time.monotonic() - start) * 1000)
            status = f"error: {exc}"
            print(f"  WARNING: clip {i} failed — {exc}")

        print(f"[{i}/{len(sentences)}] Generated {out_path.name} — {gen_ms} ms")
        results.append({
            "clip_id": i,
            "filename": out_path.name,
            "sentence": sentence,
            "generation_ms": gen_ms,
            "status": status,
        })

    # Persist generation results
    results_path = ROOT / "results" / "tts_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nAudio saved → {output_dir}")
    print(f"Results saved → {results_path}")

    print("""
============================================================
TTS AUDIO GENERATED — GATE 1 SCORING REQUIRED
============================================================
Share the folder results/tts_audio/ with 3 native Urdu speakers.
Ask them to score each clip 1–5:
  5 = Completely natural
  4 = Natural, minor artifacts
  3 = Acceptable, clearly AI
  2 = Unnatural, hard to follow
  1 = Robotic / unintelligible

When scores are collected, run this command to record them:
  python scripts/03_test_voxcpm2.py --score

============================================================
WARNING: VoxCPM2 does NOT list Urdu in its 30 supported
languages. Hindi is supported and shares some phonemes.
Expect scores of 1.5-2.5/5. If average < 3.0, Gate 1 FAILS
for TTS and ElevenLabs Multilingual v2 becomes mandatory.
============================================================
""")
    return results


def collect_scores() -> None:
    sentences = load_sentences()
    print("Entering scoring mode — enter scores for each clip.\n")
    print("Format: three comma-separated integers per clip, one per speaker.")
    print("Example: 3,4,2  (speaker1=3, speaker2=4, speaker3=2)\n")

    all_clip_averages: list[float] = []
    clip_records: list[dict] = []

    for i, sentence in enumerate(sentences, start=1):
        while True:
            raw = input(f"Clip {i:02d} scores (s1,s2,s3): ").strip()
            try:
                parts = [float(x.strip()) for x in raw.split(",")]
                if len(parts) != 3:
                    raise ValueError("Need exactly 3 scores")
                if not all(1 <= v <= 5 for v in parts):
                    raise ValueError("Scores must be 1–5")
                break
            except ValueError as e:
                print(f"  Invalid input: {e}. Try again.")

        clip_avg = round(sum(parts) / len(parts), 2)
        all_clip_averages.append(clip_avg)
        clip_records.append({
            "clip_id": i,
            "sentence": sentence,
            "speaker_scores": parts,
            "clip_average": clip_avg,
        })
        print(f"  Clip {i:02d} average: {clip_avg:.2f}\n")

    scorer = Gate1Scorer()
    tts_score = scorer.score_tts(all_clip_averages)

    # Persist scores
    results_path = ROOT / "results" / "tts_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        for r in clip_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps({"type": "aggregate", "tts_score": tts_score}, ensure_ascii=False) + "\n")

    print("── TTS Gate 1 Score ─────────────────────────")
    print(f"Average naturalness : {tts_score['average']} / 5.0")
    print(f"Threshold           : ≥ {tts_score['threshold']}")
    print(f"Gate passed         : {'YES' if tts_score['passed'] else 'NO'}")
    if not tts_score["passed"]:
        print("\nVoxCPM2 does not support Urdu — ElevenLabs Multilingual v2 is now the mandatory TTS.")
    print(f"\nResults saved → {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxCPM2 TTS validation")
    parser.add_argument("--score", action="store_true", help="Enter naturalness scores from reviewers")
    args = parser.parse_args()

    cfg = load_config()

    if args.score:
        collect_scores()
    else:
        generate_audio(cfg)


if __name__ == "__main__":
    main()
