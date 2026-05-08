"""Aggregate Gate 1 results and produce the final pass/fail report."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from utils.scorer import Gate1Scorer


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def require_file(path: Path, label: str) -> list[dict]:
    if not path.exists():
        print(f"ERROR: {path} not found.\nRun the corresponding test script first: {label}")
        sys.exit(1)
    return load_jsonl(path)


def extract_aggregate(records: list[dict], key: str) -> dict | None:
    for r in records:
        if r.get("type") == "aggregate" and key in r:
            return r[key]
    return None


def main() -> None:
    results_dir = ROOT / "results"

    stt_records = require_file(results_dir / "stt_results.json",   "python scripts/01_test_whisper.py")
    llm_records = require_file(results_dir / "llm_results.json",   "python scripts/02_test_qwen.py")
    tts_records = require_file(results_dir / "tts_results.json",   "python scripts/03_test_voxcpm2.py --score")

    scorer = Gate1Scorer()

    # ── STT ──────────────────────────────────────────────────────────
    stt_score = extract_aggregate(stt_records, "stt_score")
    if stt_score is None:
        raw_stt = [r for r in stt_records if "latency_ms" in r]
        stt_score = scorer.score_stt(raw_stt)

    # ── LLM ──────────────────────────────────────────────────────────
    llm_raw = [r for r in llm_records if r.get("type") != "aggregate"]
    llm_score = scorer.score_llm(llm_raw)

    # ── TTS ──────────────────────────────────────────────────────────
    tts_score = extract_aggregate(tts_records, "tts_score")
    if tts_score is None:
        clip_avgs = [r["clip_average"] for r in tts_records if "clip_average" in r]
        if not clip_avgs:
            print(
                "ERROR: tts_results.json contains generation data but no scores.\n"
                "Run:  python scripts/03_test_voxcpm2.py --score"
            )
            sys.exit(1)
        tts_score = scorer.score_tts(clip_avgs)

    # ── Report ───────────────────────────────────────────────────────
    report = scorer.generate_gate1_report(stt_score, llm_score, tts_score)

    report_path = results_dir / "gate1_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nReport saved → {report_path}")

    overall_passed = (
        stt_score.get("gate_passed", False)
        and llm_score.get("gate_passed", False)
        and tts_score.get("passed", False)
    )
    sys.exit(0 if overall_passed else 1)


if __name__ == "__main__":
    main()
