from datetime import datetime


class Gate1Scorer:
    LLM_PASS_THRESHOLD = 0.80
    TTS_NATURALNESS_THRESHOLD = 3.0
    STT_LATENCY_THRESHOLD_MS = 500

    def score_llm(self, results: list[dict]) -> dict:
        total = len(results)
        if total == 0:
            return {"total": 0, "passed": 0, "pass_rate": 0.0, "threshold": self.LLM_PASS_THRESHOLD, "gate_passed": False}

        passed = sum(1 for r in results if r.get("manual_pass") is True)
        pass_rate = passed / total
        return {
            "total": total,
            "passed": passed,
            "pass_rate": round(pass_rate, 4),
            "threshold": self.LLM_PASS_THRESHOLD,
            "gate_passed": pass_rate >= self.LLM_PASS_THRESHOLD,
        }

    def score_tts(self, scores: list[float]) -> dict:
        if not scores:
            return {"average": 0.0, "passed": False, "threshold": self.TTS_NATURALNESS_THRESHOLD}
        average = round(sum(scores) / len(scores), 2)
        return {
            "average": average,
            "passed": average >= self.TTS_NATURALNESS_THRESHOLD,
            "threshold": self.TTS_NATURALNESS_THRESHOLD,
            "sample_count": len(scores),
        }

    def score_stt(self, results: list[dict]) -> dict:
        latencies = sorted(r["latency_ms"] for r in results if "latency_ms" in r)
        if not latencies:
            return {"p95_latency_ms": None, "threshold_ms": self.STT_LATENCY_THRESHOLD_MS, "gate_passed": False}
        idx = int(len(latencies) * 0.95)
        p95 = latencies[min(idx, len(latencies) - 1)]
        return {
            "p95_latency_ms": p95,
            "min_ms": latencies[0],
            "max_ms": latencies[-1],
            "median_ms": latencies[len(latencies) // 2],
            "sample_count": len(latencies),
            "threshold_ms": self.STT_LATENCY_THRESHOLD_MS,
            "gate_passed": p95 < self.STT_LATENCY_THRESHOLD_MS,
        }

    def generate_gate1_report(self, stt: dict, llm: dict, tts: dict) -> str:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        stt_pass = stt.get("gate_passed", False)
        llm_pass = llm.get("gate_passed", False)
        tts_pass = tts.get("passed", False)
        overall = stt_pass and llm_pass and tts_pass

        def badge(passed: bool) -> str:
            return "**PASSED**" if passed else "**FAILED**"

        tts_warning = ""
        if not tts_pass:
            tts_warning = (
                "\n> **TTS FALLBACK REQUIRED:** VoxCPM2 does not support Urdu — "
                "ElevenLabs Multilingual v2 is now the mandatory TTS for VOXA.\n"
            )

        verdict = (
            "## GATE 1 PASSED\n\nAll three models meet production thresholds. Pipeline development may proceed."
            if overall
            else "## GATE 1 FAILED — ElevenLabs fallback required\n\nOne or more models did not meet the Gate 1 threshold. Review the table above and take corrective action before proceeding."
        )

        report = f"""# VOXA Gate 1 Validation Report

**Generated:** {now}
**Environment:** RunPod RTX 4090 (24 GB VRAM)

---

## Results Summary

| Model | Metric | Result | Threshold | Status |
|-------|--------|--------|-----------|--------|
| Whisper Turbo (STT) | p95 latency | {stt.get("p95_latency_ms", "N/A")} ms | < {stt.get("threshold_ms", 500)} ms | {badge(stt_pass)} |
| Qwen3.5-9B (LLM) | Pass rate | {llm.get("passed", 0)}/{llm.get("total", 20)} ({round(llm.get("pass_rate", 0) * 100, 1)}%) | ≥ 80% | {badge(llm_pass)} |
| VoxCPM2 (TTS) | Urdu naturalness avg | {tts.get("average", "N/A")} / 5.0 | ≥ {tts.get("threshold", 3.0)} / 5.0 | {badge(tts_pass)} |

---

## STT Detail

- **p95 Latency:** {stt.get("p95_latency_ms", "N/A")} ms (target: < {stt.get("threshold_ms", 500)} ms)
- **Min latency:** {stt.get("min_ms", "N/A")} ms
- **Max latency:** {stt.get("max_ms", "N/A")} ms
- **Median latency:** {stt.get("median_ms", "N/A")} ms
- **Samples tested:** {stt.get("sample_count", "N/A")}
- **Status:** {badge(stt_pass)}

## LLM Detail

- **Prompts tested:** {llm.get("total", 20)}
- **Prompts passed (manual review):** {llm.get("passed", 0)}
- **Pass rate:** {round(llm.get("pass_rate", 0) * 100, 1)}% (threshold: ≥ 80%)
- **Status:** {badge(llm_pass)}

## TTS Detail

- **Sentences generated:** {tts.get("sample_count", 10)}
- **Average naturalness score:** {tts.get("average", "N/A")} / 5.0 (threshold: ≥ {tts.get("threshold", 3.0)})
- **Status:** {badge(tts_pass)}
{tts_warning}
---

{verdict}
"""
        return report
