"""STT validation — Whisper Turbo via WhisperLiveKit WebSocket."""

import asyncio
import json
import sys
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


async def wait_for_server(host: str, port: int, timeout: int = 60) -> bool:
    import websockets

    url = f"ws://{host}:{port}"
    deadline = time.time() + timeout
    print(f"Waiting for WhisperLiveKit server at {url} ...")
    while time.time() < deadline:
        try:
            async with websockets.connect(url, open_timeout=3):
                print("Server is up.")
                return True
        except Exception:
            print(".", end="", flush=True)
            await asyncio.sleep(2)
    print()
    return False


async def transcribe_file(ws_url: str, wav_path: Path, language: str) -> dict:
    import websockets

    start = time.monotonic()
    transcript = ""

    audio_data = wav_path.read_bytes()
    chunks = [audio_data[i : i + 4096] for i in range(0, len(audio_data), 4096)]

    async with websockets.connect(ws_url) as ws:
        # Send config handshake expected by WhisperLiveKit
        config_msg = json.dumps({
            "uid": wav_path.stem,
            "language": language,
            "task": "transcribe",
            "model": "turbo",
            "use_vad": True,
        })
        await ws.send(config_msg)

        # Stream audio chunks
        for chunk in chunks:
            await ws.send(chunk)
            await asyncio.sleep(0.01)

        # Signal end of stream and collect response
        await ws.send(json.dumps({"uid": wav_path.stem, "eof": True}))

        # Collect responses for up to 10 seconds
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                data = json.loads(msg)
                if segments := data.get("segments"):
                    transcript = " ".join(s.get("text", "") for s in segments).strip()
                if data.get("message") == "DISCONNECT":
                    break
            except asyncio.TimeoutError:
                break

    latency_ms = int((time.monotonic() - start) * 1000)
    return {
        "transcript": transcript,
        "latency_ms": latency_ms,
        "language_detected": language,
    }


async def main() -> None:
    cfg = load_config()
    stt_cfg = cfg["models"]["stt"]
    host = "localhost"
    port = stt_cfg["port"]

    audio_dir = ROOT / "data" / "audio_samples"
    wav_files = sorted(audio_dir.glob("*.wav"))

    if not wav_files:
        print(
            "\nNo audio files found in data/audio_samples/\n"
            "Please upload .wav files (16kHz mono recommended) to this folder.\n"
            "Urdu audio: download from https://commonvoice.mozilla.org/ur/datasets\n"
            "Then re-run this script.\n"
        )
        sys.exit(0)

    server_up = await wait_for_server(host, port, timeout=60)
    if not server_up:
        print(
            "\nWhisperLiveKit server not running. Start it first with:\n"
            "  python -m whisper_live.server --port 9090 --backend faster_whisper\n"
        )
        sys.exit(1)

    logger = ResultLogger(
        model_name="Whisper Turbo",
        output_file=str(ROOT / "results" / "stt_results.json"),
    )
    ws_url = f"ws://{host}:{port}"

    print(f"\nRunning STT tests on {len(wav_files)} audio file(s)...\n")

    for i, wav_path in enumerate(wav_files, start=1):
        print(f"[{i}/{len(wav_files)}] Processing {wav_path.name} ...")
        try:
            result = await transcribe_file(ws_url, wav_path, stt_cfg["language"])
            status = "pass" if result["transcript"] else "empty"
            print(f"  Transcript : {result['transcript'][:80] or '(empty)'}")
            print(f"  Latency    : {result['latency_ms']} ms")
        except Exception as exc:
            result = {"transcript": "", "latency_ms": 9999, "language_detected": "error"}
            status = "error"
            print(f"  ERROR: {exc}")

        logger.log(
            test_id=f"stt_{i:02d}",
            category="urdu_audio",
            input_text=wav_path.name,
            output=result["transcript"],
            metadata={
                "latency_ms": result["latency_ms"],
                "language_detected": result["language_detected"],
                "status": status,
            },
        )

    print("\n── Summary ─────────────────────────────────")
    logger.summary()

    scorer = Gate1Scorer()
    stt_score = scorer.score_stt(logger.records)
    print(f"\np95 Latency : {stt_score['p95_latency_ms']} ms  (target: < {stt_score['threshold_ms']} ms)")
    print(f"Gate passed : {'YES' if stt_score['gate_passed'] else 'NO'}")

    # Append aggregate score to the results file
    score_record = {"type": "aggregate", "stt_score": stt_score}
    with open(ROOT / "results" / "stt_results.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(score_record, ensure_ascii=False) + "\n")

    print("\nResults saved → results/stt_results.json")


if __name__ == "__main__":
    asyncio.run(main())
