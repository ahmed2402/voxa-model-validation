import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


class ResultLogger:
    def __init__(self, model_name: str, output_file: str):
        self.model_name = model_name
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []

    def log(
        self,
        test_id: str,
        category: str,
        input_text: str,
        output: str,
        metadata: dict,
    ) -> None:
        record = {
            "test_id": test_id,
            "model": self.model_name,
            "category": category,
            "input": input_text,
            "output": output,
            "timestamp": datetime.utcnow().isoformat(),
            **metadata,
        }
        self.records.append(record)
        with self.output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def summary(self) -> None:
        table = Table(title=f"Results — {self.model_name}", show_lines=True)
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Category", style="magenta", width=20)
        table.add_column("Status", style="green", width=10)
        table.add_column("Latency (ms)", style="yellow", width=14)

        for r in self.records:
            latency = str(r.get("latency_ms", "—"))
            status = str(r.get("status", r.get("manual_pass", "—")))
            if status == "True" or status == "pass":
                status_str = "[green]PASS[/green]"
            elif status == "False" or status == "fail":
                status_str = "[red]FAIL[/red]"
            else:
                status_str = f"[yellow]{status}[/yellow]"
            table.add_row(str(r["test_id"]), r["category"], status_str, latency)

        console.print(table)

    def save_report(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Results Report — {self.model_name}",
            f"\nGenerated: {datetime.utcnow().isoformat()}Z",
            f"\nTotal records: {len(self.records)}",
            "\n| ID | Category | Status | Latency (ms) |",
            "|---|---|---|---|",
        ]
        for r in self.records:
            latency = str(r.get("latency_ms", "—"))
            status = str(r.get("status", r.get("manual_pass", "—")))
            lines.append(f"| {r['test_id']} | {r['category']} | {status} | {latency} |")
        path.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[dim]Report saved → {filepath}[/dim]")
