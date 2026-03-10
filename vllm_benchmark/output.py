import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def _status_text(passed: bool, label: str = None) -> Text:
    if passed:
        text = Text(label or "PASSED", style="bold green")
    else:
        text = Text(label or "FAILED", style="bold red")
    return text


def print_header(model: str, base_url: str):
    console.print()
    console.print(
        Panel(
            f"[bold]Model:[/bold]    {model}\n[bold]Base URL:[/bold] {base_url}",
            title="[bold]VLLM Benchmark[/bold]",
            border_style="blue",
        )
    )


def print_chat_result(result: dict, elapsed: float):
    table = Table(title="Function Calling (Chat) Benchmark", show_lines=True)
    table.add_column("Field", style="cyan", min_width=16)
    table.add_column("Value", min_width=40)

    table.add_row("Success", str(result["success"]))
    table.add_row("Tool Called", str(result.get("tool_called", False)))
    table.add_row("Tool Calls Count", str(result.get("tool_calls_count", 0)))

    for i, tr in enumerate(result.get("tool_results", [])):
        table.add_row(f"Tool [{i}] Name", tr.get("tool", ""))
        table.add_row(f"Tool [{i}] Input", str(tr.get("input", {})))
        table.add_row(f"Tool [{i}] Result", str(tr.get("result", "")))

    final = result.get("final_output", "")
    if len(final) > 200:
        final = final[:200] + "..."
    table.add_row("Final Output", final)

    console.print(table)
    status = _status_text(result["success"])
    console.print(f"  Result: ", end="")
    console.print(status, end="")
    console.print(f"  ({elapsed:.2f}s)")
    console.print()


def print_embedding_result(result: dict, elapsed: float):
    table = Table(title="Embedding Benchmark", show_lines=True)
    table.add_column("Field", style="cyan", min_width=16)
    table.add_column("Value", min_width=40)

    table.add_row("Success", str(result["success"]))
    table.add_row("Dimensions", str(result.get("dimensions", "N/A")))

    if result.get("error"):
        table.add_row("Error", result["error"])

    console.print(table)
    status = _status_text(result["success"])
    console.print(f"  Result: ", end="")
    console.print(status, end="")
    console.print(f"  ({elapsed:.2f}s)")
    console.print()


def print_vision_result(result: dict, elapsed: float):
    table = Table(title="Vision / OCR Benchmark", show_lines=True)
    table.add_column("Field", style="cyan", min_width=16)
    table.add_column("Value", min_width=40)

    table.add_row("Success", str(result["success"]))
    table.add_row("Vision Supported", str(result.get("vision_supported", False)))
    table.add_row("OCR Supported", str(result.get("ocr_supported", False)))

    test = result.get("test_results", {})
    if test:
        table.add_row("Expected Text", test.get("expected_text", ""))
        table.add_row("Extracted Text", str(test.get("extracted_text", "")))
        table.add_row("Accuracy", f"{test.get('accuracy', 0):.1f}%")

    if result.get("error"):
        table.add_row("Error", result["error"])

    console.print(table)

    if result.get("vision_supported") and not result.get("ocr_supported"):
        status = _status_text(False, "PARTIAL")
    else:
        status = _status_text(result["success"])

    console.print(f"  Result: ", end="")
    console.print(status, end="")
    console.print(f"  ({elapsed:.2f}s)")
    console.print()


def print_summary(results: dict):
    console.print()
    total = len(results)
    passed = sum(1 for r in results.values() if r["result"]["success"])
    failed = total - passed

    if failed == 0:
        style = "bold green"
        msg = f"All {total} benchmark(s) passed"
    else:
        style = "bold red"
        msg = f"{failed}/{total} benchmark(s) failed"

    console.print(Panel(msg, title="[bold]Summary[/bold]", border_style=style))


def print_json_results(results: dict):
    output = {}
    for name, data in results.items():
        output[name] = data["result"]
    print(json.dumps(output, indent=2, default=str))
