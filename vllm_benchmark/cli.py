import argparse
import json
import logging
import os
import sys
import time


def _resolve_api_key(args_key: str | None) -> str:
    if args_key:
        return args_key
    for env_var in ("VLLM_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(env_var)
        if val:
            return val
    return "EMPTY"


def main():
    parser = argparse.ArgumentParser(
        prog="vllm_benchmark",
        description="Standalone CLI tool for benchmarking VLLM offline deployments (function calling, embedding, vision/OCR).",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="VLLM server base URL (e.g. http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name / identifier",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (falls back to VLLM_API_KEY, then OPENAI_API_KEY env vars, then 'EMPTY')",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Run function-calling (chat) benchmark",
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Run embedding benchmark",
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Run vision/OCR benchmark",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--model-args",
        default=None,
        help='JSON string of extra model kwargs (e.g. \'{"temperature": 0}\')',
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(name)s - %(levelname)s - %(message)s")

    # Validate benchmark selection
    if not any([args.chat, args.embedding, args.vision, args.run_all]):
        parser.error(
            "No benchmark selected. Use --chat, --embedding, --vision, or --all."
        )
        sys.exit(2)

    if args.run_all:
        args.chat = True
        args.embedding = True
        args.vision = True

    # Parse model_args
    model_args = {}
    if args.model_args:
        try:
            model_args = json.loads(args.model_args)
        except json.JSONDecodeError as e:
            parser.error(f"Invalid JSON for --model-args: {e}")
            sys.exit(2)

    api_key = _resolve_api_key(args.api_key)

    # Import here to avoid slow startup when just checking --help
    from vllm_benchmark.benchmarks import (
        run_chat_benchmark,
        run_embedding_benchmark,
        run_vision_benchmark,
    )
    from vllm_benchmark.output import (
        print_chat_result,
        print_embedding_result,
        print_header,
        print_json_results,
        print_summary,
        print_vision_result,
    )

    if not args.json_output:
        print_header(args.model, args.base_url)

    results = {}

    # Chat benchmark
    if args.chat:
        if not args.json_output:
            from rich.console import Console
            Console().print("\n[bold yellow]Running chat (function calling) benchmark...[/bold yellow]")
        start = time.time()
        try:
            result = run_chat_benchmark(args.base_url, api_key, args.model, model_args)
        except Exception as e:
            result = {"success": False, "error": str(e)}
        elapsed = time.time() - start
        results["chat"] = {"result": result, "elapsed": elapsed}
        if not args.json_output:
            print_chat_result(result, elapsed)

    # Embedding benchmark
    if args.embedding:
        if not args.json_output:
            from rich.console import Console
            Console().print("[bold yellow]Running embedding benchmark...[/bold yellow]")
        start = time.time()
        try:
            result = run_embedding_benchmark(args.base_url, api_key, args.model)
        except Exception as e:
            result = {"success": False, "error": str(e)}
        elapsed = time.time() - start
        results["embedding"] = {"result": result, "elapsed": elapsed}
        if not args.json_output:
            print_embedding_result(result, elapsed)

    # Vision benchmark
    if args.vision:
        if not args.json_output:
            from rich.console import Console
            Console().print("[bold yellow]Running vision/OCR benchmark...[/bold yellow]")
        start = time.time()
        try:
            result = run_vision_benchmark(
                args.base_url, api_key, args.model, model_args
            )
        except Exception as e:
            result = {"success": False, "error": str(e)}
        elapsed = time.time() - start
        results["vision"] = {"result": result, "elapsed": elapsed}
        if not args.json_output:
            print_vision_result(result, elapsed)

    # Output
    if args.json_output:
        print_json_results(results)
    else:
        print_summary(results)

    # Exit code
    any_failed = any(not r["result"]["success"] for r in results.values())
    sys.exit(1 if any_failed else 0)
