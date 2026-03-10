# VLLM Benchmark

Standalone CLI tool for benchmarking VLLM offline deployments. Validates function calling (tool use), embedding, and vision/OCR capabilities.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run all benchmarks
python -m vllm_benchmark --base-url http://localhost:8000/v1 --model Qwen/Qwen3-VL-8B-Thinking --all

# Run specific benchmarks
python -m vllm_benchmark --base-url http://localhost:8000/v1 --model my-model --chat --vision

# With API key
python -m vllm_benchmark --base-url http://localhost:8000/v1 --model my-model --all --api-key sk-xxx

# With extra model args
python -m vllm_benchmark --base-url http://localhost:8000/v1 --model my-model --chat --model-args '{"temperature": 0}'

# JSON output (for scripting)
python -m vllm_benchmark --base-url http://localhost:8000/v1 --model my-model --all --json

# Verbose logging
python -m vllm_benchmark --base-url http://localhost:8000/v1 --model my-model --all --verbose
```

## Benchmarks

### Chat (Function Calling)
Tests whether the model can correctly use tools via a LangChain ReAct agent. Sends a timezone query and verifies the model calls the `ISO-Datetime-Getter` tool.

### Embedding
Tests the embedding model by embedding a sample string and returning the vector dimensions.

### Vision / OCR
Tests vision capabilities by sending a base64-encoded image containing "HELLO WORLD" and verifying the model can extract the text.

## API Key Resolution

The CLI resolves API keys in this order:
1. `--api-key` argument
2. `VLLM_API_KEY` environment variable
3. `OPENAI_API_KEY` environment variable
4. `"EMPTY"` (default for VLLM servers without auth)

## Exit Codes

- `0` — All selected benchmarks passed
- `1` — At least one benchmark failed
- `2` — CLI argument error
