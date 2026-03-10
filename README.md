# VLLM Benchmark

Standalone CLI tool for benchmarking VLLM offline deployments. Validates function calling (tool use), embedding, and vision/OCR capabilities.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m vllm_benchmark --base-url <URL> --model <MODEL> [OPTIONS]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--base-url` | Yes | VLLM server base URL (e.g. `http://localhost:8000/v1`) |
| `--model` | Yes | Model name / identifier (e.g. `Qwen/Qwen3-VL-8B-Thinking`) |
| `--api-key` | No | API key (see [API Key Resolution](#api-key-resolution) for fallback order) |
| `--chat` | No | Run function-calling (chat) benchmark |
| `--embedding` | No | Run embedding benchmark |
| `--vision` | No | Run vision/OCR benchmark |
| `--all` | No | Run all benchmarks |
| `--model-args` | No | JSON string of extra model kwargs (e.g. `'{"temperature": 0}'`) |
| `--json` | No | Output results as JSON (for scripting) |
| `--verbose` | No | Enable debug logging |

At least one benchmark flag (`--chat`, `--embedding`, `--vision`, or `--all`) is required.

> **Note:** A single vLLM instance serves one model type. Use `--chat` and `--vision` against chat/VL models, and `--embedding` against embedding models. The `--all` flag is only useful if your model supports all three capabilities (rare). In practice, run benchmarks separately against each deployment.

### Examples

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

## vLLM Server Setup

### Qwen3-VL (Vision + Reasoning + Tool Calling)

```bash
vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --max-model-len 131072 \
  --host 0.0.0.0 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1 \
  --max-num-seqs 10 \
  --limit-mm-per-prompt.video 0 \
  --async-scheduling \
  --tensor-parallel-size 4
```

### Qwen3.5 (Natively Multimodal + Reasoning + Tool Calling)

All Qwen3.5 models are natively multimodal (vision + text) — there is no separate `-VL` variant.

```bash
vllm serve Qwen/Qwen3.5-4B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --enable-prefix-caching \
  --max-num-seqs 10
```

### Qwen3-Embedding

```bash
vllm serve Qwen/Qwen3-Embedding-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code
```

### Key Flags

| Flag | Description |
|------|-------------|
| `--tool-call-parser` | Parser for structured tool calls. Use `hermes` for Qwen3, `qwen3_coder` for Qwen3.5 |
| `--reasoning-parser` | Parser for thinking/reasoning content. Use `deepseek_r1` for Qwen3, `qwen3` for Qwen3.5 |
| `--enable-auto-tool-choice` | Let the model decide when to call tools |
| `--tensor-parallel-size N` | Distribute the model across N GPUs |
| `--mm-encoder-tp-mode data` | Data-parallel vision encoder for better throughput (Qwen3.5 VL) |
| `--mm-processor-cache-type shm` | Shared memory cache for preprocessed multimodal inputs (Qwen3.5 VL) |
| `--enable-prefix-caching` | Cache common prompt prefixes for faster inference |
| `--trust-remote-code` | Required for some models (e.g. embedding models) |
| `--limit-mm-per-prompt.video 0` | Disable video input processing |
| `--max-num-seqs N` | Max concurrent sequences |
| `--max-model-len N` | Max context length |

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
