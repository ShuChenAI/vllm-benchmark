import asyncio
import logging

from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


async def _async_vllm_embedding_benchmark(
    base_url: str, api_key: str, model: str
) -> int:
    """
    Async version of VLLM embedding benchmark.
    Tests the embedding model and returns the embedding dimensions.
    """
    embedding = OpenAIEmbeddings(model=model, base_url=base_url, api_key=api_key)

    example_string = "You should stay, study and sprint."
    embedding_result = await embedding.aembed_query(example_string)

    return len(embedding_result)


def run_embedding_benchmark(base_url: str, api_key: str, model: str) -> dict:
    """
    Synchronous entry point for the embedding benchmark.

    Returns:
        dict: Benchmark results with dimensions info
    """
    logger.info("Starting VLLM embedding benchmark")
    logger.info(f"  Model: {model}")
    logger.info(f"  Base URL: {base_url}")

    try:
        dimensions = asyncio.run(
            _async_vllm_embedding_benchmark(base_url, api_key, model)
        )
        logger.info(f"BENCHMARK PASSED: Embedding dimensions = {dimensions}")
        return {
            "success": True,
            "dimensions": dimensions,
        }
    except Exception as e:
        logger.warning(f"BENCHMARK FAILED: {e}")
        return {
            "success": False,
            "dimensions": None,
            "error": str(e),
        }
