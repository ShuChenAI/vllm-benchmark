import difflib
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Sample base64 encoded image with text "HELLO WORLD"
# Simple white background with black text for testing
TEST_IMAGE_BASE64 = "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"

EXPECTED_TEXT = "HELLO WORLD"


def run_vision_benchmark(
    base_url: str, api_key: str, model: str, model_args: dict = None
) -> dict:
    """
    Benchmark VLLM model's vision/OCR capabilities with simple English text extraction.

    Returns:
        dict: Benchmark results with vision/OCR capability verification
    """
    if model_args is None:
        model_args = {}

    try:
        llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, **model_args)

        logger.info("Starting vision/OCR capability benchmark")
        logger.info(f"  Model: {model}")
        logger.info(f"  Base URL: {base_url}")

        system_message = SystemMessage(
            content="You are a helpful assistant that able to extract text from image."
        )

        vision_message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Please extract and return ONLY the text you see in this image. Do not add any explanation, just return the exact text.",
                },
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": TEST_IMAGE_BASE64,
                    "mime_type": "image/jpeg",
                },
            ]
        )

        logger.info("Testing OCR capability with sample image...")

        response = llm.invoke([system_message, vision_message])
        extracted_text = (
            response.content.strip()
            if hasattr(response, "content")
            else str(response).strip()
        )

        logger.info(f"  Expected text: '{EXPECTED_TEXT}'")
        logger.info(f"  Extracted text: '{extracted_text}'")

        vision_supported = False
        ocr_accurate = False

        if extracted_text:
            vision_supported = True
            if EXPECTED_TEXT.lower() in extracted_text.lower():
                ocr_accurate = True
                accuracy = 100.0
            else:
                similarity = difflib.SequenceMatcher(
                    None, EXPECTED_TEXT.lower(), extracted_text.lower()
                ).ratio()
                accuracy = similarity * 100
                ocr_accurate = accuracy > 80
        else:
            accuracy = 0.0

        if vision_supported and ocr_accurate:
            logger.info(f"BENCHMARK PASSED: OCR accuracy {accuracy:.1f}%")
        elif vision_supported:
            logger.warning(
                f"BENCHMARK PARTIAL: Vision supported but accuracy low ({accuracy:.1f}%)"
            )
        else:
            logger.warning("BENCHMARK FAILED: No vision/OCR support detected")

        return {
            "success": vision_supported and ocr_accurate,
            "vision_supported": vision_supported,
            "ocr_supported": ocr_accurate,
            "test_results": {
                "expected_text": EXPECTED_TEXT,
                "extracted_text": extracted_text,
                "accuracy": accuracy,
                "passed": ocr_accurate,
            },
            "error": None,
        }

    except Exception as e:
        error_msg = str(e)
        logger.warning(f"Vision test failed with error: {error_msg}")

        is_vision_error = any(
            kw in error_msg.lower() for kw in ("image", "vision", "multimodal")
        )
        if is_vision_error:
            logger.warning("Model does not support vision/image inputs")
        else:
            logger.warning(f"Unexpected error: {error_msg}")

        return {
            "success": False,
            "vision_supported": False,
            "ocr_supported": False,
            "test_results": {
                "expected_text": EXPECTED_TEXT,
                "extracted_text": None,
                "accuracy": 0.0,
                "passed": False,
            },
            "error": error_msg,
        }
