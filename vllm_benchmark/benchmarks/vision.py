import difflib
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# Sample base64 encoded image with text "HELLO WORLD"
# Simple white background with black text for testing
TEST_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAMgAAABkCAIAAABM5OhcAAAD2UlEQVR4nO3czSt0bQDHcfPgVkpk5y8g08zkSJw5XuZF9v4DK7splFjYsJlY2FsZW/wBzCgsSJNGjbwk2ZCsjKTojOY8i/M859GNcYtfqef7WZ3rOnOduZy+OScLPsdxKoDv9te3XxEgLKjwGwsShAUJwoIEYUGCsCBBWJAgLEgQFiQICxKEBQnCggRhQYKwIEFYkCAsSBAWJAgLEoQFCcKCBGFBgrAgQViQICxIEBYkCAsShAUJwoIEYUGCsPAjw2poaHhzWFtbG/nX/Pz860++ObO4uNje3m6aZnt7+9LSkjc/NDS0srLiHodCodHRUfd4ZGRkdXX1vYXuHvr6+gzD2N7efvMb3c9Eo1HLslKp1BfvBv7jfE19ff2bw9/m/2RmbW3NsqxCoeA4TqFQsCwrk8m4pxYWFsbHxx3Hub+/NwzDNE13vqur6+bm5r2F3vXz+XwgECi/h4eHh1gstry8/LX7gX/8oLDi8fju7q433NnZ6e/vd48PDw+j0ajjOOl0empqKhQKPT092bYdDAbLLPSuXyqVGhsbP9zD/v6+ZVmfvwd4Q1XFj3FyctLW1uYNDcM4Pj52j/1+/8XFheM4u7u7PT0919fXBwcHlZWVHR0d5Re60ul0LBb7cAPBYPD8/Pxbf6b/r6+GZdt2JBJ5OXw9n0wmTdP87JUdx/H5fO6xz+draWk5OzvLZrNjY2OXl5d7e3tVVVW9vb1lFrp7KBaLp6enR0dHH37j8/NzdXX1Z/cJSVi/fv3a2tryht7b8W/zf6K1tTWXy4XDYXeYy+X8fr93NhwOZ7PZx8fHurq6cDg8PT1dXV09MzNTZqG3h7m5uVQqNTk5WX4D2Ww2EAh8as94l/Nj3rHW19cty7q7u/PewTc2NryzmUwmHo8nEgn3nckwDPetq8xC7/q5XG5wcLD8Hm5vbzs7Ozc3N79wM6B/x3r5KDRNM5lM2rbd3d3tzliWNTs7+3rm6uoqGo3W1NTYtp1IJOLxuHfBrq6u7e3t4eFh98nY1NRUX1/vnhoYGCizsKKiorm5OZ/Pl0qlN/cQiUR8Pl+xWJyYmHj5WMdX+Pg/71DgL++QICxIEBYkCAsShAUJwoIEYUGCsCBBWJAgLEgQFiQICxKEBQnCggRhQYKwIEFYkCAsSBAWJAgLEoQFCcKCBGFBgrAgQViQICxIEBYkCAsShAUJwoIEYUGCsCBBWJAgLEgQFiQICxKEBQnCggRhQYKwIEFYkCAsSBAWJAgLEoQFCcKCBGFBgrAgQViQICxIEBYkCAsShAUJwoIEYUGCsCBBWJAgLEgQFioU/gYeDUILqmg2UwAAAABJRU5ErkJggg=="

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
                    "mime_type": "image/png",
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
