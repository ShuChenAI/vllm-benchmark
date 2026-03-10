import asyncio
import enum
import logging
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools.structured import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import pytz

logger = logging.getLogger(__name__)

ALL_PYTZ_TIMEZONES = {x.replace("/", "_"): x for x in pytz.all_timezones}
ALL_PYTZ_TIMEZONES_ENUM = enum.Enum("ALL_PYTZ_TIMEZONES", ALL_PYTZ_TIMEZONES)


class Timezone(BaseModel):
    timezone: ALL_PYTZ_TIMEZONES_ENUM = Field(..., description="PyTZ timezone")


def get_iso_format_datetime(timezone: ALL_PYTZ_TIMEZONES_ENUM) -> str:
    return datetime.now(pytz.timezone(timezone.value)).isoformat()


DATETIME = StructuredTool.from_function(
    name="ISO-Datetime-Getter",
    func=get_iso_format_datetime,
    description="Returns the time in ISO format for the given timezone, input should be one of PyTZ timezones.",
    args_schema=Timezone,
    handle_tool_error=True,
    handle_validation_error=True,
)


async def vllm_chat_benchmark(
    base_url: str, api_key: str, model: str, model_args: dict
) -> dict:
    """
    Benchmark VLLM agent with tool usage verification.

    Returns:
        dict: Benchmark results including tool usage verification
    """
    llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, **model_args)
    tools = [DATETIME]
    system_prompt = "You are a helpful assistant that can utilize tools to check datetime from specific timezones."

    logger.info("Starting VLLM chat benchmark with tool verification")
    logger.info(f"  Model: {model}")
    logger.info(f"  Base URL: {base_url}")

    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="What time is it in Asia/Taipei?")]}
    )

    tool_used = False
    tool_results = []
    final_output = ""

    if isinstance(response, dict) and "messages" in response:
        messages_list = response["messages"]

        for i, msg in enumerate(messages_list):
            if (
                isinstance(msg, AIMessage)
                and hasattr(msg, "tool_calls")
                and msg.tool_calls
            ):
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "unknown")

                    if tool_name == "ISO-Datetime-Getter":
                        tool_used = True

                        tool_result = None
                        for j in range(i + 1, len(messages_list)):
                            if isinstance(messages_list[j], ToolMessage):
                                if hasattr(
                                    messages_list[j], "tool_call_id"
                                ) and messages_list[j].tool_call_id == tool_call.get(
                                    "id"
                                ):
                                    tool_result = messages_list[j].content
                                    break

                        tool_results.append(
                            {
                                "tool": tool_name,
                                "input": tool_call.get("args", {}),
                                "result": tool_result,
                            }
                        )

                        logger.info(f"Tool '{tool_name}' was successfully called")
                        logger.info(f"  Input: {tool_call.get('args', {})}")
                        logger.info(f"  Result: {tool_result}")

            if isinstance(msg, AIMessage) and hasattr(msg, "content"):
                final_output = msg.content

    if tool_used:
        logger.info("BENCHMARK PASSED: Agent successfully utilized the tool")
    else:
        logger.warning("BENCHMARK FAILED: Agent did not utilize the tool")

    logger.info(f"Final output: {final_output}")

    return {
        "success": tool_used,
        "tool_called": tool_used,
        "tool_results": tool_results,
        "final_output": final_output,
        "tool_calls_count": len(tool_results),
    }


def run_chat_benchmark(
    base_url: str, api_key: str, model: str, model_args: dict
) -> dict:
    """Synchronous entry point for the chat benchmark."""
    return asyncio.run(vllm_chat_benchmark(base_url, api_key, model, model_args))
