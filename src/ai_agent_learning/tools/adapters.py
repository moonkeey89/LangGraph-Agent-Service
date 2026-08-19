from langchain_core.tools import tool
from langgraph.types import interrupt

from ai_agent_learning.skills import (
    calculate as calculate_skill,
    get_weather as get_weather_skill,
    save_memory as save_memory_skill,
    search_attraction as search_attraction_skill,
    run_unstable_operation as unstable_operation_skill,
)


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""

    return get_weather_skill(city)


@tool
def calculate(expression: str) -> str:
    """计算给定的数学表达式。"""

    return calculate_skill(expression)


@tool
def search_attraction(city: str) -> list[str] | str:
    """查询指定城市的旅游景点。"""

    return search_attraction_skill(city)


@tool
def unstable_tool(task: str) -> str:
    """执行无副作用的教学任务；临时超时时由Graph进行有限重试。"""

    return unstable_operation_skill(task)


@tool
def save_memory(content: str) -> str:
    """保存用户明确要求记住的信息；执行前必须获得人工批准。"""

    decision = interrupt(
        {
            "action": "save_user_memory",
            "tool_name": "save_memory",
            "arguments": {"content": content},
            "message": "该操作将写入一条模拟记忆，是否批准？",
        }
    )

    if not isinstance(decision, dict) or decision.get("approved") is not True:
        reason = (
            decision.get("reason", "未获得明确批准")
            if isinstance(decision, dict)
            else "未获得明确批准"
        )
        return f"保存操作已取消：{reason}"

    return save_memory_skill(content)


TOOLS = [
    get_weather,
    calculate,
    search_attraction,
    unstable_tool,
    save_memory,
]
