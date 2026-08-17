from langchain_core.tools import tool

from skills import (
    calculate as calculate_skill,
    get_weather as get_weather_skill,
    search_attraction as search_attraction_skill,
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


TOOLS = [
    get_weather,
    calculate,
    search_attraction,
]
