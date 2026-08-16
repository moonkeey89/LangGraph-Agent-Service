from langgraph.prebuilt import ToolNode


from tools import (
    get_weather,
    calculate,
    search_attraction
)


tools=[
    get_weather,
    calculate,
    search_attraction
]


tool_node=ToolNode(tools)