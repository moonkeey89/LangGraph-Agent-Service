from openai import OpenAI
from dotenv import load_dotenv
import os

from planner import Planner
from executor import Executor
from response_generator import ResponseGenerator
from state import AgentState
from graph import AgentGraph
from registry import ToolRegistry

from skills.weather import get_weather
from skills.attraction import search_attraction
from skills.calculator import calculate

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 创建Tool Registry
registry = ToolRegistry()

registry.register("get_weather", get_weather)
registry.register("search_attraction", search_attraction)
registry.register("calculate", calculate)

# 创建模块
planner = Planner(client)
executor = Executor(registry)
generator = ResponseGenerator(client)

# 创建Graph
graph = AgentGraph()

# 注册Node
graph.add_node("planner", planner.create_plan)
graph.add_node("executor", executor.execute)
graph.add_node("response", generator.generate)

# 注册Edge
graph.add_edge("planner", "executor")
graph.add_edge("executor", "response")
graph.add_edge("response", None)

# 编译Graph
app = graph.compile()

while True:
    user_input = input("\n请输入任务:")

    if user_input == "exit":
        break

    state: AgentState = {
        "task": user_input,
        "plan": [],
        "results": [],
        "final_answer": ""
    }

    final_state = app.invoke(state)

    print("\nAgent回答:")
    print(final_state["final_answer"])