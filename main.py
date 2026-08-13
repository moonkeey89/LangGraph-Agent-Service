from openai import OpenAI
from dotenv import load_dotenv
import os

from planner import Planner
from executor import Executor
from response_generator import ResponseGenerator
from state import AgentState
from graph import AgentGraph
from router_node import RouterNode

from router import entry_router

from router import router


load_dotenv()


client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

router_node=RouterNode(client)

# 创建Graph
graph = AgentGraph()


# 创建模块
planner = Planner(client)

executor = Executor()

generator = ResponseGenerator(client)



# 注册Node

graph.add_node(
    "router",
    router_node.decide
)

graph.add_node(
    "planner",
    planner.create_plan
)


graph.add_node(
    "executor",
    executor.execute
)


graph.add_node(
    "response",
    generator.generate
)



# 注册Edge
graph.add_conditional_edge(
    "router",
    entry_router
)

graph.add_edge(
    "planner",
    "executor"
)

graph.add_conditional_edge(
    "executor",
    router
)

graph.add_edge(
    "response",
    None
)


state=AgentState()
while True:

    user_input=input("\n请输入任务:")


    if user_input=="exit":
        break


    # state=AgentState(user_input)
    state.task=user_input


    final_state=graph.run(state)


    print("\nAgent回答:")

    # print("git change test??")

    print(final_state.final_answer)