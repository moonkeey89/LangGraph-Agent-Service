from langgraph.graph import StateGraph, START, END
from state import AgentState


class AgentGraph:
    def __init__(self):
        self.graph = StateGraph(AgentState)

    def add_node(self, name, function):
        self.graph.add_node(name, function)

    def add_edge(self, start, end):
        if end is None:
            end = END
        self.graph.add_edge(start, end)

    def compile(self):
        self.graph.add_edge(START, "planner")
        return self.graph.compile()

    def run(self, state):
        app = self.compile()
        return app.invoke(state)
