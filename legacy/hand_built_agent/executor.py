import json


class Executor:
    def __init__(self, registry):
        self.registry = registry

    def execute(self, state):
        print("Executor收到Plan:")
        print(state["plan"])

        results = []

        try:
            tasks = json.loads(state["plan"])

            for task in tasks:
                tool_name = task["tool"]
                arguments = task["arguments"]
                tool = self.registry.get_tool(tool_name)
                result = tool(**arguments)

                results.append({"tool": tool_name, "result": result})

            state["results"] = results
        except Exception as error:
            state["results"].append({"error": str(error)})

        return state
