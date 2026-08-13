import json

from skills.weather import get_weather
from skills.attraction import search_attraction
from skills.calculator import  calculate


class Executor:


    def __init__(self, registry):
        self.registry = registry


    def execute(self, state):
        print("Executor收到Plan:")
        print(state.plan)

        results = []

        try:

            tasks = json.loads(state.plan)

            for task in tasks:
                tool_name = task["tool"]
                arguments = task["arguments"]

                tool = self.registry.get_tool(tool_name)

                result = tool(**arguments)

                results.append(
                    {
                        "tool": tool_name,
                        "result": result
                    }
                )

            state.results = results

            # 执行成功
            state.status = "success"


        except Exception as e:

            state.status = "failed"

            state.results.append(
                {
                    "error": str(e)
                }
            )

        return state