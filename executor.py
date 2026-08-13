import json

from skills.weather import get_weather
from skills.attraction import search_attraction
from skills.calculator import  calculate


class Executor:


    def __init__(self):

        self.tools = {

            "get_weather": get_weather,
            "search_attraction": search_attraction,
            "calculate": calculate

        }

    def execute(self, state):
        print("Executor收到Plan:")
        print(state.plan)

        results = []

        try:

            tasks = json.loads(state.plan)

            for task in tasks:
                tool_name = task["tool"]

                arguments = task["arguments"]

                tool = self.tools[tool_name]

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