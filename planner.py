import json
from tools_schema import tools_schema

class Planner:
    def __init__(self, client):
        self.client = client

    def create_plan(self, state):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "根据用户任务选择合适的工具。如果需要工具，请调用对应工具。"
                },
                {
                    "role": "user",
                    "content": state["task"]
                }
            ],
            tools=tools_schema,
            tool_choice="auto"
        )

        message = response.choices[0].message

        print("Planner返回:")
        print(message)

        # 没有调用工具
        if not message.tool_calls:
            state["plan"] = []
            state.status = "success"
            return state

        # 保存模型产生的tool_calls
        plan = []

        for tool_call in message.tool_calls:
            plan.append({
                "tool": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments)
            })

        state["plan"] = json.dumps(plan, ensure_ascii=False, indent=4)

        return state