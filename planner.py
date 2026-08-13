from openai import OpenAI


class Planner:

    def __init__(self, client):

        self.client = client

    def create_plan(self, state):
        state.retry_count += 1
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": """
                你是一个Agent任务规划器。
            
                你的目标不是直接回答用户问题，
                而是生成可以被程序执行的工具调用计划。
            
                目前你拥有以下工具：
            
                1. get_weather
                功能：查询指定城市天气
                参数：
                {
                    "city": "城市名称"
                }
            
                2. search_attraction
                功能：查询指定城市旅游景点
                参数：
                {
                    "city": "城市名称"
                }
            
                3. calculate
                功能：数学计算
                参数：
                {
                    "expression": "数学表达式"
                }
            
                你的输出必须严格为JSON数组。
            
                例如：
                [
                    {
                        "tool": "get_weather",
                        "arguments": {
                            "city": "北京"
                        }
                    }
                ]
            
                不要输出解释文字。
                """
                },
                {
                    "role": "user",
                    "content": state.task
                }
            ]
        )

        # 将LLM生成的计划保存到State
        state.plan = response.choices[0].message.content
        state.status = "planning"

        return state