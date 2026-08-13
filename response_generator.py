class ResponseGenerator:


    def __init__(self, client):

        self.client = client

    def generate(self, state):
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": """
    你是一个智能助手。

    根据用户任务和工具执行结果，
    生成最终回答。

    要求：
    - 回答自然
    - 结合工具结果
    - 不要提及工具名称
    - 不要输出JSON
    """
                },
                {
                    "role": "user",
                    "content": f"""
    用户任务：

    {state.task}

    工具执行结果：

    {state.results}

    请生成最终回答。
    """
                }
            ]
        )
        state.final_answer = response.choices[0].message.content

        return state
        # return response.choices[0].message.content