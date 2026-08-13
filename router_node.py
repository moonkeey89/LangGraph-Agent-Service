class RouterNode:


    def __init__(self,client):

        self.client=client



    def decide(self,state):


        response=self.client.chat.completions.create(

            model="deepseek-chat",


            messages=[

                {
                    "role":"system",

                    "content":
                            """
                            你是一个Agent任务路由器。
                            
                            你的任务是判断用户问题是否需要调用外部工具。
                            
                            
                            当前Agent拥有以下工具：
                            
                            1. get_weather
                            作用：
                            查询城市天气。
                            
                            例如：
                            北京天气如何
                            上海今天冷吗
                            广州有没有下雨
                            
                            
                            2. calculate
                            作用：
                            数学计算。
                            
                            例如：
                            1000*999
                            
                            
                            3. search_attraction
                            作用：
                            查询城市景点。
                            
                            例如：
                            北京有哪些景点
                            
                            
                            如果用户的问题属于以上工具能力范围：
                            输出 TOOL
                            
                            
                            如果只是普通聊天、闲聊、自我介绍、知识问答：
                            输出 CHAT
                            
                            
                            严格要求：
                            只能输出 TOOL 或 CHAT。
                            禁止输出解释。
                            """
                },


                {
                    "role":"user",

                    "content":state.task
                }

            ]

        )


        result=response.choices[0].message.content.strip()

        print("Router原始输出:")
        print(result)
        if result=="TOOL":

            state.next_step="planner"


        else:

            state.next_step="response"



        return state