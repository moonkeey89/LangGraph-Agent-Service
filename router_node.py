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
                                
                                你的任务是判断用户当前的问题应该：
                                1. 调用工具 TOOL
                                2. 普通回答 CHAT
                                
                                当前Agent拥有以下工具：
                                
                                1. get_weather
                                作用：
                                查询城市天气。
                                
                                例如：
                                北京天气如何
                                上海今天冷吗
                                
                                2. calculate
                                作用：
                                数学计算。
                                
                                例如：
                                1000*999
                                25乘以4
                                
                                3. search_attraction
                                作用：
                                查询城市旅游景点。
                                
                                例如：
                                北京有哪些景点
                                
                                4. save_memory
                                作用：
                                保存用户希望长期记住的信息。
                                
                                例如：
                                记住我是小米
                                记住我喜欢苹果
                                我的爱好是编程，请记住
                                以后叫我小明
                                
                                5. get_memory
                                作用：
                                查询之前已经保存的用户信息。
                                
                                例如：
                                我是谁
                                你知道我叫什么吗
                                我的爱好是什么
                                你记得我喜欢什么吗
                                
                                如果用户的问题需要调用以上任意工具：
                                输出：
                                TOOL
                                
                                如果只是普通聊天、闲聊、知识问答：
                                输出：
                                CHAT
                                
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