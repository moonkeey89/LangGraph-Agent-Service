class AgentNode:


    def __init__(self, llm, tools):

        self.llm = llm.bind_tools(tools)



    def run(self,state):

        messages=state["messages"]


        response=self.llm.invoke(
            messages
        )

        return {
            "messages": [response]
        }
