from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()



class AgentNode:


    def __init__(self,tools):

        self.llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )


        self.llm=self.llm.bind_tools(tools)



    def run(self,state):

        messages=state["messages"]


        response=self.llm.invoke(
            messages
        )

        return {
            "messages": [response]
        }