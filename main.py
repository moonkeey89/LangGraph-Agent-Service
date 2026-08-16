from react_graph import app

from langchain_core.messages import HumanMessage



while True:


    user=input("请输入:")


    result=app.invoke(
        {
            "messages":[
                HumanMessage(
                    content=user
                )
            ]
        }
    )


    print(
        result["messages"][-1].content
    )