from langchain_core.tools import tool


@tool
def get_weather(city:str):

    """
    查询城市天气
    """

    return "晴天，25℃"



@tool
def calculate(expression:str):

    """
    数学计算
    """

    return str(eval(expression))



@tool
def search_attraction(city:str):

    """
    查询城市景点
    """

    return [
        "故宫",
        "天安门",
        "颐和园"
    ]