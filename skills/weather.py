"""
天气查询Skill
"""


# 支持查询的城市

SUPPORTED_CITIES = {

    "北京",
    "上海",
    "广州"

}


def get_weather(city):
    """
    查询天气

    参数:
        city:
        城市名称

    """


    # 第一步：
    # 参数合法性检查

    if city not in SUPPORTED_CITIES:

        return (
            f"暂不支持查询'{city}'的天气，"
            "目前支持北京、上海、广州"
        )


    # 第二步：
    # 正常执行


    weather_data = {

        "北京":
        "晴天，25℃",

        "上海":
        "小雨，22℃",

        "广州":
        "多云，28℃"

    }

    # raise Exception("接口错误")
    return weather_data[city]