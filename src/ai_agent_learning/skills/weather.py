"""Weather lookup business capability."""


SUPPORTED_CITIES = {"北京", "上海", "广州"}

WEATHER_DATA = {
    "北京": "晴天，25℃",
    "上海": "小雨，22℃",
    "广州": "多云，28℃",
}


def get_weather(city: str) -> str:
    """Return weather data for a supported city."""

    if city not in SUPPORTED_CITIES:
        return (
            f"暂不支持查询'{city}'的天气，"
            "目前支持北京、上海、广州"
        )

    return WEATHER_DATA[city]
