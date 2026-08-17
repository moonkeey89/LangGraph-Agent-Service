"""Attraction lookup business capability."""


ATTRACTIONS = {
    "北京": [
        "故宫",
        "天安门",
        "颐和园",
        "八达岭长城",
    ]
}


def search_attraction(city: str) -> list[str] | str:
    """Return attractions for a city when available."""

    return ATTRACTIONS.get(city, "没有找到相关景点")
