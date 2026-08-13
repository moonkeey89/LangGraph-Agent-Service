def search_attraction(city):

    attractions = {

        "北京":
        [
            "故宫",
            "天安门",
            "颐和园",
            "八达岭长城"
        ]

    }


    return attractions.get(
        city,
        "没有找到相关景点"
    )