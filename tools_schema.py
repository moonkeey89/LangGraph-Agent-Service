tools_schema = [

    {
        "type": "function",

        "function": {

            "name": "get_weather",

            "description":
            "查询指定城市的天气信息",

            "parameters": {

                "type": "object",

                "properties": {

                    "city": {

                        "type": "string",

                        "description":
                        "城市名称，例如北京、上海"

                    }

                },

                "required":[
                    "city"
                ]

            }
        }
    },


    {
        "type":"function",

        "function":{

            "name":"calculate",

            "description":
            "执行数学表达式计算",

            "parameters":{

                "type":"object",

                "properties":{

                    "expression":{

                        "type":"string",

                        "description":
                        "数学表达式，例如100*121"

                    }

                },

                "required":[
                    "expression"
                ]

            }
        }
    },


    {
        "type":"function",

        "function":{

            "name":"search_attraction",

            "description":
            "查询指定城市旅游景点",

            "parameters":{

                "type":"object",

                "properties":{

                    "city":{

                        "type":"string",

                        "description":
                        "城市名称"

                    }

                },

                "required":[
                    "city"
                ]

            }
        }
    }

]