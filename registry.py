class ToolRegistry:


    def __init__(self):

        # 保存所有工具
        self.tools = {}



    # 注册工具
    def register(self, name, function):

        self.tools[name] = function



    # 获取工具
    def get_tool(self, name):

        return self.tools.get(name)