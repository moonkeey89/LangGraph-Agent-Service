class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, name, function):
        self.tools[name] = function

    def get_tool(self, name):
        return self.tools.get(name)
