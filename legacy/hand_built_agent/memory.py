class Memory:
    def __init__(self):
        self.memories = {}

    def save(self, key, value):
        self.memories[key] = value

    def get(self, key):
        return self.memories.get(key)

    def get_all(self):
        return self.memories

    def save_memory(self, key, value):
        self.save(key, value)
        return f"已经记住：{key} = {value}"

    def get_memory(self, key):
        value = self.get(key)

        if value is None:
            return f"没有找到关于 {key} 的记忆"

        return f"{key} = {value}"
