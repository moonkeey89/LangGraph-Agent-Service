class AgentState:

    def __init__(self, task=""):

        # 用户任务
        self.task = task

        # Planner生成的计划
        self.plan = []

        # 工具执行结果
        self.results = []

        # 历史消息
        self.history = []

        # 当前Agent状态
        self.status = "start"

        # 下一步动作
        self.next_step = ""

        self.final_answer = ""