class AgentGraph:

    def __init__(self):

        self.nodes = {}

        self.edges = {}


    # 添加节点
    def add_node(self,name,function):

        self.nodes[name] = function


    # 添加边
    def add_edge(self,start,end):

        self.edges[start]=end


    # 运行Graph
    def run(self,state):

        current="planner"


        while current:


            print("\n当前Node:")
            print(current)


            node=self.nodes[current]


            # 执行节点
            state=node(state)


            # 找下一节点
            current=self.edges.get(current)


        return state