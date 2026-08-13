class AgentGraph:

    def __init__(self):

        self.nodes = {}

        self.edges = {}
        self.conditional_edges = {}


    # 添加节点
    def add_node(self,name,function):

        self.nodes[name] = function


    # 添加边
    def add_edge(self,start,end):

        self.edges[start]=end

    def add_conditional_edge(
            self,
            start,
            router
    ):
        self.conditional_edges[start] = router

    # 运行Graph
    def run(self,state):

        current="router"


        while current:


            print("\n当前Node:")
            print(current)


            node=self.nodes[current]


            # 执行节点
            state=node(state)

            if current in self.conditional_edges:

                router = self.conditional_edges[current]

                current = router(state)

            else:

                current = self.edges.get(current)


        return state