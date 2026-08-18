# AI Agent 工程学习笔记（教学文档）
## 1. 项目目标

这个项目用于从零理解 AI Agent 的工作原理，并逐步学习如何使用 LangChain 和 LangGraph 构建结构清晰、可扩展的 Agent。

学习路线不是直接调用一个封装好的 Agent API，而是先亲手实现关键模块，再将它们逐步迁移到成熟框架中。这样可以同时回答两个问题：

1. Agent 在底层到底做了什么？
2. LangChain 和 LangGraph 分别替我们解决了什么？

当前项目已经经历了两个阶段：

- 第一阶段：手动实现 Skill、Tool Schema、ToolRegistry、Router、Planner、Executor、ResponseGenerator、AgentState、AgentGraph 和 Memory。
- 第二阶段：使用 LangChain Tool 和 LangGraph 构建一个基础 ReAct Agent，支持天气查询、数学计算和景点查询。

当前代码仍处于迁移阶段，仓库中同时保留了旧模块和新的 LangGraph 实现。这些旧模块适合用于理解原理，但并不都在当前执行链中运行。

---

## 2. 先建立 Agent 的核心心智模型

普通 LLM 应用通常是：

```text
用户问题
   ↓
LLM
   ↓
文本回答
```

这种模式只能依赖模型已有的知识生成文本，无法可靠地查询实时天气、访问数据库、执行代码或调用外部系统。

Agent 在 LLM 周围增加了工具、状态和控制流程：

```text
用户目标
   ↓
LLM 理解和决策
   ↓
选择是否调用工具
   ↓
程序执行确定性能力
   ↓
把执行结果反馈给 LLM
   ↓
LLM 继续决策或生成最终回答
```

可以把它概括为：

> Agent 是一个以 LLM 为决策核心，通过 Tool 使用外部能力，并通过 State 和 Workflow 持续推进任务的执行系统。

这里最重要的不是“让 LLM 回答问题”，而是建立一个循环：

```text
Reasoning → Action → Observation → Reasoning
```

- Reasoning：模型理解当前情况并决定下一步。
- Action：模型生成工具调用请求。
- Observation：程序执行工具，把结果返回给模型。
- 再次 Reasoning：模型根据新结果继续行动或结束任务。

这就是 ReAct 的核心思想。

---

## 3. 必须区分的三层职责

### 3.1 业务能力层：真正完成任务

业务能力就是项目中的 Skill，例如：

- 查询天气；
- 计算数学表达式；
- 查询城市景点；
- 获取当前时间；
- 保存或读取用户信息。

Skill 只关心业务输入和业务输出，不应该关心：

- 使用哪个 LLM；
- 是否使用 LangChain；
- 是否使用 LangGraph；
- 当前位于哪个 Graph Node；
- LLM 为什么选择了它。

理想关系是：

```text
输入参数 → Skill → 业务结果
```

当前项目中的业务能力主要位于：

```text
skills/
├── weather.py
├── calculator.py
├── attraction.py
└── time_tool.py
```

### 3.2 Agent 基础设施层：连接 LLM 和业务能力

Agent 基础设施负责解决以下问题：

- 如何告诉 LLM 当前有哪些工具？
- 每个工具接收什么参数？
- LLM 如何表达“我要调用这个工具”？
- 程序如何根据名称找到对应函数？
- 工具结果如何反馈给 LLM？
- 对话和工具执行记录如何保存？

这一层包括：

```text
Tool 描述
Tool Calling 协议
Tool Registry
Tool Executor
Agent Node
Message
State
Memory
```

### 3.3 LangGraph 编排层：控制执行顺序

编排层不负责天气查询，也不负责模型推理内容。它负责控制：

- 从哪个节点开始；
- 一个节点执行后去哪里；
- 什么时候执行工具；
- 工具完成后是否回到模型；
- 什么时候结束；
- State 如何在节点之间传递和合并。

核心概念是：

```text
State：节点共享的数据
Node：读取 State、执行动作、返回状态更新
Edge：连接两个节点
Conditional Edge：根据 State 决定下一条路径
Graph：描述完整工作流
```

---

## 4. 第一阶段：先把 Skill 写出来

学习 Agent 的第一步不是调用 LLM，而是先准备可以独立运行的确定性能力。

例如天气 Skill 的思路：

```text
接收 city
   ↓
校验是否支持该城市
   ↓
查询或读取天气数据
   ↓
返回天气结果
```

这个阶段需要掌握：

- Python 函数；
- 参数和返回值；
- 输入校验；
- 异常处理；
- 业务逻辑与框架代码分离；
- 如何为函数编写单元测试。

判断 Skill 是否设计合理，可以问：

> 不启动 LLM、不启动 LangGraph，我能不能直接调用并测试这个函数？

如果答案是可以，它通常才是独立的业务能力。

---

## 5. 第二阶段：让 LLM 知道有哪些工具

LLM 无法直接看到 Python 中已经定义的函数。程序必须向模型发送一份工具描述，例如：

```text
工具名称：get_weather
工具作用：查询指定城市的天气
参数：city，字符串，必填
```

早期代码通过 `tools_schema.py` 手工维护 OpenAI Tool Calling JSON Schema。

这个阶段需要理解：

```text
Skill
    = 真正执行任务的 Python 函数

Tool Schema
    = 给 LLM 阅读的接口说明

Tool Call
    = LLM 根据 Tool Schema 产生的结构化调用请求
```

重要结论：

> LLM 不会直接执行 Python 函数。LLM 只会生成“希望调用什么工具、传入什么参数”，真正的函数调用仍然由程序完成。

典型流程：

```text
用户：北京天气怎么样？
   ↓
程序把用户消息和 Tool Schema 一起发送给 LLM
   ↓
LLM 返回：get_weather({"city": "北京"})
   ↓
程序解析调用请求
   ↓
程序执行 Python 函数
```

---

## 6. 第三阶段：实现 ToolRegistry 和 Executor

当 LLM 返回工具名称时，程序需要找到真正的 Python 函数。

### 6.1 ToolRegistry

ToolRegistry 保存名称到函数的映射：

```text
get_weather       → skills.weather.get_weather
calculate         → skills.calculator.calculate
search_attraction → skills.attraction.search_attraction
```

它解决的是：

> LLM 返回的是字符串名称，程序如何找到真正的可调用对象？

### 6.2 Executor

Executor 负责：

1. 读取 Planner 生成的计划；
2. 取得工具名称和参数；
3. 从 ToolRegistry 查找函数；
4. 执行函数；
5. 保存成功结果或异常信息。

执行链如下：

```text
Plan
   ↓
Executor
   ↓
ToolRegistry.get_tool(tool_name)
   ↓
Skill(**arguments)
   ↓
Result
```

这个阶段需要掌握：

- 函数也是 Python 对象；
- 字典映射；
- `**arguments` 参数展开；
- JSON 解析；
- 执行结果和异常的结构化表达；
- 为什么 LLM 决策和确定性代码执行必须分开。

---

## 7. 第四阶段：加入 Router、Planner 和 ResponseGenerator

### 7.1 Router

Router 回答的是：

> 当前任务应该走哪条工作流？

旧实现首先使用 RouterNode 判断：

```text
TOOL：需要调用工具
CHAT：可以直接回答
```

工具执行后，另一个 Router 根据执行状态决定：

```text
success → response
failed 且可以重试 → planner
failed 且达到上限 → response/fallback
```

### 7.2 Planner

Planner 回答的是：

> 为了完成当前任务，具体应该调用什么工具、传入什么参数？

当前项目早期 Planner 的主要能力是解析 LLM 产生的 Tool Calls，并生成 JSON Plan。它更接近“工具选择器”，还不是完整的多步规划系统。

真正的显式多步 Planner 还可能负责：

- 拆分多个子任务；
- 表达步骤依赖；
- 规划并行任务；
- 为每一步设置完成条件；
- 失败后只重新规划受影响的部分；
- 在执行前让用户审批计划。

### 7.3 ResponseGenerator

ResponseGenerator 将用户任务和工具结果重新交给 LLM，生成自然语言答案：

```text
用户问题 + 工具执行结果
             ↓
       ResponseGenerator
             ↓
        最终自然语言回答
```

这个阶段需要理解：

- Router 决定“走哪条路”；
- Planner 决定“具体做什么”；
- Executor 负责“真正执行”；
- ResponseGenerator 负责“把结果说清楚”。

---

## 8. 第五阶段：实现 AgentState 和 AgentGraph

### 8.1 AgentState

State 是 Agent 当前工作现场的结构化表示。

旧 AgentState 曾保存：

```text
task          用户任务
plan          Planner 生成的计划
results       工具执行结果
history       历史消息
status        当前执行状态
next_step     下一节点
retry_count   重试次数
final_answer  最终回答
```

可以这样理解：

```text
Memory = 人长期记住的信息
State  = 人完成当前任务时摆在桌面上的全部材料
```

State 不只是数据容器。Router、Planner、Executor 等节点都通过 State 交换信息，所以它也是节点之间的协议。

### 8.2 AgentGraph

旧 AgentGraph 手工实现了：

- Node 注册；
- Edge 注册；
- Conditional Edge 注册；
- 当前节点推进；
- 节点循环；
- 结束条件。

旧执行链可以概括为：

```text
用户输入
   ↓
AgentState.task
   ↓
RouterNode
   ├── CHAT ──────────────────────────┐
   │                                  │
   └── TOOL → Planner → Executor      │
                         │            │
                         ▼            │
                  ToolRegistry        │
                         │            │
                         ▼            │
                       Skill          │
                         │            │
                         ▼            │
                    router.py         │
                         │            │
                         └──→ ResponseGenerator
                                      │
                                      ▼
                              AgentState.final_answer
```

完成这一阶段后，应该能够回答：

- 为什么需要 State？
- Node 为什么应该返回状态更新？
- 普通 Edge 和 Conditional Edge 有什么区别？
- 如何避免 Agent 无限循环？
- 为什么重试次数必须真正递增，而不能只写判断条件？

---

## 9. 第六阶段：使用 LangChain Tool 替代手工 Tool Schema

LangChain 的 `@tool` 可以根据以下信息生成 Tool Schema：

- 函数名称；
- docstring；
- 参数类型注解；
- Pydantic 参数模型。

概念关系应该是：

```text
Skill
   ↓ 被薄薄地包装
LangChain Tool
   ↓ 暴露名称、描述和参数 Schema
LLM
```

推荐分层：

```text
src/ai_agent_learning/skills/
    真正的业务能力
    不依赖 LangChain 和 LangGraph

src/ai_agent_learning/tools/adapters.py
    使用 @tool 暴露 Skill
    负责 Tool 名称、描述和参数协议
    不重新实现业务逻辑
```

例如正确的思维方式是：

```text
LangChain get_weather Tool
    ↓
调用 weather Skill
    ↓
返回 Skill 的业务结果
```

而不是在 Skill 和 Tool 中分别实现一套天气查询逻辑。

---

## 10. 第七阶段：使用 LangGraph 构建 ReAct Agent

当前主执行链由以下模块组成：

```text
main.py
   ↓
ai_agent_learning.cli
   ├── config.py
   ├── llm.py
   └── agent/graph.py
       ├── agent/state.py
       ├── agent/node.py
       ├── LangGraph ToolNode
       └── tools/adapters.py → skills/
```

### 10.1 当前 State

`agent/state.py` 使用 `TypedDict` 定义 State：

```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

这里需要理解两个知识点：

1. `messages` 保存 HumanMessage、AIMessage 和 ToolMessage。
2. `add_messages` 是 reducer，负责把节点返回的新消息追加到已有消息中。

节点通常只返回增量：

```python
return {"messages": [response]}
```

LangGraph 再根据 reducer 将新消息合并到原 State。

### 10.2 AgentNode

`agent/node.py` 负责：

```text
读取 state["messages"]
   ↓
调用绑定了 Tools 的 LLM
   ↓
返回 AIMessage
```

`bind_tools(tools)` 的作用是把 Tool Schema 暴露给模型，让模型能够产生标准 `tool_calls`。

AgentNode 的职责是决策，不是执行工具。

### 10.3 ToolNode

ToolNode 负责：

```text
读取 AIMessage.tool_calls
   ↓
根据工具名称找到 LangChain Tool
   ↓
校验参数并执行 Tool
   ↓
生成 ToolMessage
```

ToolNode 在职责上相当于旧架构中的：

```text
ToolRegistry + Executor
```

### 10.4 tools_condition

`tools_condition` 检查最后一条 AIMessage：

```text
存在 tool_calls → tools 节点
不存在 tool_calls → END
```

它替代了简单的 TOOL/CHAT 路由，但不能替代复杂业务路由、错误路由或多 Agent 路由。

### 10.5 StateGraph

当前 Graph 结构是：

```text
                         ┌─────────────────────┐
                         │                     │
                         ▼                     │
START → AgentNode → tools_condition            │
                    │             │             │
                    │有调用       │无调用       │
                    ▼             ▼             │
                 ToolNode        END            │
                    │                           │
                    └───────────────────────────┘
```

完整运行过程：

```text
1. `cli.py` 将用户输入包装为 HumanMessage。
2. StateGraph 从 AgentNode 开始执行。
3. AgentNode 调用绑定 Tools 的 LLM。
4. LLM 直接回答，或者返回 tool_calls。
5. tools_condition 决定去 ToolNode 还是结束。
6. ToolNode 执行 Tool，并返回 ToolMessage。
7. add_messages 将 ToolMessage 加入 State。
8. Graph 再次进入 AgentNode。
9. LLM 读取工具结果，继续调用工具或生成最终回答。
10. `cli.py` 打印最后一条 AIMessage 的内容。
```

---

## 11. 旧模块与当前框架的职责映射

| 旧模块 | 当前对应组件 | 说明 |
| --- | --- | --- |
| Skill | Skill 仍然应该保留 | 框架不会替代真正的业务能力 |
| `tools_schema.py` | LangChain `@tool` | 自动生成 Tool Schema |
| `ToolRegistry` | `ToolNode` 内部工具映射 | 根据名称找到 Tool |
| `Executor` | `ToolNode` | 执行 Tool 并生成 ToolMessage |
| `RouterNode` | LLM 决策 + `tools_condition` | 决定调用工具还是结束 |
| `Planner` | `AgentNode + bind_tools` | 当前是隐式、逐步的工具决策，不是显式多步计划 |
| `ResponseGenerator` | 工具执行后再次调用 AgentNode | 同一个 LLM 节点生成最终回答 |
| 自研 `AgentState` | TypedDict State + reducer | 当前只保留 messages |
| 自研 `AgentGraph` | LangGraph `StateGraph` | 节点、边、条件分支和循环 |
| `Memory` | `InMemorySaver`（仅短期部分） | 当前只实现按 thread_id 保存图状态；长期 Memory 仍未接入 |

需要特别注意：

> LangChain/LangGraph 替代的是重复的 Agent 基础设施和编排代码，不是业务 Skill，也不会自动提供长期记忆、错误策略和安全控制。

---

## 12. 当前仓库的实际状态

当前真正运行的主链是：

```text
main.py
   ↓
ai_agent_learning.cli
   ↓
Settings → LLM Factory ───────────────┐
InMemorySaver ────────────────────────┤
                                     ↓
                         build_graph → compile(checkpointer)
                                     ↓
用户消息 + thread_id → AgentNode ↔ ToolNode
                                     ↓
                               Tool → Skill
```

旧手写架构已经隔离到：

```text
legacy/hand_built_agent/
```

当前已经完成以下分层：

```text
skills：真正的业务能力实现
   ↓
tools：使用 LangChain @tool 暴露 Skill
   ↓
ToolNode：执行 Tool
   ↓
AgentNode：负责 LLM 决策
   ↓
StateGraph：负责编排流程
```

当前调用链：

```text
用户输入
   ↓
StateGraph
   ↓
AgentNode
   ↓
LLM 产生 tool_calls
   ↓
ToolNode
   ↓
LangChain Tool 适配器
   ↓
Skill
   ↓
业务结果
   ↓
ToolMessage
   ↓
AgentNode
   ↓
最终回答
```

---

## 13. 推荐的代码阅读顺序

不要按照文件名随机阅读，建议按照一次请求真实经过的顺序阅读。

### 第一次阅读：只看主流程

```text
1. main.py
2. src/ai_agent_learning/cli.py
3. src/ai_agent_learning/config.py
4. src/ai_agent_learning/llm.py
5. src/ai_agent_learning/agent/graph.py
6. src/ai_agent_learning/agent/state.py
7. src/ai_agent_learning/agent/node.py
8. src/ai_agent_learning/tools/adapters.py
```

阅读目标：能完整描述用户输入如何变成最终回答。

### 第二次阅读：看业务能力

```text
9. src/ai_agent_learning/skills/weather.py
10. src/ai_agent_learning/skills/calculator.py
11. src/ai_agent_learning/skills/attraction.py
12. src/ai_agent_learning/skills/time.py
```

阅读目标：区分业务函数和 Agent Tool 接口。

### 第三次阅读：回顾自研原理

```text
13. legacy/hand_built_agent/registry.py
14. legacy/hand_built_agent/executor.py
15. legacy/hand_built_agent/router.py
16. legacy/hand_built_agent/graph.py
17. legacy/hand_built_agent/tools_schema.py
18. legacy/hand_built_agent/memory.py
```

阅读目标：理解 LangChain 和 LangGraph 帮我们减少了哪些基础设施代码。

---

## 14. 每个阶段应该能够回答的问题

### Skill 阶段

- 什么是确定性业务能力？
- 为什么 Skill 不应该依赖 LLM？
- Skill 如何独立测试？

### Tool Calling 阶段

- LLM 是否真的执行了 Python 函数？
- Tool Schema 有什么作用？
- 工具名称、描述和参数类型如何影响 LLM 决策？

### Registry 与 Executor 阶段

- 如何把字符串工具名映射到 Python 函数？
- Executor 应该如何处理工具异常？
- 为什么工具执行结果需要结构化？

### State 与 Graph 阶段

- State 和 Memory 有什么不同？
- Node 为什么应该返回状态更新？
- Conditional Edge 如何控制流程？
- 如何防止无限循环？

### LangChain 阶段

- `@tool` 替代了哪些手工代码？
- Skill 和 LangChain Tool 为什么不应该是同一个概念？
- `bind_tools` 做了什么？

### LangGraph 阶段

- AgentNode 和 ToolNode 分别负责什么？
- `tools_condition` 根据什么路由？
- `add_messages` 为什么是 reducer？
- ToolMessage 如何形成 Observation？
- ReAct 循环什么时候结束？

---

## 15. 当前已知问题与后续学习路线

### 15.1 已完成：Skill 与 Tool 分层

目标：

```text
Tool Adapter 不再重新实现业务逻辑
LangChain Tool 只调用 skills 中的函数
AgentNode 和 ToolNode 使用同一份 Tool 集合
```

### 15.2 已完成基础安全计算，后续完善错误契约

当前 Calculator Skill 已使用 AST 白名单解析替代 `eval()`，只允许受控的数学表达式。

后续仍需要学习：

- Pydantic 参数校验；
- 统一工具错误结构；
- 超时、重试和 fallback。

### 15.3 已完成：基于 Checkpointer 的多轮短期会话

当前 CLI 在启动时确定一个稳定的 `thread_id`，在创建应用时使用 `InMemorySaver`，并把它传给 `graph.compile(checkpointer=...)`。循环中的每次 `graph.invoke()` 都携带同一份 `thread_id` 配置，因此 LangGraph 可以在新一轮执行前恢复该线程的历史 State。

需要区分：

```text
State
    当前一次图执行中的共享状态

Short-term Memory
    同一个对话线程中的历史，由当前 InMemorySaver 在进程内保存

Long-term Memory
    跨线程、跨进程保存的用户偏好或事实，需要 Store 或数据库
```

这里的边界很重要：当前 Checkpointer 让同一进程内的连续对话具有上下文，但程序重启后数据会消失，也不会把一个线程中的信息自动共享给另一个线程。

### 15.4 第四优先级：可靠性和循环控制

需要加入：

- 最大工具调用轮数；
- 工具超时；
- 明确的 retry_count；
- 按异常类型重试；
- 模型 fallback；
- 高风险操作人工审批；
- 无法完成时的安全终止回答。

### 15.5 第五优先级：测试和可观测性

需要建立：

```text
Skill 单元测试
Tool 适配测试
ToolNode 测试
Graph 集成测试
使用 Fake LLM 的离线测试
Agent 评估数据集
结构化日志和 Trace
```

评估 Agent 时不能只看“最后答案像不像”，还应该检查：

- 是否选择了正确工具；
- 参数是否正确；
- 是否调用了不必要的工具；
- 工具失败后是否正确处理；
- 是否在限制轮数内完成；
- 最终答案是否忠实使用工具结果。

### 15.6 已完成基础工程结构

当前项目已经整理为：

```text
src/ai_agent_learning/
├── agent/
├── tools/
├── skills/
├── config.py
├── llm.py
├── logging_config.py
└── cli.py

tests/
├── unit/
└── integration/
```

后续可以逐步补充：

- lint、format 和 type check；
- CI；
- API 或服务入口；
- 流式输出；
- 日志、指标和链路追踪。

---

## 16. 最终知识框架

可以用下面这组关系检验自己是否真正理解了当前项目：

```text
Skill
    真正完成业务任务

LangChain Tool
    把 Skill 描述成 LLM 可选择的标准接口

Tool Calling
    让 LLM 产生结构化的工具调用意图和参数

AgentNode
    调用 LLM，负责理解、推理和决策

ToolNode
    根据 tool_calls 执行 Tool，并返回 ToolMessage

State
    保存节点之间共享的动态上下文

Reducer
    定义节点返回的数据如何合并到 State

Conditional Edge
    根据 State 或消息决定下一条路径

StateGraph
    编排 Node、Edge、循环和结束条件

Checkpointer
    按 thread_id 保存和恢复图执行产生的 State 快照

Short-term Memory
    当前线程中可被后续轮次读取的历史消息和运行状态

Long-term Memory
    跨线程或跨进程长期保存的用户事实、偏好和知识

ReAct
    Reasoning → Action → Observation 的循环执行方式
```

最后，用一句话描述当前目标架构：

> CLI 用稳定的 thread_id 调用已配置 Checkpointer 的 LangGraph；图先恢复当前线程的 State，再由 AgentNode 根据消息和 Tool 描述进行决策；ToolNode 执行由 LangChain Tool 暴露的 Skill；结果以 ToolMessage 合并回 State；LangGraph 持续编排 ReAct 循环并保存新的 Checkpoint，直到模型生成最终回答。

---

## 17. LangGraph Checkpoint 学习流程

### 17.1 先理解它解决的问题

没有 Checkpointer 时，每次调用只看到本次传入的消息：

```text
第一轮 invoke({messages: [“我的名字是小明”]}) → 本轮 State → 执行结束
第二轮 invoke({messages: [“我叫什么名字？”]}) → 新的 State → 看不到第一轮
```

接入 Checkpointer 后，LangGraph 会用 `thread_id` 找回上次保存的 State：

```text
thread_id=user_001
    第一轮：载入空状态 → 合并“我的名字是小明” → 执行 → 保存 Checkpoint
    第二轮：恢复第一轮状态 → 合并“我叫什么名字？” → 执行 → 保存新 Checkpoint

thread_id=user_002
    没有 user_001 的 Checkpoint → 从自己的状态开始
```

因此，真正被 Agent “记住”的不是 `thread_id` 字符串本身。`thread_id` 是查找会话状态的键，历史消息实际保存在 Checkpointer 中。

### 17.2 当前代码中每一层的职责

```text
cli.py
    创建 InMemorySaver
    在程序启动时确定 thread_id
    每轮通过 config 把同一个 thread_id 传给 graph.invoke()

agent/graph.py
    接收 Checkpointer
    保留 AgentNode ↔ ToolNode 的 ReAct 图结构
    调用 graph.compile(checkpointer=checkpointer)

agent/state.py
    用 add_messages reducer 定义 messages 的合并规则
    新消息追加到恢复出的历史消息，而不是整体覆盖

AgentNode
    读取合并后的 messages，让 LLM 基于当前线程上下文决策

ToolNode
    按原有方式执行工具并生成 ToolMessage

InMemorySaver
    按 thread_id 隔离、保存和恢复图状态
```

AgentNode 和 ToolNode 不需要自己读写 Checkpoint。状态恢复和保存发生在编译后的 LangGraph 运行时边界，这正是 Checkpointer 属于“编排基础设施”而不是业务逻辑的原因。

### 17.3 一轮对话的完整调用链

```text
程序启动
  → create_agent_app()
  → 创建 LLM
  → 创建 InMemorySaver
  → build_graph(..., checkpointer)
  → graph.compile(checkpointer=checkpointer)

进入 CLI
  → 用户输入或使用默认 thread_id（循环期间保持不变）
  → 用户输入一条新消息
  → graph.invoke(
        {messages: [HumanMessage]},
        config={configurable: {thread_id: ...}}
    )
  → Checkpointer 按 thread_id 恢复历史 AgentState
  → add_messages 把新 HumanMessage 合并进 messages
  → AgentNode 调用 LLM
  → 若有 tool_calls：ToolNode → LangChain Tool → Skill
  → ToolMessage 合并回 messages → 再次进入 AgentNode
  → 输出最终 AIMessage
  → Checkpointer 保存该 thread_id 的最新 State
  → CLI 打印回答并等待下一轮
```

### 17.4 学习时应能回答的判断题

- `messages` 字段存在，不代表天然具有跨请求记忆；还需要 Checkpointer 和稳定的 `thread_id`。
- 每轮生成新的 `thread_id`，相当于每轮开启新会话，无法恢复上一轮状态。
- 两个用户误用同一个 `thread_id`，会共享同一份会话状态，因此生产系统必须正确生成和鉴权会话 ID。
- `add_messages` 决定新旧消息怎样合并；没有正确 reducer 时，新输入可能覆盖历史列表。
- `InMemorySaver` 只适合当前教学阶段和单进程测试；进程退出后不会保留数据。
- Checkpoint 保存图状态，长期 Memory 则解决跨会话的事实、偏好与知识沉淀，两者不能混为一谈。
