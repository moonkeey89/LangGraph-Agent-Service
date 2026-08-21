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
| `Memory` | `SqliteSaver`（仅短期部分） | 当前按 thread_id 持久化图状态；长期 Memory 仍未接入 |

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
SqliteSaver ──────────────────────────┤
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

当前 CLI 在启动时确定一个稳定的 `thread_id`，打开 `data/checkpoints.sqlite` 中的 `SqliteSaver`，并把它传给 `graph.compile(checkpointer=...)`。循环中的每次 `graph.invoke()` 都携带同一份 `thread_id` 配置，因此 LangGraph 可以在新一轮执行前恢复该线程的历史 State。

需要区分：

```text
State
    当前一次图执行中的共享状态

Short-term Memory
    同一个对话线程中的历史，由当前 SqliteSaver 持久化保存

Long-term Memory
    跨线程、跨进程保存的用户偏好或事实，需要 Store 或数据库
```

这里的边界很重要：当前 Checkpointer 允许程序重启后继续恢复同一个线程，但不会把一个线程中的信息自动共享给另一个线程；跨线程用户偏好仍属于长期 Memory 的范围。

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
    在 SQLite 连接有效期内编译 Graph 并运行 CLI
    在程序启动时确定 thread_id
    每轮通过 config 把同一个 thread_id 传给 graph.invoke()

checkpoint.py
    创建 data 目录但不删除已有数据库
    打开 data/checkpoints.sqlite
    创建 SqliteSaver 并初始化表
    在上下文退出时关闭 SQLite 连接

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

SqliteSaver
    按 thread_id 隔离、持久化和恢复图状态
```

AgentNode 和 ToolNode 不需要自己读写 Checkpoint。状态恢复和保存发生在编译后的 LangGraph 运行时边界，这正是 Checkpointer 属于“编排基础设施”而不是业务逻辑的原因。

### 17.3 一轮对话的完整调用链

```text
程序启动
  → open_sqlite_checkpointer()
  → 打开 data/checkpoints.sqlite
  → create_agent_app(settings, checkpointer)
  → 创建 LLM
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
  → CLI 退出后关闭 SQLite 连接
```

### 17.4 学习时应能回答的判断题

- `messages` 字段存在，不代表天然具有跨请求记忆；还需要 Checkpointer 和稳定的 `thread_id`。
- 每轮生成新的 `thread_id`，相当于每轮开启新会话，无法恢复上一轮状态。
- 两个用户误用同一个 `thread_id`，会共享同一份会话状态，因此生产系统必须正确生成和鉴权会话 ID。
- `add_messages` 决定新旧消息怎样合并；没有正确 reducer 时，新输入可能覆盖历史列表。
- `SqliteSaver` 适合本地开发和轻量教学项目；同一数据库文件可以跨进程恢复状态。
- Checkpoint 保存图状态，长期 Memory 则解决跨会话的事实、偏好与知识沉淀，两者不能混为一谈。

### 17.5 查看当前 StateSnapshot

当前项目把只读调试功能放在 `agent/checkpoint_debug.py`，不会让 AgentNode 或 ToolNode 负责观察 Checkpoint：

```text
show_current_state(graph, thread_id)
    → graph.get_state(config)
    → 输出最新 StateSnapshot 的主要字段

show_state_history(graph, thread_id)
    → graph.get_state_history(config)
    → 从新到旧输出历史快照摘要
```

`StateSnapshot` 表示图在某个执行步骤开始时的状态快照，当前使用的字段含义如下：

| 字段 | 含义 |
| --- | --- |
| `values` | 当前各 State channel 的值；本项目主要是完整的 `messages` |
| `next` | 从该快照继续执行时的下一个节点；空元组表示图已结束 |
| `config` | 定位当前快照的配置，通常包含 `thread_id`、`checkpoint_ns` 和 `checkpoint_id` |
| `metadata` | 快照来源、执行步数、节点写入以及父图信息等运行元数据 |
| `created_at` | Checkpoint 创建时间 |
| `parent_config` | 上一个 Checkpoint 的定位配置；可用来理解快照之间的父子关系 |

历史摘要额外提取了：

```text
Checkpoint 序号
创建时间
messages 数量
最后一条消息的类型与内容
next 指向的执行节点
checkpoint_id
```

CLI 中可以直接输入：

```text
/state
    查看当前 thread_id 的最新完整状态

/history
    查看当前 thread_id 的历史快照摘要
```

这两个命令会在 CLI 边界被识别，不会作为 HumanMessage 发送给 LLM，也不会创建新的对话消息。

### 17.6 从历史快照观察一次工具调用

例如输入“计算 6 * 7”后，`/history` 会从新到旧显示快照。按实际执行时间从旧到新理解，可以看到：

```text
HumanMessage("计算 6 * 7")
    next = agent
        ↓
AIMessage(tool_calls=[calculate])
    next = tools
        ↓
ToolMessage("42")
    next = agent
        ↓
AIMessage("计算结果是 42")
    next = END
```

这里的 `next` 特别适合学习图的控制流：消息说明“状态里已经有什么”，`next` 说明“接下来准备执行谁”。因此历史快照不仅是聊天记录，也是一条可用于调试的 Graph 执行轨迹。

---

## 18. SQLite Checkpoint 持久化

### 18.1 为什么选择同步 SqliteSaver

当前项目调用的是同步 API：

```text
graph.invoke()
graph.get_state()
graph.get_state_history()
```

因此使用同步的 `langgraph.checkpoint.sqlite.SqliteSaver`。`AsyncSqliteSaver` 对应的是 `ainvoke()`、`aget_state()` 等异步调用；为了更换存储而改变整个调用模型没有必要。

### 18.2 数据库与连接生命周期

```text
main()
  → open_sqlite_checkpointer()
      → 创建 data/（如果不存在）
      → 打开 data/checkpoints.sqlite（不删除、不覆盖）
      → SqliteSaver.setup() 创建或迁移所需表
  → create_agent_app(settings, checkpointer)
  → build_graph(...)
  → graph.compile(checkpointer=checkpointer)
  → prompt_thread_id()
  → run_cli()
      → invoke / get_state / get_state_history
  → 退出 with 作用域
  → 关闭 SQLite 连接
```

Graph 持有 Checkpointer，所以 Graph 的编译、执行和状态查看都必须发生在 SQLite 连接仍然有效的 `with` 作用域内。不能在 `with` 内创建 Graph 后，把它返回到作用域外继续使用。

### 18.3 保存与恢复不需要业务 SQL

每次执行仍然只传：

```python
config = {"configurable": {"thread_id": thread_id}}
graph.invoke({"messages": [new_message]}, config=config)
```

LangGraph 运行时会调用 `SqliteSaver` 的 Checkpointer 接口保存每一步 State；下一次启动时，它用相同 `thread_id` 从同一个 SQLite 文件恢复最新 Checkpoint。项目代码不需要为 `messages` 编写 `INSERT` 或 `SELECT`，否则会重复实现框架已经负责的序列化、版本和父子快照管理。

### 18.4 它仍然是短期会话记忆

SQLite 让短期记忆具有了持久性，但没有改变它的作用域：

```text
相同 thread_id
    → 恢复同一段会话

不同 thread_id
    → 状态隔离

长期用户偏好
    → 由独立 SqliteStore 按 user_id 保存
```

数据库文件包含用户对话，应当视为本地敏感数据。项目通过 `.gitignore` 忽略 `data/*.sqlite` 及其辅助文件，只提交 `data/.gitkeep`，不把真实会话状态推送到 Git。

---

## 19. interrupt 与人工审批

### 19.1 为什么需要 save_memory

改造前的 Tool 清单只有：

```text
get_weather
calculate
search_attraction
```

它们都是查询或计算能力，不适合演示“批准之后才发生写入”。项目先用进程内列表教学 HITL，当前阶段已经把该列表替换为持久化 `SqliteStore`。`save_memory` 现在会在明确意图检查、敏感信息检查和人工批准之后，按 `user_id` 保存真正的长期记忆。

### 19.2 interrupt 在哪里

审批发生在 `tools/adapters.py` 的 `save_memory` LangChain Tool 中：

```python
decision = interrupt(
    {
        "action": "save_user_memory",
        "tool_name": "save_memory",
        "arguments": {
            "content": explicit_content,
            "memory_type": memory_type,
        },
        "message": "该操作将写入一条长期记忆，是否批准？",
    }
)

if decision.get("approved") is not True:
    return "保存操作已取消"

return save_memory_skill(
    runtime.store,
    user_id=runtime.context.user_id,
    memory_id=runtime.tool_call_id,
    content=explicit_content,
    memory_type=memory_type,
    source_thread_id=thread_id,
)
```

`interrupt()` 的 payload 只包含字符串、布尔值和字典，因此可以被 JSON 序列化。真正产生副作用的 `save_memory_skill()` 位于 interrupt 之后。

### 19.3 暂停时发生了什么

```text
HumanMessage("请记住，我喜欢Python")
  → AgentNode 生成 save_memory tool_call
  → tools_condition 路由到 ToolNode
  → ToolNode 调用 save_memory Tool
  → interrupt(JSON payload)
  → Graph 暂停
  → SqliteSaver 保存：
      当前 messages
      待执行节点 tools
      ToolNode 任务
      Interrupt payload 和 interrupt id
      Checkpoint 父子关系
```

此时没有调用 Skill，所以长期记忆 Store 中仍然没有新记录。`graph.get_state(config)` 的 `snapshot.next` 是 `("tools",)`，`snapshot.interrupts` 中包含待审批信息。

### 19.4 CLI 如何恢复

当前项目保持同步 `invoke()`：

```text
首次执行：
graph.invoke({messages: [HumanMessage]}, config)
    → 返回 __interrupt__

批准：
graph.invoke(Command(resume={"approved": True}), config, context=context)

拒绝：
graph.invoke(
    Command(resume={"approved": False, "reason": "用户拒绝"}),
    config,
    context=context,
)
```

CLI 会循环处理 interrupt，直到 Graph 完成或用户输入 `exit`。如果用户在审批时退出，下一次使用相同 `user_id` 和 `thread_id` 启动时，CLI 通过 `graph.get_state(config).interrupts` 找到 SQLite 中的待审批任务，再带同一个 Runtime Context 发送 Command 恢复。

批准或拒绝都不是一条新的对话内容，所以不能包装为 HumanMessage。HumanMessage 会让 Graph 从输入边界开始一次新运行，LLM 可能重新规划工具；`Command(resume=...)` 则把数据交还给原来那个 interrupt 调用点。

### 19.5 为什么必须使用相同 thread_id

```text
data/checkpoints.sqlite
├── thread_hitl_003 → tools 节点中待恢复的 interrupt
└── thread_hitl_004 → 独立状态
```

Command 本身不携带“恢复哪个任务”的完整位置。LangGraph 使用调用 config 中的 thread_id 定位 SQLite Checkpoint，再根据其中保存的任务和 interrupt id，把 resume 数据交给正确的 interrupt。换一个 thread_id 就找不到原暂停点。

### 19.6 节点为什么会从开头重新执行

恢复 interrupt 时，LangGraph 会重新执行包含 interrupt 的节点。当前 interrupt 位于 ToolNode 调用的 `save_memory` Tool 中，因此 Tool 调用入口和 interrupt 之前的纯计算会再次运行：

```text
重新进入 ToolNode
  → 再次解析同一个 tool_call
  → 再次构造审批 payload
  → 再次调用 interrupt(payload)
  → interrupt 返回已保存的 resume 数据
  → 根据 approved 决定是否调用 Skill
```

当前 interrupt 之前只有意图、安全检查和字典构造等无副作用操作。真正的 Store 写入在 interrupt 之后，并使用稳定 `tool_call_id` 作为 memory_id，所以普通暂停恢复不会新增重复记录。

但需要理解更严格的工程边界：如果进程恰好在“外部写入已经成功、LangGraph 尚未保存写入后的 Checkpoint”之间崩溃，恢复仍可能再次执行写入。真实支付、发邮件或数据库更新还需要幂等键、唯一约束或业务事务；interrupt 本身不提供分布式 exactly-once 保证。

### 19.7 批准、拒绝和普通工具的路径

```text
普通工具：
AgentNode → ToolNode → calculate/get_weather/... → AgentNode → END

敏感工具批准：
AgentNode → ToolNode → interrupt
                         ↓ Command(approved=True)
                       save_memory Skill → ToolMessage → AgentNode → END

敏感工具拒绝：
AgentNode → ToolNode → interrupt
                         ↓ Command(approved=False)
                       取消 ToolMessage → AgentNode → END
```

Graph 节点和边没有改变。HITL 能力来自敏感 Tool 内的动态 interrupt、SQLite Checkpoint 和 CLI 的 Command 恢复协议。

---

## 20. update_state、Replay 与 Time Travel/Fork

### 20.1 三种“继续执行”不是一回事

| 操作 | 解决的问题 | 当前项目调用方式 |
|---|---|---|
| Resume | 给当前暂停的 `interrupt()` 提供决定 | `invoke(Command(resume=...), current_config)` |
| Replay | 从历史状态的 `next` 重新执行 | `invoke(None, selected_snapshot.config)` |
| Fork | 修改历史状态，创建新 Checkpoint 后继续 | `update_state(...)`，再 `invoke(None, fork_config)` |

Resume 针对的是尚未完成的任务；Replay 不改变选中的历史状态，只从那里重新向前运行；Fork 则先制造一条“如果当时状态不同”的新分支。批准或拒绝仍然只能通过 `Command(resume=...)`，不能借 Replay 代替。

### 20.2 历史列表与序号

`/history` 从新到旧打印：

```text
Checkpoint #1（最新）
  checkpoint_id: ...
  metadata.step: ...
  metadata.source: ...
  创建时间: ...
  最后一条消息: ...
  下一步执行节点: ...
```

这里的 `#1` 只是当前列表中的显示序号，方便人在 CLI 中选择；真正定位历史快照的是 `checkpoint_id`。用户输入序号后，程序取回对应的 `StateSnapshot`，后续始终使用它的完整 `snapshot.config`，不会自己拼装或猜测 checkpoint ID。

`metadata.source` 常见值：

```text
input  → 新输入产生的 Checkpoint
loop   → Graph 循环中的节点执行产生
update → update_state() 创建
fork   → 从历史 Checkpoint 重放时建立分支
```

### 20.3 Replay 执行哪些节点

Replay 使用：

```python
graph.invoke(None, selected_snapshot.config)
```

`None` 表示没有新的 HumanMessage。完整 config 同时包含 `thread_id` 和 `checkpoint_id`：前者定位会话，后者定位该会话中的具体历史状态。如果只传 thread_id，LangGraph 会恢复最新状态，那就不再是从所选历史点重放。

`StateSnapshot` 表示一个 step 开始时的状态，所以执行从 `snapshot.next` 开始：

```text
next = ("tools",)
  → 之前生成 tool_call 的 AgentNode 不重跑
  → ToolNode 重跑
  → 后续 AgentNode 重跑

next = ("agent",)
  → 之前的 ToolNode 不重跑
  → AgentNode 重跑
```

第一版只允许 calculate：选择 `next=("tools",)` 可以观察 ToolNode 和 AgentNode 重跑；选择 calculate ToolMessage 后的 `next=("agent",)` 可以观察只有 AgentNode 重跑。

### 20.4 Fork 修改什么

当前 `AgentState` 只有 `messages`，因此 Fork 不新增教学专用字段，而是修改 calculate 返回的 `ToolMessage`。程序要求选择满足以下条件的快照：

```text
最后一条消息是 name="calculate" 的 ToolMessage
next = ("agent",)
没有 pending interrupt
```

例如把工具结果从 `42` 修改为 `43`。替换消息保留原 `id`、`tool_call_id` 和其他字段，只改变 `content`：

```python
replacement_message = original_message.model_copy(
    update={"content": "43"}
)
fork_config = graph.update_state(
    selected_snapshot.config,
    {"messages": [replacement_message]},
    as_node="tools",
)
result = graph.invoke(None, fork_config)
```

`messages` 使用 `add_messages` reducer。相同消息 ID 表示替换原位置；新 ID 则表示追加。如果错误地创建一个新 ID，就会同时留下 `42` 和 `43` 两条互相冲突的 ToolMessage，而不是修改历史观察值。

这里明确使用 `as_node="tools"`，因为这条人工状态更新在语义上模拟 ToolNode 的输出。LangGraph 因此沿现有的 `tools → agent` 边安排后继节点。如果省略 `as_node`，框架需要猜测更新来自哪个节点，在包含 agent 和 tools 的循环图中不够清晰。

### 20.5 新分支不会覆盖旧历史

`update_state()` 返回一个新的 `RunnableConfig`，其中含有新 Checkpoint 的 `checkpoint_id`。旧 `selected_snapshot.config` 仍指向原来结果为 `42` 的快照，新 `fork_config` 指向结果为 `43` 的更新快照：

```text
历史 Checkpoint（42）
├── 原执行 → 最终回答 42
└── update_state（43，source=update）
      └── invoke(None, fork_config) → 最终回答 43
```

后续必须使用 `fork_config`，否则无法明确告诉 LangGraph 从刚创建的分支继续。原历史和新分支都由 SqliteSaver 保存；关闭程序再打开同一个 `data/checkpoints.sqlite`，它们仍能通过各自 checkpoint ID 被读取。

### 20.6 副作用安全边界

Time Travel 会重新执行节点，所以不能假设历史工具已经执行过就不会再执行。当前安全规则是：

- 只将 `calculate` 放入 Replay 白名单；
- 含 pending interrupt 的快照必须使用 Resume，拒绝 Replay；
- 即将执行 `save_memory` 或无法识别工具的快照直接拒绝；
- Fork 只允许替换 calculate 的 ToolMessage；
- Replay/Fork 后若 Agent 新生成敏感调用，`interrupt()` 仍会暂停，CLI 不自动批准；
- thread_id 必须与快照 config 中的 thread_id 相同，禁止跨线程使用快照。

SQLite 保存 Graph 状态不等于业务操作具有 exactly-once 语义。未来如果工具会支付、发邮件或写业务数据库，还需要幂等键、唯一约束和业务事务；这些不属于本阶段。

---

## 21. 错误分类、有限重试与失败降级

### 21.1 改造前各工具怎样处理错误

| 工具/边界 | 可能的错误 | 原处理方式 |
|---|---|---|
| calculate Skill | 语法、类型、除零、溢出 | Skill 内转换为“无法计算”，不会抛给 Graph |
| weather | 不支持的城市 | 返回业务提示 |
| search_attraction | 找不到城市 | 返回业务提示 |
| LangChain Tool 参数校验 | 缺字段、字段类型错误 | ToolNode 默认只把参数校验错误转换为错误 ToolMessage |
| save_memory | `interrupt()` 暂停；批准后的写入异常 | Graph interrupt 正常向外冒泡；其他异常原来会到 CLI 总异常边界 |
| AgentNode/LLM | 网络、认证、限流等 | 原来由 CLI 记录日志并输出统一失败提示 |

本阶段聚焦“工具执行边界”。LLM 调用错误仍由 CLI 最外层保护，不与工具副作用重试混为一套策略。

### 21.2 为什么要先分类

```text
transient
  TimeoutError、ConnectionError、限流等
  → 可以有限重试

invalid_arguments
  Tool schema、ValueError、TypeError
  → 返回错误 ToolMessage 给 Agent 修改参数

permission
  PermissionError、认证失败
  → 不自动重试

permanent
  明确永久失败或未知异常
  → 不自动重试，生成失败说明

side_effect_unknown
  写操作可能已经成功，但调用方没有收到确定结果
  → 绝不能盲目重试
```

代码虽然在 ToolNode wrapper 使用 `except Exception` 作为工具边界，但没有把所有异常视为 transient：`GraphBubbleUp` 被明确重新抛出，已知异常逐类判断，未知异常保守归入 permanent。CLI 的宽泛异常捕获只是最后日志边界，也不会触发自动重试。

### 21.3 AgentState 新字段

```text
status       当前执行状态
error        可读错误信息
error_type   五类错误之一
failed_node  当前为 tools
retry_count  已失败的自动执行次数
max_retries  当前固定为 3
```

只有 `messages` 继续使用 `add_messages` reducer；这些标量字段由节点用新值覆盖。初始 AgentNode 写入 `retry_count=0`。第一次临时失败写入 1，第二次写入 2，第三次失败写入 3 并进入人工复核。也就是说，教学版的 `max_retries=3` 表示最多自动执行三次，而不是无限重试。

每次 ToolNode 失败都会形成 SQLite Checkpoint，因此进程在第一次失败后退出，重新打开相同 SQLite 并使用相同 thread_id，仍会恢复 `retry_count=1`、错误类别和下一步节点。

### 21.4 为什么仍然使用 ToolNode

当前 LangGraph 的 ToolNode 支持 `wrap_tool_call`。项目没有重新实现参数注入、ToolMessage 绑定和 interrupt 传播，而是在真实 ToolNode 的单次工具调用边界增加 `tool_error_boundary`：

```text
ToolNode
  → tool_error_boundary(request, execute)
      → execute(request)
      ├── 成功：返回 ToolMessage
      ├── GraphBubbleUp：重新抛出，保留 interrupt 语义
      └── 普通异常：分类并用 Command(update=...) 写入 State
```

失败 ToolMessage 使用稳定消息 ID。同一 tool_call 的后续重试会替换上一次错误观察，而不是为一个 tool_call 追加多条互相冲突的 ToolMessage。

### 21.5 条件路由

```text
AgentNode
  → ToolNode
      ├── success
      │     → tool_success
      │     → 清空 error/error_type/failed_node
      │     → retry_count=0
      │     → AgentNode
      │
      ├── retry
      │     → ToolNode（只重跑工具边界）
      │
      ├── agent_correction
      │     → AgentNode（LLM 读取错误 ToolMessage 并修改参数）
      │
      ├── human_review
      │     → interrupt(retry/cancel)
      │
      └── fail
            → failure 节点生成明确说明
            → END
```

正常 ReAct 的核心仍是 AgentNode 决策、ToolNode 执行、结果回到 AgentNode；`tool_success` 只是清理错误状态，`human_review` 和 `failure` 是故障分支。

### 21.6 unstable_tool

`unstable_tool` 对应的 Skill 使用进程内计数器模拟临时故障：同一任务前两次固定抛出 TimeoutError，第三次返回成功。它不访问网络、不写业务数据库，计数器只用于确定性教学测试，并提供 reset 函数。

执行过程：

```text
retry_count=0
  → attempt 1 TimeoutError
  → retry_count=1，Checkpoint，route=retry
  → attempt 2 TimeoutError
  → retry_count=2，Checkpoint，route=retry
  → attempt 3 成功
  → tool_success 清理错误，retry_count=0
  → AgentNode 生成最终回答
```

普通工具重试没有使用 Replay。Replay 会从历史 Checkpoint 重跑其后的所有节点，可能重复 LLM 调用、其他工具或无关步骤；错误恢复只应重跑明确失败且确认无副作用的工具调用边界。

### 21.7 超限后的 interrupt

第三次临时失败后，Graph 路由到 human_review：

```python
interrupt(
    {
        "failed_node": "tools",
        "error": "...",
        "retry_count": 3,
        "max_retries": 3,
        "options": ["retry", "cancel"],
    }
)
```

SQLite 保存当前 State、失败 ToolMessage、human_review 任务和 Interrupt。退出程序后用同一 thread_id 可以继续：

```text
Command(resume={"action": "retry"})
  → route=retry
  → ToolNode 再执行一次

Command(resume={"action": "cancel", "reason": "..."})
  → 添加取消说明
  → END
  → ToolNode 不再执行
```

### 21.8 副作用工具为何不自动重试

`save_memory` 仍然先 interrupt、批准后才写入。工具边界明确把它标记为副作用工具：如果批准后的执行出现超时等“不知道是否已写入”的情况，错误类型会被提升为 `side_effect_unknown`，直接进入 fail，不走 retry。

```text
写入请求已发出
  → 对方可能已成功
  → 本地收到 TimeoutError
  → 盲目重试可能写两次
```

真实工程还要使用幂等键、唯一约束和事务。本阶段只是保证自动恢复层不会替用户做危险的重复执行。

---

## 22. 跨 thread 长期记忆第一阶段

### 22.1 先区分 Checkpoint 和长期记忆

本阶段将早期用于 HITL 教学的进程内模拟列表替换为真正的持久化长期记忆。两套 SQLite 的职责不同：

```text
data/checkpoints.sqlite
  → SqliteSaver
  → key 是 thread_id
  → 保存 messages、执行位置、interrupt、retry_count 和 Checkpoint 分支

data/memories.sqlite
  → SqliteStore
  → namespace 包含 user_id
  → 保存用户明确要求长期记住的简洁事实及向量索引
```

同一 `thread_id` 用来恢复同一段 Graph 会话；同一 `user_id` 的多个不同 thread 则共享长期记忆。更换 thread 后，旧 thread 的 HumanMessage 不会自动进入新 State，但 `search_memory` 仍能访问同一用户的 Memory namespace。

### 22.2 user_id 为什么放在 Runtime Context

如果把 `user_id` 定义成普通 Tool 参数，LLM 就可能生成：

```json
{"query": "编程语言", "user_id": "另一个用户"}
```

当前 Graph 使用：

```python
StateGraph(AgentState, context_schema=AgentContext)
```

应用调用时分别传递：

```python
config = {"configurable": {"thread_id": thread_id}}
context = AgentContext(user_id=user_id)
graph.invoke(input, config=config, context=context)
```

Tool 声明 `runtime: ToolRuntime[...]`，ToolNode 自动注入 Runtime。暴露给 LLM 的 schema 中没有 `runtime`、`store` 或 `user_id`：

```text
save_memory  → content, memory_type
search_memory → query
list_memories → 无参数
delete_memory → memory_id
```

因此 namespace 只能由应用提供的可信 `runtime.context.user_id` 构造，不能由模型选择。

### 22.3 Store namespace 才是隔离边界

每个用户使用独立 namespace：

```python
("ai_agent_learning", "users", user_id, "memories")
```

`search_memory`、`list_memories` 和 `delete_memory` 都先构造这个精确 namespace，再执行查询或更新。`user_id` 虽然也保存在记录中用于调试，但隔离不能只依赖 metadata filter，因为调用者可能忘记过滤；namespace 是更稳定的访问边界。

每条记录包含：

```text
memory_id
content
user_id
memory_type
source=user_explicit/memory_manager
source_thread_id
created_at
updated_at
status=active/deleted
```

删除采用软删除：只在当前用户 namespace 查找目标，将 `status` 改成 `deleted`。搜索和列表固定过滤 `status=active`，所以已删除记忆不会再次返回；另一个用户即使知道 memory_id，也无法访问原用户 namespace。

### 22.4 为什么保存不能只相信 LLM

显式保存路径不依赖自动 Memory Manager。即使模型误调用 `save_memory`，Tool 也会反向查找当前 State 中最近的 HumanMessage，并确定性检查是否存在“请记住”“帮我记住”“记住这件事”等明确表达。

```text
普通问题
  → 即使模型调用 save_memory
  → 策略检查失败
  → 不 interrupt、不写入

明确保存请求
  → 从原 HumanMessage 提取“请记住”之后的事实
  → 不把 LLM 提供的 content 当成用户授权证据
  → 检查长度和敏感凭据
  → interrupt 等待人工批准
  → 批准后 Skill 才执行 store.put()
```

API Key、密码、验证码、访问令牌、私钥等内容在 interrupt 前直接拒绝。这样审批界面也不会鼓励保存明显敏感的凭据。

### 22.5 Model2Vec 语义检索

项目没有假设 DeepSeek 提供 Embedding。聊天模型仍由 `llm.py` 创建，长期记忆使用独立的本地多语言模型：

```text
模型：minishlab/potion-multilingual-128M
实现：Model2Vec 静态多语言 Embedding
维度：256
费用：不需要付费 API Key
下载：第一次真正保存或搜索时下载到 Hugging Face 本机缓存
后续：复用缓存，不随项目 Git 提交
```

`SqliteStore` 只索引记录中的 `content` 字段。搜索顺序是：

```text
runtime.context.user_id
  → 精确 Memory namespace
  → filter status=active
  → query 生成本地向量
  → SQLite sqlite-vec 相似度检索
  → 最多返回 top_k=3
  → ToolMessage 返回 AgentNode
```

测试使用一个确定性小型 Embedding 实现，避免测试依赖网络或真实模型下载；生产路径和测试路径都实现 LangChain `Embeddings` 接口，并由同一个 `SqliteStore` 使用。

### 22.6 四个 Memory Tool 的调用链

保存：

```text
AgentNode → save_memory Tool
  → Runtime 取得 user_id/Store/thread_id/tool_call_id
  → 检查明确意图和敏感信息
  → interrupt
  → Command(resume={"approved": true})
  → memory Skill
  → SqliteStore.put(user namespace, tool_call_id, record)
  → ToolMessage → AgentNode
```

`tool_call_id` 作为 `memory_id`，使同一个审批调用在异常恢复时重复写入同一个 key，而不是产生多条重复记录。

检索和列表：

```text
AgentNode → search_memory/list_memories
  → ToolRuntime 注入 user_id 和 Store
  → Memory Skill 只查询当前用户 namespace
  → 有界结果 → ToolMessage → AgentNode
```

删除：

```text
AgentNode → delete_memory(memory_id)
  → interrupt 审批
  → 当前用户 namespace 内查找
  ├─ 找到：软删除
  └─ 未找到：不修改任何其他用户数据
```

Resume、Replay 和 Fork 在继续执行 Graph 时也会重新传入当前 `AgentContext`。`get_state()` 和 `get_state_history()` 只读取 Checkpoint，不执行 Tool，因此不需要 Memory Runtime。

## 23. Memory Manager 最小闭环

### 23.1 为什么拆成决策和执行两个节点

Memory Manager 位于正常 ReAct 最终回答之后：

```text
AgentNode 最终回答
  → Memory Manager
      → 只读取最新 HumanMessage
      → 只查询当前 user_id 的 Top-K 记忆
      → 结构化输出 ADD/UPDATE/DELETE/NONE
  → Memory Executor
      → 代码校验
      → 通过后调用 Memory Skill
  → END
```

LLM 擅长理解“这是新事实还是对旧事实的修正”，但它不应直接拥有写数据库的权限。因此 Memory Manager 只产生不包含 `user_id` 的 `MemoryDecision`；Memory Executor 才能从可信 `Runtime[AgentContext]` 获取用户身份和 Store。

### 23.2 MemoryDecision

```text
operation          ADD / UPDATE / DELETE / NONE
memory_type        preference / profile / fact / instruction / other
content            简洁、独立的记忆正文
target_memory_id   UPDATE/DELETE 的候选目标
confidence         0 到 1
reason             决策理由
```

模型输入只有最新用户消息、当前用户候选记忆以及决策规则。不会提供 AI 最终回答，也不会提供真实 `user_id`。这样模型不能把自己的回答当成用户事实，也不能构造其他用户的 namespace。

### 23.3 Memory Policy 是最终安全边界

模型输出仍是不可信输入。Executor 在写入前确定性检查：

```text
可信 Runtime user_id
  → confidence >= 配置阈值（默认 0.75）
  → ADD/UPDATE 正文不含敏感凭据
  → ADD 与现有有效记忆不完全重复
  → UPDATE/DELETE target 在本次候选 ID 中
  → target 在当前 user_id namespace 中仍然存在
  → 才调用 save/update/delete Memory Skill
```

任何检查失败都会把执行结果安全降级为 `NONE`，只在 State 中记录拒绝原因，不影响已经生成的 Agent 回答。ADD 使用由 `user_id + thread_id + 最新用户消息` 派生的稳定 key；UPDATE 使用当前用户 namespace 内的原 key；DELETE 继续软删除，因此节点重放不会悄悄产生多条重复记录。

### 23.4 如何控制额外调用和显式记忆重复

代码先做轻量候选判断。天气、计算、寒暄、普通问句直接 `NONE`；“我叫”“我喜欢”“我使用”“我正在”“我现在改用”“以后请”“我的目标”“忘记”等表达才调用结构化决策 LLM。

“请记住”仍保留原有 Tool + interrupt 审批链。Memory Manager 会检查从最新 HumanMessage 开始的本轮 ToolMessage 结果；保存成功、用户拒绝或安全策略拒绝会直接 `NONE`，不会在最终回答后再写一份。

实际模型可能在用户只说“我喜欢……”时也误调用显式 `save_memory`。因此最终实现不能只判断“是否调用过工具”，而要判断 ToolMessage 的结果：

```text
保存成功 / 用户拒绝 / 敏感信息拒绝
  → 已形成最终处理结果，Memory Manager 跳过

仅因“未检测到明确保存意图”而拒绝
  → 显式Tool没有负责这条普通陈述
  → 继续进入自动Memory Manager
```

### 23.5 回答前的 Memory Recall

Memory Manager 解决写入和维护，但它位于最终回答之后，无法帮助当前回答。不同 thread 又不会共享 Checkpoint messages，因此增加独立的回答前召回节点：

```text
新用户消息
  → Memory Recall
      → 识别“我是谁”“我喜欢什么”等个人信息问题
      → Runtime取得可信user_id
      → 当前用户namespace语义检索Top-K
      → 只把候选正文作为不可信背景数据注入AgentNode
  → AgentNode回答
  → ReAct循环
  → Memory Manager / Executor
  → END
```

Recall 不把 `user_id` 或全部记忆交给模型。天气、计算、寒暄等问题直接跳过；Store异常安全降级为空召回，不影响主Agent回答。
