# AI Agent Learning

这是一个用于学习 AI Agent 工程开发的项目。当前实现使用 LangChain Tool 和 LangGraph 构建基础 ReAct Agent，并保留早期手写架构作为学习资料。

## 架构边界

```text
.env → Settings → LLM Factory → AgentNode
                                  ↓
用户 + thread_id → LangGraph → AgentNode → ToolNode → LangChain Tool → Skill
                         ↑      ↑            │
                         │      └─ ToolMessage┘
                         └─ InMemorySaver（按 thread_id 恢复/保存 State）
```

- `skills`：真正的业务能力，不依赖 LangChain 或 LangGraph。
- `tools`：使用 `@tool` 将 Skill 暴露给 LLM，不重复业务逻辑。
- `AgentNode`：调用绑定 Tools 的 LLM，负责决策。
- `ToolNode`：执行 LLM 生成的 Tool Calls。
- `StateGraph`：编排 AgentNode、ToolNode、条件路由和 ReAct 循环。
- `InMemorySaver`：按 `thread_id` 保存进程内 Checkpoint，为连续会话恢复短期状态。
- `legacy`：早期手写 Agent 代码，不进入当前运行链。

## 环境要求

- Python 3.11 或更高版本
- DeepSeek API Key

## 安装

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e .
```

复制环境变量示例并填写真实 Key：

```powershell
Copy-Item .env.example .env
```

```dotenv
DEEPSEEK_API_KEY=your-real-key
```


## 运行

```powershell
.venv\Scripts\python main.py
```

也可以在安装后使用：

```powershell
.venv\Scripts\python -m ai_agent_learning
```

程序启动后先输入会话 ID；直接回车会使用 `default`。一次进程运行期间，同一会话 ID 会恢复之前的消息，不同会话 ID 的状态相互隔离。输入 `exit` 或 `quit` 退出。

会话过程中可以输入以下调试命令，它们只读取 Checkpoint，不会发送给 LLM：

```text
/state      查看当前会话的最新 StateSnapshot
/history    从新到旧查看当前会话的 Checkpoint 历史
```

当前使用进程内 Checkpointer，程序退出后状态会丢失；它不是数据库或长期记忆。

## 测试

```powershell
.venv\Scripts\python -m unittest discover -s tests -t . -v
```

默认测试使用 Fake/Mock LLM，不会请求 DeepSeek API，并覆盖同会话恢复、不同会话隔离及原有工具调用循环。

## 项目结构

```text
src/ai_agent_learning/
├── cli.py              # CLI 入口与最外层错误边界
├── config.py           # 配置加载和校验
├── llm.py              # DeepSeek LLM 创建
├── logging_config.py   # 标准日志初始化
├── agent/              # State、AgentNode、Graph Builder 和 Checkpoint 调试
├── tools/              # LangChain Tool 适配层
└── skills/             # 业务能力

tests/
├── unit/               # 单模块测试
└── integration/        # ReAct Graph 集成测试

legacy/                 # 旧手写 Agent 学习代码
docs/                   # 学习文档
```

更完整的原理和学习路径参见 [docs/comprehension.md](docs/comprehension.md)。

## 当前范围

当前项目覆盖基础 LangGraph ReAct Agent 工程结构和进程内短期会话记忆，暂未实现：

- 持久化 Checkpointer 与长期 Memory
- RAG
- FastAPI
- Multi-Agent
- 部署与容器化
