# AI Agent Learning

这是一个用于学习 AI Agent 工程开发的项目。当前实现使用 LangChain Tool 和 LangGraph 构建基础 ReAct Agent，并保留早期手写架构作为学习资料。

## 架构边界

```text
.env → Settings → LLM Factory → AgentNode
                                  ↓
用户 → LangGraph → AgentNode → ToolNode → LangChain Tool → Skill
                         ↑            │
                         └─ ToolMessage┘
```

- `skills`：真正的业务能力，不依赖 LangChain 或 LangGraph。
- `tools`：使用 `@tool` 将 Skill 暴露给 LLM，不重复业务逻辑。
- `AgentNode`：调用绑定 Tools 的 LLM，负责决策。
- `ToolNode`：执行 LLM 生成的 Tool Calls。
- `StateGraph`：编排 AgentNode、ToolNode、条件路由和 ReAct 循环。
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

`.env` 已被 Git 忽略，不要提交真实密钥。

## 运行

```powershell
.venv\Scripts\python main.py
```

也可以在安装后使用：

```powershell
.venv\Scripts\python -m ai_agent_learning
```

输入 `exit` 或 `quit` 退出。

## 测试

```powershell
.venv\Scripts\python -m unittest discover -s tests -t . -v
```

默认测试使用 Fake/Mock LLM，不会请求 DeepSeek API。

## 项目结构

```text
src/ai_agent_learning/
├── cli.py              # CLI 入口与最外层错误边界
├── config.py           # 配置加载和校验
├── llm.py              # DeepSeek LLM 创建
├── logging_config.py   # 标准日志初始化
├── agent/              # State、AgentNode 和 Graph Builder
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

当前项目只覆盖基础 LangGraph ReAct Agent 工程结构，暂未实现：

- Memory/checkpointer
- RAG
- FastAPI
- Multi-Agent
- 部署与容器化
