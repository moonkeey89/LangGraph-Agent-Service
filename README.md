# AI Agent Learning

这是一个用于学习 AI Agent 工程开发的项目。当前实现使用 LangChain Tool 和 LangGraph 构建基础 ReAct Agent，并保留早期手写架构作为学习资料。

## 架构边界

```text
.env → Settings → LLM/Embedding Factory
                         ↓
用户输入 + thread_id + AgentContext(user_id)
                         ↓
LangGraph → AgentNode → ToolNode → LangChain Tool → Skill
    ↑          ↑            │             │
    │          └─ ToolMessage┘             └─ SqliteStore（按 user_id 保存长期记忆）
    └─ SqliteSaver（按 thread_id 持久化 State）
```

- `skills`：真正的业务能力，不依赖 LangChain 或 LangGraph。
- `tools`：使用 `@tool` 将 Skill 暴露给 LLM，不重复业务逻辑。
- `AgentNode`：调用绑定 Tools 的 LLM，负责决策。
- `ToolNode`：执行 LLM 生成的 Tool Calls。
- `StateGraph`：编排 AgentNode、ToolNode、条件路由和 ReAct 循环。
- `SqliteSaver`：按 `thread_id` 将 Checkpoint 保存到本地 SQLite 文件，支持程序重启后恢复会话。
- `SqliteStore`：按 `user_id` namespace 持久化用户明确要求保存的长期记忆。
- `Human-in-the-loop`：`save_memory` 和 `delete_memory` 在写入前通过 `interrupt()` 请求人工审批。
- `Error Recovery`：ToolNode 调用边界对错误分类；只有临时错误有限重试，超限后进入人工复核。
- `legacy`：早期手写 Agent 代码，不进入当前运行链。

## 环境要求

- Python 3.11 或更高版本
- DeepSeek API Key
- 首次使用长期记忆时可访问 Hugging Face，以下载本地多语言 Embedding 模型

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

程序启动后依次输入用户 ID 和会话 ID；直接回车分别使用 `default_user` 和 `default`。`thread_id` 定位 Checkpoint 会话，`user_id` 定位跨会话长期记忆。同一用户更换 thread 后不会自动继承旧消息，但仍可通过记忆工具检索自己明确保存的事实。输入 `exit` 或 `quit` 退出。

会话过程中可以输入以下 Checkpoint 调试命令：

```text
/state      查看当前会话的最新 StateSnapshot
/history    从新到旧查看当前会话的 Checkpoint 历史
/replay     按历史序号选择 Checkpoint，从其 next 节点重新执行
/fork       按历史序号选择 calculate 结果，修改后创建分支并继续
```

`/state` 和 `/history` 只读；`/replay` 和 `/fork` 会执行 Graph 并在 SQLite
中创建新的 Checkpoint。第一版只允许对无副作用的 `calculate` 路径执行
Time Travel。包含 pending interrupt、`save_memory` 或无法明确判断后继行为的
Checkpoint 会被拒绝；如果重放后的 Agent 新产生敏感调用，Graph 仍会通过
`interrupt()` 暂停，CLI 不会自动批准。

当模型选择教学用敏感工具 `save_memory` 时，程序会显示 JSON 审批信息：

```text
approve    使用 Command(resume={"approved": true}) 批准并继续
reject     使用 Command(resume={"approved": false, ...}) 拒绝并取消写入
exit       不处理审批，保留 SQLite 暂停状态后退出
```

如果在审批时退出，重新启动程序并输入相同 `user_id` 和 `thread_id`，CLI 会检测 SQLite 中尚未处理的 interrupt，并继续要求审批。批准后的记忆写入独立的 `data/memories.sqlite`；保存前还会检查明确意图和敏感凭据。

长期记忆工具包括：

```text
save_memory(content, memory_type)  明确意图 + 人工审批后保存
search_memory(query)               当前 user_id 内语义检索，最多3条
list_memories()                    列出当前用户的有效记忆
delete_memory(memory_id)           人工审批后软删除当前用户自己的记忆
```

`user_id` 和 Store 由 LangGraph `ToolRuntime` 注入，不会出现在 LLM 可填写的 Tool Schema 中。记忆使用本地 `minishlab/potion-multilingual-128M` Model2Vec 模型生成 256 维向量；模型第一次使用时下载到 Hugging Face 本机缓存，不需要新的付费 API Key。

教学工具 `unstable_tool` 不访问真实网络：同一任务前两次固定抛出
`TimeoutError`，第三次成功。临时错误的失败次数写入 AgentState；达到 3 次
执行上限后 Graph 通过 `interrupt()` 提供：

```text
retry     人工允许再执行一次
cancel    取消，不再调用工具
exit      保留失败复核状态并退出；下次使用相同 thread_id 恢复
```

参数错误返回 AgentNode 修正；权限、永久失败和副作用结果未知不会自动重试。

当前 Checkpoint 数据保存在 `data/checkpoints.sqlite`，长期记忆保存在 `data/memories.sqlite`。两个数据库均由 `.gitignore` 排除，不提交用户数据。Checkpoint 保存 Graph 执行状态；长期记忆保存按用户隔离、明确批准的简洁事实。

## 测试

```powershell
.venv\Scripts\python -m unittest discover -s tests -t . -v
```

默认测试使用 Fake/Mock LLM 和确定性测试 Embedding，不会请求 DeepSeek API，也不会下载真实模型。测试覆盖同会话恢复、不同会话隔离、跨 thread 长期记忆、跨用户隔离、跨进程恢复、敏感信息拒绝、Memory CRUD、人工审批、Replay/Fork、错误恢复及原有工具调用循环。

## 项目结构

```text
src/ai_agent_learning/
├── cli.py              # CLI 入口与最外层错误边界
├── checkpoint.py       # SQLite Checkpointer 路径和连接生命周期
├── memory_store.py     # SQLite 长期记忆 Store 路径和连接生命周期
├── config.py           # 配置加载和校验
├── llm.py              # DeepSeek LLM 创建
├── embeddings.py       # 本地多语言 Embedding 创建
├── logging_config.py   # 标准日志初始化
├── agent/              # State、AgentNode、Graph、错误恢复、Checkpoint 与 Time Travel
├── tools/              # LangChain Tool 适配层
└── skills/             # 业务能力

tests/
├── unit/               # 单模块测试
└── integration/        # ReAct Graph 集成测试

legacy/                 # 旧手写 Agent 学习代码
docs/                   # 学习文档
data/                   # 本地 Checkpoint 目录（数据库文件不提交 Git）
```

更完整的原理和学习路径参见 [docs/comprehension.md](docs/comprehension.md)。

## 当前范围

当前项目覆盖基础 LangGraph ReAct Agent、SQLite 持久化短期会话和第一阶段显式长期记忆，暂未实现：

- 自动 Memory Manager、自动事实提取与冲突合并
- RAG
- FastAPI
- Multi-Agent
- 部署与容器化
