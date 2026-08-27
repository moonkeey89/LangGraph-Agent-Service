# AI Agent Learning

这是一个用于学习 AI Agent 工程开发的项目。当前实现使用 LangChain Tool 和 LangGraph 构建基础 ReAct Agent，并提供可独立启动的 Supervisor + Travel/Math Subagents 教学模式；早期手写架构继续作为学习资料保留。

## 架构边界

```text
.env → Settings → LLM/Embedding Factory
                         ↓
用户输入 + thread_id + AgentContext(user_id)
                         ↓
LangGraph → Memory Recall → AgentNode ↔ ToolNode → LangChain Tool → Skill
    │              │
    │              └─ 最终回答 → Memory Manager → Memory Executor → END
    │                                      │
    │                                      └─ SqliteStore（按 user_id 保存长期记忆）
    └─ SqliteSaver（按 thread_id 持久化 State）
```

- `skills`：真正的业务能力，不依赖 LangChain 或 LangGraph。
- `tools`：使用 `@tool` 将 Skill 暴露给 LLM，不重复业务逻辑。
- `AgentNode`：调用绑定 Tools 的 LLM，负责决策。
- `ToolNode`：执行 LLM 生成的 Tool Calls。
- `StateGraph`：编排 AgentNode、ToolNode、条件路由和 ReAct 循环。
- `SqliteSaver`：按 `thread_id` 将 Checkpoint 保存到本地 SQLite 文件，支持程序重启后恢复会话。
- `SqliteStore`：按 `user_id` namespace 持久化显式批准或经 Memory Policy 通过的长期记忆。
- `Memory Manager`：最终回答后只分析最新用户消息和当前用户 Top-K 记忆，输出结构化 ADD/UPDATE/DELETE/NONE。
- `Memory Executor`：用确定性代码校验置信度、敏感信息、候选 ID 和用户归属，通过后才调用 Memory Skill。
- `Memory Recall`：个人信息问题回答前，按当前 `user_id` 召回 Top-K 记忆并作为受控背景数据交给 AgentNode。
- `Human-in-the-loop`：`save_memory` 和 `delete_memory` 在写入前通过 `interrupt()` 请求人工审批。
- `Error Recovery`：ToolNode 调用边界对错误分类；只有临时错误有限重试，超限后进入人工复核。
- `legacy`：早期手写 Agent 代码，不进入当前运行链。

多 Agent 模式复用同一套主图基础设施：

```text
用户 → Memory Recall → Supervisor Agent
                         ├─ ask_travel_agent(task)
                         │      └─ Travel ReAct → get_weather/search_attraction
                         ├─ ask_math_agent(task)
                         │      └─ Math ReAct → calculate
                         └─ 直接回答
                                  ↓
                         Supervisor回答草稿
                                  ↓
                         Critic结构化审查
                          ├─ PASS → Finalize
                          └─ REVISE → 最多修订一次 → Finalize
                                                   ↓
                         Memory Manager（每轮一次）→ END
```

Supervisor 的主 State 和高层 handoff 记录继续由 `thread_id` Checkpoint；两个 Subagent 每次只接收一个最小任务字符串，无 Checkpointer、无长期记忆、无独立 `thread_id`。它们只返回结构化结果摘要，内部 ToolMessage 不进入主会话。

Critic 不绑定任何 Tool，只接收当前用户请求、Supervisor 草稿、当前轮 Subagent 最终摘要和直接约束。草稿不会直接留在用户会话消息中；PASS 时原草稿进入 Finalize，REVISE 时由无工具 RevisionNode 最多修改一次。Critic 或修订模型失败时安全回退到原草稿。Critic 内部 JSON 和提示不会写入 `messages`，Memory Manager 只在 Finalize 后执行一次。

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

启动 Supervisor 多 Agent 教学模式：

```powershell
.venv\Scripts\python multi_agent_main.py
```

也可以在安装后使用：

```powershell
.venv\Scripts\python -m ai_agent_learning
```

以 editable 方式安装后，也可以运行：

```powershell
ai-agent-learning-multi
```

启动非流式 FastAPI 服务（服务默认复用 Supervisor + Subagents + Critic 主图）：

```powershell
.venv\Scripts\python -m uvicorn ai_agent_learning.api.app:app --host 127.0.0.1 --port 8000
```

FastAPI lifespan 在进程启动时只创建一次 LLM、编译后的 LangGraph、认证数据库、SQLite
Checkpointer、长期记忆Store和RAG Chroma客户端，在关闭时释放相应连接。HTTP接口为：

```text
GET  /health
POST /api/v1/auth/register
POST /api/v1/auth/login
GET  /api/v1/auth/me
POST /api/v1/auth/logout
POST /api/v1/agent/invoke
POST /api/v1/agent/resume
POST /api/v1/agent/stream
```

业务API通过登录后签发的HttpOnly Session Cookie识别用户，`X-User-ID`不再参与
生产身份判断。Session原始Token只写入Cookie，`data/auth.sqlite`仅保存Token摘要。
POST、PATCH和DELETE等修改请求还必须同时携带CSRF Cookie对应的`X-CSRF-Token`
Header。`thread_id`仍在JSON请求体中；首次调用会把可信Session用户与thread的归属
写入Checkpoint，后续请求无法使用另一个登录用户接管同一个thread。
已有但不含该归属字段的旧 CLI thread 不会被 API 自动认领，请为 API 使用新的
`thread_id`。CLI 的原有入口和数据库行为不受影响。

SSE流式接口使用与`/invoke`相同的Session Cookie和CSRF保护。原生JavaScript页面
启动时先调用`/api/v1/auth/me`恢复会话；未登录时只显示登录/注册界面，不加载项目、
任务、成果、知识库或Agent数据。注册只创建账户，随后需要登录。登录成功后服务端设置
HttpOnly Session Cookie和可由前端读取的CSRF Cookie，所有修改请求由统一请求层自动
补充服务端声明的CSRF Header。

修改请求的认证链为：

```text
Session Cookie → Session Token摘要查询 → 过期/撤销/is_active检查
              → CSRF Cookie + X-CSRF-Token + Session内CSRF摘要校验
              → 可信user_id → 原有业务隔离链
```

响应采用 `text/event-stream`，公开事件依次可能包含：

```text
started → progress* → token* → completed
started → progress* → interrupted
started → progress* → error
```

流式调用只执行一次原 Supervisor Graph。节点进度来自 LangGraph `updates`；文本
片段来自 `messages`，但 Supervisor 草稿必须先经过 Critic，因此真实片段会暂存到
Finalize 确定答案后再发送。Critic、Memory Manager 和 Subagent 的模型片段不会对外
输出；若模型集成未提供可验证的真实片段，则只返回进度和完整 `completed.answer`，
不会用定时器伪造 token。流中出现 `interrupted` 后，仍使用原 `/resume` 接口恢复。

浏览器聊天页可直接访问：

```text
http://127.0.0.1:8000/
```

页面使用原生 HTML、CSS 和 JavaScript，通过 `fetch()` 发送 POST SSE 请求，并从
`response.body` 持续解析公开事件。页面不再保存或发送开发阶段的user ID，也不能读取
HttpOnly Session Token。`localStorage`只保存当前thread和非敏感的页面选择；退出、
Session失效或切换账户时会清除当前Project、Task、Artifact、Run、知识库与会话显示状态。
前端不会保存模型密钥、Checkpoint、密码或长期记忆正文。

首次使用RAG前，先离线索引项目自带的演示文档：

```powershell
.venv\Scripts\python -m ai_agent_learning.knowledge.cli `
  examples\knowledge\demo_agent_handbook.md `
  --knowledge-base-id demo `
  --owner-user-id moon
```

入库只在命令执行时解析和切分文档，用户提问时不会重新读取全部文件。默认数据写入
`data/knowledge_chroma/`，管理目录写入`data/knowledge_catalog.sqlite`，受控源文件写入
`data/knowledge_sources/`；这些运行数据都不会提交Git。CLI和网页上传调用同一个
`KnowledgeLibraryService → KnowledgeIngestor`。如果项目中已有升级前直接写入Chroma的
文档，需要重新执行一次入库命令，目录表登记为`ready`后才能被在线检索。

浏览器访问首页后，可以进入“知识库”页新建知识库、拖拽上传TXT/Markdown/PDF、查看
状态和删除文档。回到“对话”页，在“本轮知识库”中选择当前用户拥有的知识库后再提问：

```text
根据内部手册，星河项目的内部代号是什么？
```

正确答案应包含演示文档中的唯一事实`ORBIT-731`，回答下方显示实际文件来源。当前
浏览器把选择的ID作为受控请求字段发送，后端会再次检查该知识库属于当前Session用户；
自然语言和LLM都不能修改所有者或绕过检查。未选择知识库时不会强制执行RAG。
同一文档再次入库会使用稳定document/chunk ID进行upsert，并删除该文档已失效的旧
chunk；修改文档后直接重复同一命令即可完成重新索引。

知识库管理接口包括：

```text
POST   /api/v1/knowledge-bases
GET    /api/v1/knowledge-bases
GET    /api/v1/knowledge-bases/{knowledge_base_id}
DELETE /api/v1/knowledge-bases/{knowledge_base_id}
POST   /api/v1/knowledge-bases/{knowledge_base_id}/documents
GET    /api/v1/knowledge-bases/{knowledge_base_id}/documents
DELETE /api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}
```

测试代码如需模拟用户，只能通过FastAPI `dependency_overrides[get_user_id]`注入；
生产配置没有启用`X-User-ID`兼容开关。

认证升级不会自动把旧的`moon`、`user_001`等开发数据认领给新账户。新注册账户使用
服务端生成的`usr_...`标识，因此旧owner字段与新用户天然隔离。安全迁移方式是先备份
全部data目录，再由明确知道旧owner与目标账户对应关系的离线迁移工具逐项迁移；在该
工具实现前，建议为新账户重新创建知识库并重新入库文档，不要直接批量修改多个SQLite
文件或Checkpoint/长期记忆namespace。

程序启动后依次输入用户 ID 和会话 ID；直接回车分别使用 `default_user` 和 `default`。`thread_id` 定位 Checkpoint 会话，`user_id` 定位跨会话长期记忆。同一用户更换 thread 后不会自动继承旧消息，但仍可检索自己已保存且有效的长期事实。输入 `exit` 或 `quit` 退出。

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

普通 ReAct 流程得到最终回答后，Memory Manager 使用相同 DeepSeek 模型的结构化输出能力判断最新用户消息。明显的天气、计算、寒暄、普通问句会直接得到 `NONE`，不产生额外模型调用。候选消息只读取当前 `user_id` 的 Top-K 旧记忆；模型看不到真实 `user_id`，也不能直接写 Store。默认只有置信度不低于 `0.75` 且通过代码策略的决定才执行。

```text
ADD     新增稳定且不重复的用户事实
UPDATE  在当前用户候选范围内更新一条已有记忆
DELETE  在当前用户候选范围内软删除一条已有记忆
NONE    不修改长期记忆
```

显式“请记住”仍由原有 `save_memory + interrupt` 流程处理；Memory Manager 检测到本轮记忆工具已经保存成功、被用户拒绝或被安全策略最终否决后会跳过，避免批准、拒绝或恢复执行之后产生重复写入。

如果显式 `save_memory` 只是因为用户没有说“请记住”而拒绝，本轮不会被视为已经完成记忆处理，后置 Memory Manager 仍可分析“我喜欢”“我爱”“最喜欢”等普通陈述。新 thread 中出现“我是谁”“我喜欢什么”等个人问题时，Memory Recall 会在 Agent 回答前查询同一 `user_id` 的 Top-K 长期记忆；不同 `user_id` 的 namespace 仍然严格隔离。

教学工具 `unstable_tool` 不访问真实网络：同一任务前两次固定抛出
`TimeoutError`，第三次成功。临时错误的失败次数写入 AgentState；达到 3 次
执行上限后 Graph 通过 `interrupt()` 提供：

```text
retry     人工允许再执行一次
cancel    取消，不再调用工具
exit      保留失败复核状态并退出；下次使用相同 thread_id 恢复
```

参数错误返回 AgentNode 修正；权限、永久失败和副作用结果未知不会自动重试。

当前 Checkpoint 数据保存在 `data/checkpoints.sqlite`，长期记忆保存在 `data/memories.sqlite`。两个数据库均由 `.gitignore` 排除，不提交用户数据。Checkpoint 保存 Graph 执行状态；长期记忆保存按用户隔离、通过显式审批或 Memory Policy 的简洁事实。

## 测试

```powershell
.venv\Scripts\python -m unittest discover -s tests -t . -v
```

默认测试使用 Fake/Mock LLM 和确定性测试 Embedding，不会请求 DeepSeek API，也不会下载真实模型。测试覆盖同会话恢复、不同会话隔离、跨 thread 长期记忆、跨用户隔离、跨进程恢复、Memory Manager 四种决定、安全策略、敏感信息拒绝、Memory CRUD、人工审批、Replay/Fork、错误恢复、FastAPI invoke/resume/SSE、RAG入库/检索/隔离/来源及原有工具调用循环。
多 Agent 测试还覆盖旅游/计算单领域路由、跨领域串行协作、能力隔离、普通问题直答、部分失败整合、重复 handoff 熔断、最大调用次数以及 Supervisor 主会话的 SQLite 恢复。Critic 测试覆盖 PASS、遗漏修订、矛盾修订、单次修订上限、结构化输出失败降级、上下文最小化以及 Memory Manager 执行顺序。

## 项目结构

```text
src/ai_agent_learning/
├── cli.py              # CLI 入口与最外层错误边界
├── multi_agent_cli.py  # Supervisor 模式入口，复用原 CLI 交互
├── checkpoint.py       # SQLite Checkpointer 路径和连接生命周期
├── memory_store.py     # SQLite 长期记忆 Store 路径和连接生命周期
├── config.py           # 配置加载和校验
├── llm.py              # DeepSeek LLM 创建
├── embeddings.py       # 本地多语言 Embedding 创建
├── logging_config.py   # 标准日志初始化
├── agent/              # State、AgentNode、Memory Manager、Graph、错误恢复与 Time Travel
├── agents/             # Supervisor、Travel/Math/Knowledge、Critic与高层handoff Tools
├── api/                # FastAPI lifespan、路由、DTO、依赖与 AgentService
├── knowledge/          # 文档Loader、切分、Chroma、Retriever与离线入库CLI
├── tools/              # LangChain Tool 适配层
└── skills/             # 业务能力

tests/
├── unit/               # 单模块测试
└── integration/        # ReAct Graph 集成测试

legacy/                 # 旧手写 Agent 学习代码
docs/                   # 学习文档
data/                   # 本地 Checkpoint 目录（数据库文件不提交 Git）
frontend/               # 原生浏览器聊天页、样式和独立 SSE 解析器
examples/knowledge/     # 不含敏感信息的RAG演示文档
```

更完整的原理和学习路径参见 [docs/comprehension.md](docs/comprehension.md)。

## 当前范围

当前项目覆盖基础 LangGraph ReAct Agent、SQLite 持久化短期会话、长期记忆、最小 Supervisor/Subagents 协作、单次 Critic 审查、本地Chroma RAG、FastAPI非流式与SSE接口，以及原生JavaScript教学聊天页，暂未实现：

- 复杂自动事实拆分与冲突合并
- OCR、联网搜索、混合检索、BM25、Query Rewrite、HyDE或Reranker
- WebSocket、SSE 断线补发或完整会话历史页面
- 并行 Subagent、多轮反思、多个 Critic、Writer Agent 或群聊
- 部署与容器化
