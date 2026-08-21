from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from ai_agent_learning.agent.graph import build_graph
from ai_agent_learning.agent.memory_manager import (
    DEFAULT_MEMORY_CONFIDENCE_THRESHOLD,
)
from ai_agent_learning.agents.math_agent import create_math_agent
from ai_agent_learning.agents.subagent_tools import (
    DEFAULT_MAX_SUBAGENT_CALLS,
    SubagentInvoker,
    create_subagent_tools,
)
from ai_agent_learning.agents.travel_agent import create_travel_agent
from ai_agent_learning.tools import (
    delete_memory,
    list_memories,
    save_memory,
    search_memory,
)


SUPERVISOR_PROMPT = """你是多Agent系统的 Supervisor，负责理解完整任务并统一回答用户。
旅游天气和景点任务必须委派给 ask_travel_agent；数学和预算计算必须委派给 ask_math_agent。
你不能直接执行 get_weather、search_attraction 或 calculate，因为这些底层工具没有绑定给你。
跨领域任务按顺序调用所需Subagent，每次只调用一个；收集全部结果后只输出一个整合答案。
普通知识问题可以直接回答，不要为了展示多Agent而调用Subagent。
传给Subagent的task只包含完成子任务必需的信息，不复制完整会话历史；如长期记忆与任务相关，
只在task中包含必要事实，绝不传递user_id、thread_id或无关记忆。
Subagent返回的JSON是结果摘要：status=failed时说明该部分失败，并继续整合其他已成功结果。
没有新信息时禁止以完全相同task重复调用同一Subagent。"""


SUPERVISOR_MEMORY_TOOLS: Sequence[BaseTool] = (
    save_memory,
    search_memory,
    list_memories,
    delete_memory,
)


def build_supervisor_graph(
    llm: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    memory_manager_llm: BaseChatModel | None = None,
    memory_confidence_threshold: float = DEFAULT_MEMORY_CONFIDENCE_THRESHOLD,
    max_subagent_calls: int = DEFAULT_MAX_SUBAGENT_CALLS,
    travel_agent: SubagentInvoker | None = None,
    math_agent: SubagentInvoker | None = None,
):
    """Compile the checkpointed Supervisor around two stateless specialists."""
    travel = travel_agent or create_travel_agent(llm)
    math = math_agent or create_math_agent(llm)
    handoff_tools = create_subagent_tools(
        travel,
        math,
        max_subagent_calls=max_subagent_calls,
    )
    supervisor_tools = [*handoff_tools, *SUPERVISOR_MEMORY_TOOLS]
    return build_graph(
        llm,
        supervisor_tools,
        checkpointer=checkpointer,
        store=store,
        memory_manager_llm=memory_manager_llm,
        memory_confidence_threshold=memory_confidence_threshold,
        agent_system_prompt=SUPERVISOR_PROMPT,
    )
