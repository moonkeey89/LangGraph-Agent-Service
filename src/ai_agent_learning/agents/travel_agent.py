from langchain_core.language_models.chat_models import BaseChatModel

from ai_agent_learning.agents.subagent import StatelessReActSubagent
from ai_agent_learning.tools import get_weather, search_attraction


TRAVEL_AGENT_PROMPT = """你是 Travel Agent，只负责天气、景点和基础旅游信息。
只能使用 get_weather 和 search_attraction；不能计算预算、操作长期记忆、调用其他Agent。
只处理 Supervisor 给出的当前子任务，不假设你拥有主会话历史。
每一步最多调用一个工具，拿到结果后再决定是否调用下一个工具，不并行执行。
完成工具调用后返回简洁摘要，明确列出城市、天气和景点；不要返回内部消息历史。"""


def create_travel_agent(llm: BaseChatModel) -> StatelessReActSubagent:
    return StatelessReActSubagent(
        agent_name="travel_agent",
        llm=llm,
        tools=[get_weather, search_attraction],
        system_prompt=TRAVEL_AGENT_PROMPT,
    )
