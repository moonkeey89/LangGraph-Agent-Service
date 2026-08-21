from langchain_core.language_models.chat_models import BaseChatModel

from ai_agent_learning.agents.subagent import StatelessReActSubagent
from ai_agent_learning.tools import calculate


MATH_AGENT_PROMPT = """你是 Math Agent，只负责数学和预算计算。
只能使用 calculate；不能查询天气、景点、操作长期记忆或调用其他Agent。
只处理 Supervisor 给出的当前子任务，不假设你拥有主会话历史。
最终摘要必须包含计算表达式、简短过程和最终结果；不要返回内部消息历史。"""


def create_math_agent(llm: BaseChatModel) -> StatelessReActSubagent:
    return StatelessReActSubagent(
        agent_name="math_agent",
        llm=llm,
        tools=[calculate],
        system_prompt=MATH_AGENT_PROMPT,
    )
