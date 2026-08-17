import logging

from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from ai_agent_learning.agent import build_graph
from ai_agent_learning.config import Settings
from ai_agent_learning.llm import create_llm
from ai_agent_learning.logging_config import configure_logging
from ai_agent_learning.tools import TOOLS


logger = logging.getLogger(__name__)


def create_agent_app(settings: Settings):
    llm = create_llm(settings)
    return build_graph(llm, TOOLS)


def run_cli(app) -> None:
    while True:
        try:
            user_input = input("请输入（输入 exit 退出）:").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if user_input.lower() in {"exit", "quit"}:
            return

        if not user_input:
            continue

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]}
            )
        except Exception:
            logger.exception("Agent request failed")
            print("抱歉，本次请求执行失败，请稍后重试。")
            continue

        print(result["messages"][-1].content)


def main() -> int:
    try:
        settings = Settings()
    except ValidationError:
        configure_logging("ERROR")
        logger.error("配置无效，请检查 .env 中的 DeepSeek 配置")
        return 1

    configure_logging(settings.log_level)

    try:
        app = create_agent_app(settings)
    except Exception:
        logger.exception("Agent startup failed")
        return 1

    logger.info("AI Agent started with model %s", settings.deepseek_model)
    run_cli(app)
    return 0
