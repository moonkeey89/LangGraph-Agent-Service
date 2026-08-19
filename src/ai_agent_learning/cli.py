import logging

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from ai_agent_learning.agent import (
    build_graph,
    show_current_state,
    show_state_history,
)
from ai_agent_learning.config import Settings
from ai_agent_learning.llm import create_llm
from ai_agent_learning.logging_config import configure_logging
from ai_agent_learning.tools import TOOLS


logger = logging.getLogger(__name__)
DEFAULT_THREAD_ID = "default"


def create_agent_app(settings: Settings):
    llm = create_llm(settings)
    checkpointer = InMemorySaver()
    return build_graph(llm, TOOLS, checkpointer=checkpointer)


def prompt_thread_id() -> str | None:
    try:
        thread_id = input(
            f"请输入会话 ID（直接回车使用 {DEFAULT_THREAD_ID}）:"
        ).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    return thread_id or DEFAULT_THREAD_ID


def run_cli(app, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}

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

        if user_input.lower() in {"/state", "/history"}:
            try:
                if user_input.lower() == "/state":
                    show_current_state(app, thread_id)
                else:
                    show_state_history(app, thread_id)
            except Exception:
                logger.exception("Checkpoint inspection failed")
                print("状态查看失败，请检查当前 Graph 是否配置了 Checkpointer。")
            continue

        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
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

    thread_id = prompt_thread_id()
    if thread_id is None:
        return 0

    logger.info(
        "AI Agent started with model %s and thread %s",
        settings.deepseek_model,
        thread_id,
    )
    run_cli(app, thread_id)
    return 0
