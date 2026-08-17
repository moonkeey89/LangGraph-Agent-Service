from langchain_openai import ChatOpenAI

from settings import Settings


def create_llm(settings: Settings) -> ChatOpenAI:
    llm_config = {
        "model": settings.deepseek_model,
        "api_key": settings.deepseek_api_key.get_secret_value(),
        "base_url": settings.deepseek_base_url,
    }

    if settings.deepseek_temperature is not None:
        llm_config["temperature"] = settings.deepseek_temperature

    return ChatOpenAI(**llm_config)
