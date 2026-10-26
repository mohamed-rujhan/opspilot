from langchain_core.language_models.chat_models import BaseChatModel

from opspilot.config import Settings


class LLMNotConfiguredError(RuntimeError):
    pass


def get_chat_model(settings: Settings, model: str | None = None) -> BaseChatModel:
    """Create a chat model for the configured provider (``model`` overrides the configured name)."""
    name = model or settings.llm_model
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMNotConfiguredError("Set OPSPILOT_GEMINI_API_KEY to use Gemini.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=name,
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise LLMNotConfiguredError("Set OPSPILOT_OPENAI_API_KEY to use OpenAI.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=name,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    raise LLMNotConfiguredError(f"Unknown LLM provider: {settings.llm_provider!r}")


def get_fallback_models(settings: Settings) -> list[BaseChatModel]:
    names = [n.strip() for n in settings.llm_fallback_models.split(",") if n.strip()]
    return [get_chat_model(settings, n) for n in names if n != settings.llm_model]
