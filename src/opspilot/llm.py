from langchain_core.language_models.chat_models import BaseChatModel

from opspilot.config import Settings


class LLMNotConfiguredError(RuntimeError):
    pass


def get_chat_model(settings: Settings) -> BaseChatModel:
    """Create the chat model for the configured provider."""
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise LLMNotConfiguredError("Set OPSPILOT_GEMINI_API_KEY to use Gemini.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
        )
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise LLMNotConfiguredError("Set OPSPILOT_OPENAI_API_KEY to use OpenAI.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
        )
    raise LLMNotConfiguredError(f"Unknown LLM provider: {settings.llm_provider!r}")
