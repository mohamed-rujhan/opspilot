from opspilot.config import Settings


def test_settings_read_prefixed_env_vars(monkeypatch):
    monkeypatch.setenv("OPSPILOT_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPSPILOT_LLM_MODEL", "gpt-4o-mini")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o-mini"


def test_api_keys_default_to_none(monkeypatch):
    monkeypatch.delenv("OPSPILOT_GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPSPILOT_OPENAI_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.gemini_api_key is None
    assert settings.openai_api_key is None
