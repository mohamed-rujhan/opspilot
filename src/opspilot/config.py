from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings, read from environment variables or a local .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="OPSPILOT_", extra="ignore")

    env: str = "development"
    log_level: str = "INFO"

    # LLM
    llm_provider: str = "gemini"
    llm_model: str = "gemini-flash-latest"  # alias that follows Google's current Flash model
    llm_fallback_models: str = "gemini-flash-lite-latest"  # comma-separated, tried in order on errors
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 1
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    # Optional, for cost estimates in evals (USD per 1M tokens)
    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0

    # Retrieval
    policies_dir: Path = PROJECT_ROOT / "data" / "policies"
    retriever: str = "bm25"  # "bm25" or "embeddings"
    embedding_model: str = "models/gemini-embedding-001"
    retrieval_top_k: int = 3

    # Integrations
    ticket_api_url: str = "http://localhost:8001"
    ticket_db_path: Path = PROJECT_ROOT / "data" / "tickets.sqlite3"
    holiday_api_url: str = "https://date.nager.at/api/v3"
    holiday_country: str = "GB"
    http_timeout_seconds: float = 10.0

    # Agent
    max_agent_steps: int = 12
    max_input_chars: int = 2000


@lru_cache
def get_settings() -> Settings:
    return Settings()
