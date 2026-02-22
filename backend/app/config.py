from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://compliancescope:compliancescope@db:5432/compliancescope"
    openai_api_key: str = ""
    edgar_user_agent: str = "ComplianceScope ryan@example.com"

    # Embedding config
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # LLM config
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0  # Deterministic for compliance analysis — no creative drift

    # Chunking config
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # RAG config
    top_k: int = 5  # Number of chunks to retrieve for context

    # Risk analysis config
    risk_confidence_threshold: float = 0.7  # Only keep LLM risk flags above this

    class Config:
        env_file = ".env"


settings = Settings()