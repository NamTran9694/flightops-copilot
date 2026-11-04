"""Central settings for the project."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")    # read secrets and settings from the .env file (if it exists)

# Every provider below speaks the same "OpenAI-compatible" chat API,
# so switching model = changing two lines in .env. All have a free option.
PROVIDERS = {
    #            base_url                                                   default model              key name
    "groq":   ("https://api.groq.com/openai/v1",                            "openai/gpt-oss-120b", "GROQ_API_KEY"),
    "ollama": ("http://localhost:11434/v1",                                 "qwen3:8b",                None),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/",  "gemini-flash-latest",     "GEMINI_API_KEY"),
}

@dataclass(frozen=True)
class Settings:
    db_path: Path = Path(os.getenv("FLIGHTOPS_DB", ROOT / "data" / "flightops.db"))
    docs_dir: Path = Path(os.getenv("FLIGHTOPS_DOCS", ROOT / "data" / "docs"))
    sql_row_limit: int = int(os.getenv("SQL_ROW_LIMIT", "200"))
    sql_timeout_s: float = float(os.getenv("SQL_TIMEOUT_S", 5))
    use_embeddings: bool = os.getenv("USE_EMBEDDINGS", "0") == "1"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # --- AI model settings
    provider: str = os.getenv("LLM_PROVIDER", "groq").lower()
    model_override: str = os.getenv("LLM_MODEL", "")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))

    @property
    def model(self) -> str:
        return self.model_override or PROVIDERS[self.provider][1]

settings = Settings()

def make_client():
    """Create an OpenAI-compatible client for the chosen provider."""
    from openai import OpenAI

    if settings.provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDERS must be one of {list(PROVIDERS)}")
    base_url, _, key_name = PROVIDERS[settings.provider]
    base_url = os.getenv("LLM_BASE_URL", base_url)
    if key_name is None:
        api_key = "ollama"  # Ollama runs on your computer and needs no key
    else:
        api_key = os.getenv(key_name)
        if not api_key:
            raise RuntimeError(f"Missing {key_name}. Add it to your .env file.")
    return OpenAI(base_url=base_url, api_key=api_key)    