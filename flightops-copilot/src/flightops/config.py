"""Central settings. Values come from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path(os.getenv("FLIGHTOPS_DB", ROOT / "data" / "flightops.db"))
    docs_dir: Path = Path(os.getenv("FLIGHTOPS_DOCS", ROOT / "data" / "docs"))
    model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2048"))
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "8"))
    sql_row_limit: int = int(os.getenv("SQL_ROW_LIMIT", "200"))
    sql_timeout_s: float = float(os.getenv("SQL_TIMEOUT_S", "5"))
    use_embeddings: bool = os.getenv("USE_EMBEDDINGS", "0") == "1"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


settings = Settings()
