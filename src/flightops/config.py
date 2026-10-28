"""Central settings for the project."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    db_path: Path = Path(os.getenv("FLIGHTOPS_DB", ROOT / "data" / "flightops.db"))
    docs_dir: Path = Path(os.getenv("FLIGHTOPS_DOCS", ROOT / "data" / "docs"))
    sql_row_limit: int = int(os.getenv("SQL_ROW_LIMIT", "200"))
    sql_timeout_s: float = float(os.getenv("SQL_TIMEOUT_S", 5))
    use_embeddings: bool = os.getenv("USE_EMBEDDINGS", "0") == "1"
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


settings = Settings()