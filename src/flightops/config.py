"""Central settings for the project."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class Settings:
    db_path: Path = Path(os.getenv("FLIGHTOPS_DB", ROOT / "data" / "flightops.db"))
    sql_row_limit: int = int(os.getenv("SQL_ROW_LIMIT", "200"))
    sql_timeout_s: float = float(os.getenv("SQL_TIMEOUT_S", 5))


settings = Settings()