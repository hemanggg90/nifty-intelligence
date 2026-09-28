"""Initialize the local SQLite (or configured) database with all tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_intelligence.database.db import init_db

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
