"""Database engine/session management."""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.database.models import Base

_engine = create_engine(
    SETTINGS.database_url,
    connect_args={"check_same_thread": False} if SETTINGS.database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)


def _sync_schema() -> None:
    """Add any model columns missing from existing tables (SQLite ADD COLUMN migration).

    ``create_all`` only creates tables that don't exist yet - it never alters an
    existing table when a new column is added to a model, which left this DB's
    ``orders``/``positions`` tables missing the option-trading columns and crashing
    ``08_Positions_and_Orders.py``.
    """
    inspector = inspect(_engine)
    existing_tables = set(inspector.get_table_names())
    with _engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing_columns:
                    continue
                col_type = column.type.compile(dialect=_engine.dialect)
                quoted_name = _engine.dialect.identifier_preparer.quote(column.name)
                conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {quoted_name} {col_type}"))


def init_db() -> None:
    Base.metadata.create_all(_engine)
    _sync_schema()


def get_engine():
    return _engine


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
