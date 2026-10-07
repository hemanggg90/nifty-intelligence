"""Database engine/session management."""
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.database.models import Base



def normalise_database_url(url: str) -> str:
    """Accept the URL forms hosting dashboards hand out. `postgres://` (Heroku/Neon/Supabase style) and a bare
    `postgresql://` are rewritten to SQLAlchemy's `postgresql+psycopg2://`; everything else is untouched."""
    url = (url or "").strip()
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg2://" + url[len(prefix):]
    return url


def engine_options(url: str) -> dict:
    """create_engine keyword arguments for the backend in `url`.

    SQLite: allow use from the keeper/runner threads. Server databases (Postgres on Neon/Supabase): serverless
    hosts close idle connections, so test each connection before use and recycle them before the host does."""
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {"pool_pre_ping": True, "pool_recycle": 300, "pool_size": 5, "max_overflow": 5}


def backend_name(url: str | None = None) -> str:
    """"sqlite" / "postgresql" / ... - for display; never includes credentials."""
    return (url or DATABASE_URL).split(":", 1)[0].split("+", 1)[0] or "unknown"


def is_durable(url: str | None = None) -> bool:
    """False for the local SQLite file: on hosts with an ephemeral disk (Streamlit Cloud) it is wiped on every
    reboot, taking the trade history with it."""
    return backend_name(url) != "sqlite"


DATABASE_URL = normalise_database_url(SETTINGS.database_url)
_engine = create_engine(DATABASE_URL, **engine_options(DATABASE_URL))
SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)


def _sync_schema(engine=None) -> None:
    """Add any model columns missing from existing tables (SQLite ADD COLUMN migration).

    ``create_all`` only creates tables that don't exist yet - it never alters an
    existing table when a new column is added to a model, which left this DB's
    ``orders``/``positions`` tables missing the option-trading columns and crashing
    ``08_Positions_and_Orders.py``.
    """
    engine = engine or _engine
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            existing_columns = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing_columns:
                    continue
                col_type = column.type.compile(dialect=engine.dialect)
                quoted_name = engine.dialect.identifier_preparer.quote(column.name)
                conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {quoted_name} {col_type}"))


def init_db(engine=None) -> None:
    """Create missing tables and add missing columns (on `engine`, default the app's)."""
    engine = engine or _engine
    Base.metadata.create_all(engine)
    _sync_schema(engine)


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
