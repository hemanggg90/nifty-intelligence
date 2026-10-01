"""scripts/purge_test_trades.py removes only the test-fixture signature, dry-run by default, with a backup."""
import importlib.util
import sqlite3
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "purge_test_trades.py"


def _load():
    spec = importlib.util.spec_from_file_location("purge_test_trades", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _db(path):
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE positions (id INTEGER PRIMARY KEY, instrument TEXT, strategy_name TEXT, security_id TEXT, entry_price REAL);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, order_id TEXT, instrument TEXT, strategy_name TEXT, security_id TEXT, price REAL);
        CREATE TABLE fills (id INTEGER PRIMARY KEY, order_id TEXT);
        INSERT INTO positions (instrument, strategy_name, security_id, entry_price) VALUES
            ('NIFTY','Momentum',NULL,100.05), ('NIFTY','Momentum',NULL,99.95),
            ('NIFTY','Momentum','73897',100.0),            -- has a contract: real
            ('NIFTY','Donchian Channel Breakout','73897',105.1),
            ('NIFTY','Momentum',NULL,250.0);               -- not the flat fixture price
        INSERT INTO orders (order_id, instrument, strategy_name, security_id, price) VALUES
            ('O1','NIFTY','Momentum',NULL,100.0), ('O2','NIFTY','Donchian Channel Breakout','73897',105.1);
        INSERT INTO fills (order_id) VALUES ('O1'), ('O2');
    """)
    c.commit()
    c.close()


def _counts(path):
    c = sqlite3.connect(path)
    out = [c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("positions", "orders", "fills")]
    c.close()
    return out


def test_dry_run_changes_nothing(tmp_path, monkeypatch, capsys):
    db = tmp_path / "t.db"
    _db(db)
    monkeypatch.setattr(sys, "argv", ["purge", "--db", str(db)])
    assert _load().main() == 0
    assert _counts(db) == [5, 2, 2] and "Dry run" in capsys.readouterr().out
    assert not list(tmp_path.glob("*backup*"))


def test_apply_backs_up_then_deletes_only_the_fixture_rows(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    _db(db)
    monkeypatch.setattr(sys, "argv", ["purge", "--db", str(db), "--apply"])
    assert _load().main() == 0
    assert _counts(db) == [3, 1, 1]  # 2 fixture positions, 1 order, 1 fill removed
    backups = list(tmp_path.glob("t.backup-*.db"))
    assert len(backups) == 1 and _counts(backups[0]) == [5, 2, 2]
    c = sqlite3.connect(db)
    kept = {r[0] for r in c.execute("SELECT strategy_name || ':' || COALESCE(security_id,'-') || ':' || entry_price FROM positions")}
    assert "Donchian Channel Breakout:73897:105.1" in kept and "Momentum:73897:100.0" in kept and "Momentum:-:250.0" in kept
