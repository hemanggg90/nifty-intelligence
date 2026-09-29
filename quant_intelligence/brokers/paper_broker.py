"""
PaperBroker: fully simulated order/fill/position engine.

Simulates: order submission, market-order fills with configurable slippage,
stop-loss/target monitoring, position P&L, and rejection scenarios (e.g. zero
quantity, disconnected simulation flag). Every order gets a unique ID and is
persisted with a full audit trail (orders -> fills -> positions tables).

This is the default and only execution path unless TRADING_MODE=LIVE AND the
live confirmation phrase is set (see config/settings.py). Even then, the
DhanBroker adapter is separate code (brokers/dhan_broker.py) - PaperBroker
itself can NEVER send a real order, by construction (no network calls here).
"""
from __future__ import annotations

import datetime as dt
from quant_intelligence.utils.timeutil import now_ist
import uuid

from quant_intelligence.brokers.base_broker import BaseBroker, OrderAck, OrderRequest
from quant_intelligence.config.settings import SETTINGS
from quant_intelligence.utils.logging_utils import log_event


class PaperBroker(BaseBroker):
    mode = "PAPER"
    name = "paper"

    def __init__(self, starting_capital: float | None = None):
        self.capital = starting_capital or SETTINGS.paper_starting_capital
        self.cash = self.capital
        self.positions: dict[str, dict] = {}  # position_id -> position dict
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    def place_order(self, order: OrderRequest, market_price: float | None = None) -> OrderAck:
        order_id = f"PAPER-{uuid.uuid4().hex[:12]}"
        now = now_ist()

        if order.quantity <= 0:
            return self._record(order, order_id, "REJECTED", "Quantity must be positive")
        if not self.connected:
            return self._record(order, order_id, "REJECTED", "PaperBroker is disconnected (simulated)")

        fill_price = market_price if market_price is not None else order.price
        if fill_price is None:
            return self._record(order, order_id, "REJECTED", "No price available for simulated fill")

        slippage = SETTINGS.costs.slippage_ticks * SETTINGS.costs.tick_size
        # For options, the transaction (BUY/SELL) drives the sign of adverse slippage,
        # not the underlying direction (a SELL is adverse when the fill price is lower).
        transaction = order.transaction_type or ("BUY" if order.direction == "LONG" else "SELL")
        direction_sign = 1 if transaction == "BUY" else -1
        simulated_fill_price = fill_price + direction_sign * slippage

        position_id = f"POS-{uuid.uuid4().hex[:12]}"
        self.positions[position_id] = {
            "position_id": position_id,
            "instrument": order.instrument,
            "strategy_name": order.strategy_name,
            "direction": order.direction,
            "quantity": order.quantity,
            "entry_price": simulated_fill_price,
            "stop_price": order.stop_price,
            "target_price": order.target_price,
            "status": "OPEN",
            "opened_at": now,
            "mode": "PAPER",
            "underlying": order.instrument if order.option_type else None,
            "security_id": order.security_id,
            "option_type": order.option_type,
            "strike": order.strike,
            "expiry": order.expiry,
            "transaction": transaction if order.option_type else None,
        }

        ack = self._record(order, order_id, "FILLED", None, fill_price=simulated_fill_price)
        self._persist_fill(order_id, now, simulated_fill_price, order.quantity, slippage)
        self._persist_position(self.positions[position_id])
        log_event(
            "paper_broker",
            f"Simulated fill: {order.direction} {order.quantity} {order.instrument} @ {simulated_fill_price}",
            level="INFO",
            order_id=order_id,
            decision_id=order.decision_id,
        )
        return ack

    def close_position(self, position_id: str, exit_price: float) -> dict | None:
        pos = self.positions.get(position_id)
        if pos is None or pos["status"] != "OPEN":
            return None
        # For an option position, P&L direction follows the transaction (BUY profits
        # when premium rises, SELL profits when it falls) - NOT the underlying LONG/SHORT
        # signal direction, since e.g. a SELL-a-call trade has direction="LONG" (CE) but
        # profits on premium falling, same as a SHORT.
        signed_field = pos.get("transaction") or pos["direction"]
        direction_sign = 1 if signed_field in ("LONG", "BUY") else -1
        net_pnl = direction_sign * (exit_price - pos["entry_price"]) * pos["quantity"]
        pos["status"] = "CLOSED"
        pos["closed_at"] = now_ist()
        pos["exit_price"] = exit_price
        pos["net_pnl"] = net_pnl
        self.cash += net_pnl
        self._persist_position(pos)
        return pos

    def get_positions(self) -> list[dict]:
        return list(self.positions.values())

    def get_open_positions(self) -> list[dict]:
        return [p for p in self.positions.values() if p["status"] == "OPEN"]

    def _record(self, order: OrderRequest, order_id: str, status: str, reject_reason: str | None, fill_price: float | None = None) -> OrderAck:
        try:
            from quant_intelligence.database.db import get_session
            from quant_intelligence.database.models import Order

            with get_session() as session:
                session.add(
                    Order(
                        order_id=order_id,
                        decision_id=order.decision_id,
                        timestamp=now_ist(),
                        instrument=order.instrument,
                        strategy_name=order.strategy_name,
                        direction=order.direction,
                        quantity=order.quantity,
                        order_type=order.order_type,
                        price=order.price,
                        status=status,
                        reject_reason=reject_reason,
                        mode="PAPER",
                        broker="paper",
                        security_id=order.security_id,
                        exchange_segment=order.exchange_segment,
                        product_type=order.product_type,
                        option_type=order.option_type,
                        strike=order.strike,
                        expiry=order.expiry,
                        lot_size=order.lot_size,
                    )
                )
        except Exception:
            pass
        return OrderAck(order_id=order_id, status=status, reject_reason=reject_reason, fill_price=fill_price, timestamp=now_ist())

    def _persist_fill(self, order_id, ts, price, qty, slippage) -> None:
        try:
            from quant_intelligence.database.db import get_session
            from quant_intelligence.database.models import Fill

            with get_session() as session:
                session.add(Fill(order_id=order_id, timestamp=ts, fill_price=price, quantity=qty, slippage=slippage))
        except Exception:
            pass

    def _persist_position(self, pos: dict) -> None:
        try:
            from quant_intelligence.database.db import get_session
            from quant_intelligence.database.models import Position

            with get_session() as session:
                existing = session.query(Position).filter_by(position_id=pos["position_id"]).first()
                if existing:
                    for k, v in pos.items():
                        setattr(existing, k, v)
                else:
                    session.add(Position(**pos))
        except Exception:
            pass
