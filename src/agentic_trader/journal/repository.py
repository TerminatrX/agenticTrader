"""SQLite-backed journal.

SQLite rather than JSONL because two questions asked constantly — "have I
already submitted this client_key?" and "what is my realized R by strategy?" —
are a lookup and an aggregate, and both are miserable over append-only text.

The `client_key` unique constraint on `trades` is the durable half of the
idempotency story. The risk engine generates a stable key and preflight checks
it, but only a database constraint survives a process crash between the check
and the write. If a duplicate reaches `record_trade`, it raises rather than
silently upserting: a duplicate key means the cycle logic re-fired, and that is
worth failing loudly over.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from agentic_trader.journal.models import AuditEntry, CycleOutcome, TradeRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id        TEXT NOT NULL,
    occurred_at     TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    reference_price TEXT,
    confidence      REAL NOT NULL DEFAULT 0,
    signal_strength TEXT,
    original_confidence REAL,
    adjusted_confidence REAL,
    thesis          TEXT,
    invalidation_reason TEXT,
    reasons         TEXT NOT NULL DEFAULT '[]',
    failed_conditions TEXT NOT NULL DEFAULT '[]',
    risk_breaches   TEXT NOT NULL DEFAULT '[]',
    critic_notes    TEXT NOT NULL DEFAULT '[]',
    snapshot_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_symbol ON audit(symbol);
CREATE INDEX IF NOT EXISTS idx_audit_cycle  ON audit(cycle_id);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    client_key      TEXT NOT NULL UNIQUE,
    symbol          TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    mode            TEXT NOT NULL,
    opened_at       TEXT NOT NULL,
    entry_price     TEXT NOT NULL,
    quantity        TEXT NOT NULL,
    notional        TEXT NOT NULL,
    stop_price      TEXT,
    target_price    TEXT,
    entry_rationale TEXT NOT NULL DEFAULT '[]',
    thesis          TEXT,
    invalidation_reason TEXT,
    sector          TEXT,
    closed_at       TEXT,
    exit_price      TEXT,
    exit_reason     TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_open   ON trades(closed_at);
"""


# Columns added after the first schema shipped. `CREATE TABLE IF NOT EXISTS`
# leaves an existing table untouched, so a journal created before these fields
# existed would otherwise fail on insert with a confusing "no such column".
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "audit": {
        "original_confidence": "REAL",
        "adjusted_confidence": "REAL",
        "thesis": "TEXT",
        "invalidation_reason": "TEXT",
    },
    "trades": {
        "thesis": "TEXT",
        "invalidation_reason": "TEXT",
        "sector": "TEXT",
    },
}


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Additive-only migration. Never drops or rewrites existing data."""
    for table, columns in _ADDED_COLUMNS.items():
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, sql_type in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")


class JournalRepository:
    """Append-oriented store for audit entries and trades."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            _add_missing_columns(conn)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ audit

    def record_audit(self, entry: AuditEntry) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO audit (
                    cycle_id, occurred_at, symbol, strategy, outcome,
                    reference_price, confidence, signal_strength,
                    original_confidence, adjusted_confidence,
                    thesis, invalidation_reason,
                    reasons, failed_conditions, risk_breaches, critic_notes,
                    snapshot_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    entry.cycle_id,
                    entry.occurred_at.isoformat(),
                    entry.symbol,
                    entry.strategy,
                    entry.outcome.value,
                    _s(entry.reference_price),
                    entry.confidence,
                    entry.signal_strength,
                    entry.original_confidence,
                    entry.adjusted_confidence,
                    entry.thesis,
                    entry.invalidation_reason,
                    json.dumps(entry.reasons),
                    json.dumps(entry.failed_conditions),
                    json.dumps(entry.risk_breaches),
                    json.dumps(entry.critic_notes),
                    json.dumps(entry.snapshot_json) if entry.snapshot_json else None,
                ),
            )

    def recent_audit(self, limit: int = 50, symbol: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM audit"
        params: list[Any] = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            return [_audit_row(r) for r in conn.execute(query, params).fetchall()]

    # ----------------------------------------------------------------- trades

    def record_trade(self, trade: TradeRecord) -> None:
        """Insert a new trade. Raises on a duplicate `client_key`."""
        with self._connect() as conn:
            try:
                conn.execute(
                    """INSERT INTO trades (
                        client_key, symbol, strategy, mode, opened_at,
                        entry_price, quantity, notional, stop_price, target_price,
                        entry_rationale, thesis, invalidation_reason, sector,
                        closed_at, exit_price, exit_reason
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        trade.client_key,
                        trade.symbol,
                        trade.strategy,
                        trade.mode,
                        trade.opened_at.isoformat(),
                        _s(trade.entry_price),
                        _s(trade.quantity),
                        _s(trade.notional),
                        _s(trade.stop_price),
                        _s(trade.target_price),
                        json.dumps(trade.entry_rationale),
                        trade.thesis,
                        trade.invalidation_reason,
                        trade.sector,
                        trade.closed_at.isoformat() if trade.closed_at else None,
                        _s(trade.exit_price),
                        trade.exit_reason,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"client_key {trade.client_key} already recorded — "
                    "the cycle re-fired; no duplicate order was written"
                ) from exc

    def close_trade(
        self,
        client_key: str,
        exit_price: Decimal,
        closed_at: datetime,
        exit_reason: str,
    ) -> None:
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE trades SET closed_at = ?, exit_price = ?, exit_reason = ?
                   WHERE client_key = ? AND closed_at IS NULL""",
                (closed_at.isoformat(), _s(exit_price), exit_reason, client_key),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"no open trade with client_key {client_key}")

    def has_client_key(self, client_key: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM trades WHERE client_key = ? LIMIT 1", (client_key,)
            ).fetchone()
        return row is not None

    def known_client_keys(self) -> set[str]:
        with self._connect() as conn:
            return {r["client_key"] for r in conn.execute("SELECT client_key FROM trades")}

    def open_trades(self) -> list[TradeRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM trades WHERE closed_at IS NULL").fetchall()
        return [_trade_from_row(r) for r in rows]

    def closed_trades(self, strategy: str | None = None) -> list[TradeRecord]:
        query = "SELECT * FROM trades WHERE closed_at IS NOT NULL"
        params: list[Any] = []
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        query += " ORDER BY closed_at DESC"
        with self._connect() as conn:
            return [_trade_from_row(r) for r in conn.execute(query, params).fetchall()]

    def last_losing_exit(self, symbol: str) -> datetime | None:
        """Most recent losing exit for a symbol, for the cooldown gate."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT closed_at, entry_price, exit_price FROM trades
                   WHERE symbol = ? AND closed_at IS NOT NULL
                   ORDER BY closed_at DESC""",
                (symbol,),
            ).fetchall()
        for row in rows:
            if Decimal(row["exit_price"]) < Decimal(row["entry_price"]):
                return datetime.fromisoformat(row["closed_at"])
        return None

    # --------------------------------------------------------------- analysis

    def performance_summary(self, strategy: str | None = None) -> dict[str, Any]:
        """Aggregates the review skill reports on.

        Expectancy is in R rather than dollars because position sizes vary with
        stop width; dollar averages across different risk amounts describe
        nothing in particular.
        """
        trades = self.closed_trades(strategy)
        if not trades:
            return {"trade_count": 0, "note": "no closed trades yet"}

        pnls = [t.realized_pnl for t in trades if t.realized_pnl is not None]
        r_values = [t.realized_r for t in trades if t.realized_r is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        summary: dict[str, Any] = {
            "trade_count": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(pnls), 3) if pnls else None,
            "total_pnl": str(sum(pnls)) if pnls else "0",
            "avg_win": str((sum(wins) / len(wins)).quantize(Decimal("0.01"))) if wins else None,
            "avg_loss": (
                str((sum(losses) / len(losses)).quantize(Decimal("0.01"))) if losses else None
            ),
        }
        if r_values:
            expectancy = sum(r_values) / Decimal(len(r_values))
            summary["expectancy_r"] = str(expectancy.quantize(Decimal("0.01")))
            summary["best_r"] = str(max(r_values))
            summary["worst_r"] = str(min(r_values))

        by_strategy: dict[str, int] = {}
        for trade in trades:
            by_strategy[trade.strategy] = by_strategy.get(trade.strategy, 0) + 1
        summary["by_strategy"] = by_strategy
        return summary

    def rejection_reasons(self, limit: int = 200) -> dict[str, int]:
        """Which gates block the most trades — the fastest read on whether the
        filters are calibrated or merely strict."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT risk_breaches, failed_conditions FROM audit
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            for blob in (row["risk_breaches"], row["failed_conditions"]):
                for reason in json.loads(blob or "[]"):
                    # Collapse to the label before the colon so that varying
                    # numbers in the detail do not fragment the tally.
                    key = str(reason).split(":")[0].strip()
                    counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


# --------------------------------------------------------------------- helpers


def _s(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _d(value: Any) -> Decimal | None:
    return None if value is None else Decimal(str(value))


def _audit_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "cycle_id": row["cycle_id"],
        "occurred_at": row["occurred_at"],
        "symbol": row["symbol"],
        "strategy": row["strategy"],
        "outcome": row["outcome"],
        "reference_price": row["reference_price"],
        "confidence": row["confidence"],
        "signal_strength": row["signal_strength"],
        "reasons": json.loads(row["reasons"] or "[]"),
        "failed_conditions": json.loads(row["failed_conditions"] or "[]"),
        "risk_breaches": json.loads(row["risk_breaches"] or "[]"),
        "critic_notes": json.loads(row["critic_notes"] or "[]"),
    }


def _trade_from_row(row: sqlite3.Row) -> TradeRecord:
    return TradeRecord(
        client_key=row["client_key"],
        symbol=row["symbol"],
        strategy=row["strategy"],
        mode=row["mode"],
        opened_at=datetime.fromisoformat(row["opened_at"]),
        entry_price=Decimal(row["entry_price"]),
        quantity=Decimal(row["quantity"]),
        notional=Decimal(row["notional"]),
        stop_price=_d(row["stop_price"]),
        target_price=_d(row["target_price"]),
        entry_rationale=json.loads(row["entry_rationale"] or "[]"),
        thesis=row["thesis"],
        invalidation_reason=row["invalidation_reason"],
        sector=row["sector"],
        closed_at=datetime.fromisoformat(row["closed_at"]) if row["closed_at"] else None,
        exit_price=_d(row["exit_price"]),
        exit_reason=row["exit_reason"],
    )


__all__ = ["CycleOutcome", "JournalRepository"]
