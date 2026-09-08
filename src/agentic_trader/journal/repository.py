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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from agentic_trader.journal.models import AuditEntry, CycleOutcome, TradeRecord
from agentic_trader.models import ProtectionState
from agentic_trader.universe.candidate import funnel_counts

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id        TEXT NOT NULL,
    occurred_at     TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    strategy        TEXT NOT NULL,
    outcome         TEXT NOT NULL,
    -- The execution context, stored rather than inferred. See AuditEntry.mode.
    mode            TEXT,
    -- The acquisition date every ranged request was built from. Distinct from
    -- occurred_at, which is when the cycle ran.
    trading_date    TEXT,
    -- The pinned request contract behind this decision's inputs.
    acquisition_profile_ref TEXT,
    acquisition_config_fingerprint TEXT,
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

-- Resting protective stop orders at the broker. Created now and populated only
-- with what the current build knows, so the stop lifecycle lands later as
-- inserts rather than as a migration against live trading history.
--
-- `state` is a ProtectionState. The three quantity columns are separate on
-- purpose: requested and accepted can differ when a broker partially accepts,
-- and reconciling filled against accepted is how a partially-closed position
-- gets noticed. `last_reconciled_at` separates "believed protected" from
-- "confirmed protected as of a known time" — the question restart recovery
-- asks. `supersedes_protective_order_id` carries cancel-and-replace lineage;
-- no replacement behaviour exists yet, so it stays NULL.
CREATE TABLE IF NOT EXISTS protective_orders (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_client_key    TEXT NOT NULL,
    -- Our own `ref_id` for this stop, derived deterministically from the entry
    -- key plus the stop's price and quantity. Stored because it is the only
    -- identifier that exists *before* the broker replies: a submission that
    -- times out leaves no broker_order_id, and without this column that row
    -- could never be matched to the order it may have created.
    client_key          TEXT,
    broker_order_id     TEXT,
    state               TEXT NOT NULL,
    stop_price          TEXT,
    requested_quantity  TEXT,
    accepted_quantity   TEXT,
    filled_quantity     TEXT,
    capability_profile  TEXT,
    failure_reason      TEXT,
    submitted_at        TEXT,
    accepted_at         TEXT,
    triggered_at        TEXT,
    cancelled_at        TEXT,
    updated_at          TEXT NOT NULL,
    last_reconciled_at  TEXT,
    supersedes_protective_order_id INTEGER REFERENCES protective_orders(id)
);
-- One discovery run, and how much of its declared universe it actually saw.
--
-- `coverage_complete` is computed from row counts against the scanner's hard
-- cap, never from the broker's reported total, which has been observed to
-- disagree with itself. A false value does not stop a shadow run, but analysis
-- must be able to tell "the strategy saw the whole universe" from "the strategy
-- saw a truncated sample" — otherwise a broker limit reads as a lack of setups.
--
-- `scan_config_json` records the filter definition that produced the run, so a
-- run stays interpretable after the filters are retuned. Same reasoning as the
-- versioned capability profile, applied to the scan rather than the endpoint.
CREATE TABLE IF NOT EXISTS scan_runs (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                   TEXT NOT NULL UNIQUE,
    source                   TEXT NOT NULL,
    scanner_profile_ref      TEXT,
    -- The date selection was seeded with, stored because it is an *input*.
    -- Not derivable from started_at: that is a UTC instant, and a run that
    -- starts at 01:30 UTC seeds from the previous trading date. Inferring it
    -- back would be wrong in exactly the cases worth auditing.
    trading_date             TEXT,
    started_at               TEXT NOT NULL,
    completed_at             TEXT,
    shard_count              INTEGER NOT NULL DEFAULT 0,
    returned_before_dedupe   INTEGER NOT NULL DEFAULT 0,
    unique_discovered        INTEGER NOT NULL DEFAULT 0,
    duplicates_removed       INTEGER NOT NULL DEFAULT 0,
    selected_for_enrichment  INTEGER NOT NULL DEFAULT 0,
    budget_deferred_count    INTEGER NOT NULL DEFAULT 0,
    coverage_status          TEXT NOT NULL,
    coverage_complete        INTEGER NOT NULL DEFAULT 0,
    scan_definition_ref      TEXT,
    drift_status             TEXT,
    drift_findings_json      TEXT,
    aborted_reason           TEXT,
    scan_config_fingerprint  TEXT,
    scan_config_json         TEXT,
    coverage_reasons_json    TEXT,
    funnel_counts_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_scan_runs_started ON scan_runs(started_at);

-- Per-shard detail. A run can be incomplete for six different reasons, and a
-- single status on scan_runs cannot answer "why was August 18 incomplete?"
-- months later without reconstructing it from logs that no longer exist.
CREATE TABLE IF NOT EXISTS scan_shards (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             TEXT NOT NULL,
    scan_id            TEXT NOT NULL,
    scan_title         TEXT,
    returned_row_count INTEGER NOT NULL DEFAULT 0,
    candidate_count    INTEGER NOT NULL DEFAULT 0,
    rows_rejected      INTEGER NOT NULL DEFAULT 0,
    reported_total     INTEGER,
    coverage_status    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scan_shards_run ON scan_shards(run_id);

-- Every candidate a run produced, including the ones never examined. Recording
-- only survivors would make the funnel unreadable: the interesting question is
-- usually where candidates stop, not which ones finished.
CREATE TABLE IF NOT EXISTS scan_candidates (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    symbol         TEXT NOT NULL,
    instrument_id  TEXT,
    source         TEXT NOT NULL,
    sector         TEXT,
    stage          TEXT NOT NULL,
    exit_reason    TEXT,
    -- Diagnostic only. Never an input to any decision; kept so a funnel can be
    -- re-examined and so scanner drift is detectable after the fact.
    source_values_json TEXT,
    discovered_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scan_candidates_run ON scan_candidates(run_id);
CREATE INDEX IF NOT EXISTS idx_scan_candidates_symbol ON scan_candidates(symbol);

CREATE INDEX IF NOT EXISTS idx_protective_trade ON protective_orders(trade_client_key);
CREATE INDEX IF NOT EXISTS idx_protective_state ON protective_orders(state);
CREATE UNIQUE INDEX IF NOT EXISTS idx_protective_broker_order
    ON protective_orders(broker_order_id) WHERE broker_order_id IS NOT NULL;
-- Unique for the same reason the broker treats ref_id as an idempotency key:
-- two rows with one client_key would mean we asked for the same stop twice and
-- recorded both as distinct protection.
CREATE UNIQUE INDEX IF NOT EXISTS idx_protective_client_key
    ON protective_orders(client_key) WHERE client_key IS NOT NULL;
"""


# Columns added after the first schema shipped. `CREATE TABLE IF NOT EXISTS`
# leaves an existing table untouched, so a journal created before these fields
# existed would otherwise fail on insert with a confusing "no such column".
_ADDED_COLUMNS: dict[str, dict[str, str]] = {
    "audit": {
        # Nullable for the same reason scan_runs.trading_date is: a row written
        # before the column existed has no honest value, and NOT NULL DEFAULT
        # 'shadow' would retroactively assert something nobody verified. New
        # writes always carry it -- AuditEntry.mode is required.
        "mode": "TEXT",
        "trading_date": "TEXT",
        "acquisition_profile_ref": "TEXT",
        "acquisition_config_fingerprint": "TEXT",
        "original_confidence": "REAL",
        "adjusted_confidence": "REAL",
        "thesis": "TEXT",
        "invalidation_reason": "TEXT",
        "protection_state": "TEXT",
        "capability_profile": "TEXT",
        "market_regime": "TEXT",
        "market_context": "TEXT",
    },
    # scan_runs shipped at 98ee06d with `truncated_candidate_count`. Renaming
    # it in SCHEMA does nothing to an existing database — CREATE TABLE IF NOT
    # EXISTS leaves it alone — so the insert would fail on a missing column.
    # The obsolete column stays behind harmlessly; nothing reads it.
    "scan_runs": {
        # Nullable on purpose. Rows written before this column existed genuinely
        # do not know their trading date, and a NOT NULL DEFAULT would invent
        # one -- stamping every historical run with a date it never used is
        # worse than an honest NULL. New writes always carry it: the repository
        # parameter is required, so the gap cannot spread forward.
        "trading_date": "TEXT",
        "budget_deferred_count": "INTEGER NOT NULL DEFAULT 0",
        "scan_definition_ref": "TEXT",
        "scan_config_fingerprint": "TEXT",
        "coverage_reasons_json": "TEXT",
        "drift_status": "TEXT",
        "drift_findings_json": "TEXT",
        "aborted_reason": "TEXT",
    },
    "trades": {
        "thesis": "TEXT",
        "invalidation_reason": "TEXT",
        "sector": "TEXT",
        "protection_state": "TEXT",
        "capability_profile": "TEXT",
        "market_regime": "TEXT",
        "stop_basis": "TEXT",
    },
    # `protective_orders` shipped empty, ahead of the lifecycle that fills it.
    # Nullable because a row written before this column existed cannot know the
    # key it was submitted under, and inventing one would defeat the dedupe the
    # column exists to provide.
    "protective_orders": {
        "client_key": "TEXT",
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
                    mode, trading_date,
                    acquisition_profile_ref, acquisition_config_fingerprint,
                    reference_price, confidence, signal_strength,
                    original_confidence, adjusted_confidence,
                    thesis, invalidation_reason,
                    protection_state, capability_profile,
                    market_regime, market_context,
                    reasons, failed_conditions, risk_breaches, critic_notes,
                    snapshot_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    entry.cycle_id,
                    entry.occurred_at.isoformat(),
                    entry.symbol,
                    entry.strategy,
                    entry.outcome.value,
                    entry.mode.value,
                    entry.trading_date.isoformat(),
                    entry.acquisition_profile_ref,
                    entry.acquisition_config_fingerprint,
                    _s(entry.reference_price),
                    entry.confidence,
                    entry.signal_strength,
                    entry.original_confidence,
                    entry.adjusted_confidence,
                    entry.thesis,
                    entry.invalidation_reason,
                    entry.protection_state.value if entry.protection_state else None,
                    entry.capability_profile,
                    entry.market_regime,
                    json.dumps(entry.market_context) if entry.market_context else None,
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

    def audit_for_cycle(self, cycle_id: str) -> list[dict[str, Any]]:
        """Full audit rows for one cycle, snapshot included.

        Separate from `recent_audit`, which trims to a summary for display.
        Reconstructing a past decision needs the parts that summary drops: the
        snapshot it was made from, and the contract identities that say what
        the snapshot means.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit WHERE cycle_id = ? ORDER BY id", (cycle_id,)
            ).fetchall()
        return [_full_audit_row(r) for r in rows]

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
                        protection_state, capability_profile,
                        market_regime, stop_basis,
                        closed_at, exit_price, exit_reason
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                        trade.protection_state.value,
                        trade.capability_profile,
                        trade.market_regime,
                        trade.stop_basis,
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

    # --------------------------------------------------- protective stops

    def record_protection_submitted(
        self,
        *,
        trade_client_key: str,
        client_key: str,
        stop_price: Decimal,
        requested_quantity: Decimal,
        capability_profile: str,
        submitted_at: datetime,
    ) -> int:
        """Record that a stop is about to be sent, before sending it.

        Written *before* the broker call, which is the whole point. A
        submission that times out, crashes, or loses its reply may still have
        created a resting order, and a journal that only records stops it saw
        succeed cannot distinguish that from one never sent. Recovery needs the
        difference: re-placing a stop that already rests leaves two sell orders
        on a position that can only be sold once.

        Returns the row id so a caller holding it can resolve the outcome
        without re-deriving the key.
        """
        with self._connect() as conn:
            try:
                cursor = conn.execute(
                    """INSERT INTO protective_orders (
                        trade_client_key, client_key, state, stop_price,
                        requested_quantity, capability_profile,
                        submitted_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?)""",
                    (
                        trade_client_key,
                        client_key,
                        ProtectionState.SUBMITTED.value,
                        _s(stop_price),
                        _s(requested_quantity),
                        capability_profile,
                        submitted_at.isoformat(),
                        submitted_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    f"protective order {client_key} already recorded — this stop "
                    "was already submitted; resolve the existing row rather than "
                    "placing a second one"
                ) from exc
            return int(cursor.lastrowid or 0)

    def record_protection_accepted(
        self,
        *,
        client_key: str,
        broker_order_id: str,
        accepted_quantity: Decimal,
        accepted_at: datetime,
    ) -> None:
        """Promote a submitted stop to PROTECTED, on the broker's evidence alone.

        `broker_order_id` is required and must be non-empty. PROTECTED is the
        only state asserting that a resting order exists, and the order id is
        the only evidence of that which does not originate inside this system.
        Without it the claim would be our own inference wearing the broker's
        authority.

        Promotes only from SUBMITTED. A row that already failed or was
        cancelled describes an order that is not resting, and moving it to
        PROTECTED would manufacture protection out of a stale confirmation.
        """
        if not broker_order_id.strip():
            raise ValueError(
                "broker_order_id is required to mark a stop PROTECTED — without "
                "the broker's own id there is no evidence the order rests"
            )
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE protective_orders
                      SET state = ?, broker_order_id = ?, accepted_quantity = ?,
                          accepted_at = ?, updated_at = ?, last_reconciled_at = ?
                    WHERE client_key = ? AND state = ?""",
                (
                    ProtectionState.PROTECTED.value,
                    broker_order_id,
                    _s(accepted_quantity),
                    accepted_at.isoformat(),
                    accepted_at.isoformat(),
                    accepted_at.isoformat(),
                    client_key,
                    ProtectionState.SUBMITTED.value,
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError(
                    f"no submitted protective order with client_key {client_key} — "
                    "it was never recorded as submitted, or has already been resolved"
                )

    def record_protection_failed(
        self, *, client_key: str, reason: str, failed_at: datetime
    ) -> None:
        """Record that the broker refused the stop. An incident, not a property.

        Distinct from UNAVAILABLE, which says the position's shape can never be
        protected. FAILED says this attempt was rejected, so the position is
        open and uncovered right now and somebody has to decide what to do.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE protective_orders
                      SET state = ?, failure_reason = ?, updated_at = ?
                    WHERE client_key = ? AND state = ?""",
                (
                    ProtectionState.FAILED.value,
                    reason,
                    failed_at.isoformat(),
                    client_key,
                    ProtectionState.SUBMITTED.value,
                ),
            )
            if cursor.rowcount == 0:
                raise ValueError(
                    f"no submitted protective order with client_key {client_key}"
                )

    def unconfirmed_protection(self) -> list[dict[str, Any]]:
        """Stops that were sent and never resolved. The first query a restart runs.

        Every row here is a position that may or may not have a resting stop at
        the broker. The answer cannot be derived locally — it requires reading
        the broker's open orders — and until it is read, neither placing a new
        stop nor assuming the old one holds is safe.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM protective_orders WHERE state = ? ORDER BY submitted_at",
                (ProtectionState.SUBMITTED.value,),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_trade_protection(self, client_key: str, state: ProtectionState) -> None:
        """Bring the trade row's protection state in line with the stop's.

        `protective_orders` is authoritative for the guard, which reads it
        directly. This exists so the *trade* record does not keep asserting
        COMMITTED for a position whose stop was confirmed hours earlier — the
        performance review reads trades, and a review that misreports whether
        stops were ever real is worse than one that says nothing.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE trades SET protection_state = ? WHERE client_key = ?",
                (state.value, client_key),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"no trade with client_key {client_key}")

    def unprotected_live_positions(self) -> list[str]:
        """Open live positions with no confirmed resting stop. The entry blocker.

        A live entry necessarily opens a window between the fill and its stop,
        because protection cannot be established before there is a position to
        protect. This query is how that window is proven closed: anything still
        listed here is a real position carrying real risk that nothing is
        watching between cycles.

        Deliberately keyed on the *absence* of a PROTECTED row rather than on
        any state written at entry time. SUBMITTED, FAILED, a missing row, a
        stop that was cancelled and never replaced — every one of them means
        unprotected, and enumerating the bad states would leave the next one
        added silently safe.

        Shadow is excluded because a shadow position is a record, not risk.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT t.client_key FROM trades t
                    WHERE t.closed_at IS NULL
                      AND t.mode = 'live'
                      AND NOT EXISTS (
                          SELECT 1 FROM protective_orders p
                           WHERE p.trade_client_key = t.client_key
                             AND p.state = ?
                      )
                    ORDER BY t.opened_at""",
                (ProtectionState.PROTECTED.value,),
            ).fetchall()
        return [r["client_key"] for r in rows]

    def protective_orders_for(self, trade_client_key: str) -> list[dict[str, Any]]:
        """Every protective order ever raised for one entry, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM protective_orders
                    WHERE trade_client_key = ? ORDER BY id""",
                (trade_client_key,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ----------------------------------------------------------- discovery

    def record_scan_run(
        self,
        run_id: str,
        batch: Any | None,
        *,
        trading_date: date,
        candidates: Sequence[Any] | None = None,
        scanner_profile_ref: str | None = None,
        scan_definition_ref: str | None = None,
        scan_config_fingerprint: str | None = None,
        scan_config: dict[str, Any] | None = None,
        drift_status: str | None = None,
        drift_findings: list[dict[str, Any]] | None = None,
        aborted_reason: str | None = None,
        coverage_status: str | None = None,
        source: str | None = None,
        started_at: datetime | None = None,
        selected_count: int = 0,
        budget_deferred_count: int = 0,
        completed_at: datetime | None = None,
    ) -> None:
        """Persist a discovery run and every candidate it produced.

        Candidates are written whatever stage they reached, deferred ones
        included. A funnel that recorded only survivors could not distinguish
        "the strategy declined it" from "we never looked", and those two answer
        completely different questions about why a day produced no trades.

        `trading_date` is required and has no default. It is the seed selection
        rotated on, so a run that cannot state it cannot be reproduced -- and a
        default here would let a caller silently fall back to the wall clock,
        which is the exact failure this parameter exists to prevent.
        """
        rows = list(
            candidates
            if candidates is not None
            else (batch.candidates if batch is not None else [])
        )
        shards = list(batch.shards) if batch is not None else []
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO scan_runs (
                    run_id, source, scanner_profile_ref, trading_date,
                    started_at, completed_at,
                    shard_count, returned_before_dedupe, unique_discovered,
                    duplicates_removed, selected_for_enrichment,
                    budget_deferred_count, coverage_status, coverage_complete,
                    scan_definition_ref, scan_config_fingerprint,
                    scan_config_json, coverage_reasons_json, funnel_counts_json,
                    drift_status, drift_findings_json, aborted_reason
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    source or (batch.source if batch is not None else "unknown"),
                    scanner_profile_ref,
                    trading_date.isoformat(),
                    (started_at or (batch.started_at if batch is not None else datetime.now(UTC)))
                    .isoformat(),
                    (completed_at or datetime.now(UTC)).isoformat(),
                    len(shards),
                    batch.returned_before_dedupe if batch is not None else 0,
                    len(batch.candidates) if batch is not None else 0,
                    batch.duplicates_removed if batch is not None else 0,
                    selected_count,
                    budget_deferred_count,
                    coverage_status
                    or (batch.coverage.value if batch is not None else "incomplete"),
                    int(batch.coverage_complete) if batch is not None else 0,
                    scan_definition_ref,
                    scan_config_fingerprint,
                    json.dumps(scan_config) if scan_config else None,
                    json.dumps(batch.report.reasons() if batch is not None else []),
                    json.dumps(funnel_counts(rows)),
                    drift_status,
                    json.dumps(drift_findings) if drift_findings else None,
                    aborted_reason,
                ),
            )
            conn.executemany(
                """INSERT INTO scan_candidates (
                    run_id, symbol, instrument_id, source, sector, stage,
                    exit_reason, source_values_json, discovered_at
                ) VALUES (?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        run_id,
                        c.symbol,
                        c.instrument_id,
                        c.source,
                        c.sector,
                        c.stage.value,
                        c.exit_reason,
                        json.dumps(c.source_values) if c.source_values else None,
                        c.discovered_at.isoformat(),
                    )
                    for c in rows
                ],
            )

            conn.executemany(
                """INSERT INTO scan_shards (
                    run_id, scan_id, scan_title, returned_row_count,
                    candidate_count, rows_rejected, reported_total, coverage_status
                ) VALUES (?,?,?,?,?,?,?,?)""",
                [
                    (
                        run_id,
                        sh.scan_id,
                        sh.scan_title,
                        sh.returned_row_count,
                        sh.candidate_count,
                        sh.rows_rejected,
                        sh.reported_total,
                        sh.coverage.value,
                    )
                    for sh in shards
                ],
            )

    def scan_shards(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scan_shards WHERE run_id = ?", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def scan_candidates(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scan_candidates WHERE run_id = ?", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def scan_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scan_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

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
        "mode": row["mode"],
        "trading_date": row["trading_date"],
        "acquisition_profile_ref": row["acquisition_profile_ref"],
        "acquisition_config_fingerprint": row["acquisition_config_fingerprint"],
        "reference_price": row["reference_price"],
        "confidence": row["confidence"],
        "signal_strength": row["signal_strength"],
        "protection_state": row["protection_state"],
        "capability_profile": row["capability_profile"],
        "market_regime": row["market_regime"],
        "reasons": json.loads(row["reasons"] or "[]"),
        "failed_conditions": json.loads(row["failed_conditions"] or "[]"),
        "risk_breaches": json.loads(row["risk_breaches"] or "[]"),
        "critic_notes": json.loads(row["critic_notes"] or "[]"),
    }


def _full_audit_row(row: sqlite3.Row) -> dict[str, Any]:
    """Every persisted field, decoded. The reconstruction path."""
    return {
        **_audit_row(row),
        "original_confidence": row["original_confidence"],
        "adjusted_confidence": row["adjusted_confidence"],
        "thesis": row["thesis"],
        "invalidation_reason": row["invalidation_reason"],
        "market_context": json.loads(row["market_context"] or "null"),
        "snapshot_json": json.loads(row["snapshot_json"] or "null"),
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
        # Rows written before this column existed read as NULL. They predate
        # protection tracking entirely, so NOT_REQUIRED is the honest default —
        # not a claim they were protected.
        protection_state=ProtectionState(row["protection_state"] or ProtectionState.NOT_REQUIRED),
        capability_profile=row["capability_profile"],
        market_regime=row["market_regime"],
        stop_basis=row["stop_basis"],
        closed_at=datetime.fromisoformat(row["closed_at"]) if row["closed_at"] else None,
        exit_price=_d(row["exit_price"]),
        exit_reason=row["exit_reason"],
    )


__all__ = ["CycleOutcome", "JournalRepository"]
