"""Schema-validated config loading.

Risk configuration is the one file in this project that must never fail open.
An unvalidated YAML typo — `max_position_pct: 0.9` where `0.09` was meant, or a
key silently ignored because it was misspelled — reaches the market as real
money. Every field is therefore bounded, unknown keys are rejected outright,
and a malformed file raises at startup rather than defaulting to something
permissive.
"""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(RuntimeError):
    """Raised when configuration is missing, malformed, or out of bounds."""


class RiskConfig(BaseModel):
    """Hard bounds on system behaviour.

    Defaults here are deliberately conservative: if a key is absent from the
    YAML, the resulting behaviour should be *less* permissive than a human
    would have chosen, never more.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- Sizing -----------------------------------------------------------
    risk_per_trade_pct: Decimal = Field(
        default=Decimal("0.01"), gt=0, le=Decimal("0.05"),
        description="Fraction of account value risked between entry and stop.",
    )
    max_position_pct: Decimal = Field(
        default=Decimal("0.25"), gt=0, le=Decimal("1.0"),
        description="Ceiling on any single position as a fraction of account value.",
    )
    min_order_notional: Decimal = Field(
        default=Decimal("1.00"), gt=0,
        description="Orders below this are not worth the spread; drop them.",
    )
    max_order_notional: Decimal | None = Field(
        default=None, gt=0,
        description="Absolute dollar ceiling per order, independent of account size.",
    )

    # --- Exposure ---------------------------------------------------------
    max_open_positions: int = Field(default=3, ge=1, le=50)
    max_daily_loss_pct: Decimal = Field(
        default=Decimal("0.03"), gt=0, le=Decimal("0.25"),
        description="Realized loss for the day that halts all new entries.",
    )
    max_portfolio_exposure_pct: Decimal = Field(
        default=Decimal("0.80"), gt=0, le=Decimal("1.0"),
    )
    max_sector_exposure_pct: Decimal = Field(
        default=Decimal("0.25"), gt=0, le=Decimal("1.0"),
        description=(
            "Ceiling on combined exposure to any one sector. Guards the case "
            "where several 'independent' positions are the same bet."
        ),
    )

    # --- Reward ----------------------------------------------------------
    min_risk_reward: Decimal = Field(
        default=Decimal("2.0"), gt=0, le=Decimal("20"),
        description=(
            "Minimum reward-to-risk from entry to target. Note that a strategy "
            "deriving its target as a fixed R multiple always satisfies this by "
            "construction; the gate binds for strategies whose targets come "
            "from structure."
        ),
    )

    # --- Kill switch (sticky) --------------------------------------------
    kill_switch_daily_loss_pct: Decimal = Field(
        default=Decimal("0.06"), gt=0, le=Decimal("0.50"),
        description=(
            "Realized daily loss that writes HALT and stops the system until a "
            "human clears it. Distinct in kind from max_daily_loss_pct, which "
            "blocks new entries and resets tomorrow."
        ),
    )

    # --- Stops ------------------------------------------------------------
    default_stop_pct: Decimal = Field(default=Decimal("0.05"), gt=0, le=Decimal("0.5"))
    max_stop_pct: Decimal = Field(
        default=Decimal("0.12"), gt=0, le=Decimal("0.5"),
        description="A stop wider than this means the setup is too loose to size.",
    )

    # --- Timing gates -----------------------------------------------------
    earnings_blackout_days: int = Field(
        default=3, ge=0, le=30,
        description="Block new entries within N calendar days before earnings.",
    )
    symbol_cooldown_days: int = Field(
        default=2, ge=0, le=30,
        description="Days to wait before re-entering a symbol after a losing exit.",
    )

    # --- Cash-account settlement -----------------------------------------
    respect_unsettled_funds: bool = Field(
        default=True,
        description=(
            "On a cash account, spending unsettled sale proceeds causes a "
            "good-faith violation. Leave this on unless the account has margin."
        ),
    )

    # --- Liquidity and execution quality ----------------------------------
    min_avg_volume_30d: Decimal = Field(default=Decimal("500000"), ge=0)
    max_spread_pct: Decimal = Field(
        default=Decimal("0.005"), gt=0, le=Decimal("0.1"),
        description=(
            "Widest tolerated bid/ask spread, as a fraction of the mid. Entries "
            "are market orders by broker constraint, so the spread is paid in "
            "full on every fill."
        ),
    )
    max_price_drift_pct: Decimal = Field(
        default=Decimal("0.005"), gt=0, le=Decimal("0.1"),
        description=(
            "How far the live price may move from the price the decision was "
            "made at before the setup must be re-evaluated rather than chased. "
            "Distinct from max_spread_pct: that is the cost of crossing the "
            "book now, this is the staleness of the thesis."
        ),
    )

    # --- Protection -------------------------------------------------------
    allow_unprotected_shadow_entries: bool = Field(
        default=True,
        description=(
            "Whether SHADOW mode may open a position the broker cannot cover "
            "with a resting stop. Scoped to shadow by name and by design: live "
            "and approval execution refuse an unprotected entry structurally, "
            "and no configuration value can override that. This key can only "
            "make shadow stricter, never make live permissive."
        ),
    )

    # --- Kill switch ------------------------------------------------------
    halt_file: str = Field(
        default="HALT",
        description="If this file exists at the project root, no orders are permitted.",
    )

    @model_validator(mode="after")
    def _check_internal_consistency(self) -> RiskConfig:
        if self.default_stop_pct > self.max_stop_pct:
            raise ValueError(
                f"default_stop_pct ({self.default_stop_pct}) exceeds "
                f"max_stop_pct ({self.max_stop_pct})"
            )
        if (
            self.max_order_notional is not None
            and self.max_order_notional < self.min_order_notional
        ):
            raise ValueError("max_order_notional is below min_order_notional")
        # Sizing that can never fill the position ceiling is a configuration
        # mistake worth surfacing loudly rather than silently under-trading.
        if self.risk_per_trade_pct >= self.max_position_pct:
            raise ValueError(
                "risk_per_trade_pct must be well below max_position_pct; "
                "otherwise every trade is capped by the position ceiling"
            )
        # A kill switch that fires before the soft limit makes the soft limit
        # dead code — the system would halt outright where it should merely
        # have stopped opening new positions.
        if self.kill_switch_daily_loss_pct <= self.max_daily_loss_pct:
            raise ValueError(
                f"kill_switch_daily_loss_pct ({self.kill_switch_daily_loss_pct}) must "
                f"exceed max_daily_loss_pct ({self.max_daily_loss_pct}); otherwise the "
                "sticky halt fires first and the daily limit never applies"
            )
        # A per-position ceiling above the sector ceiling cannot be reached
        # whenever the sector is known, which makes sizing behaviour depend on
        # whether fundamentals happened to load. Surface it as a config error.
        if self.max_position_pct > self.max_sector_exposure_pct:
            raise ValueError(
                f"max_position_pct ({self.max_position_pct}) exceeds "
                f"max_sector_exposure_pct ({self.max_sector_exposure_pct}); a single "
                "position could never reach its ceiling once its sector is known"
            )
        return self


class StrategyEntry(BaseModel):
    """Per-strategy switch and parameter bag."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    universe: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategies: dict[str, StrategyEntry] = Field(default_factory=dict)

    def enabled_names(self) -> list[str]:
        return sorted(n for n, s in self.strategies.items() if s.enabled)


class AccountConfig(BaseModel):
    """Broker account identity, loaded from a gitignored local file.

    Deliberately optional. The authoritative source for which account may be
    traded is the `get_accounts` MCP tool at runtime — a config file cannot
    know whether `agentic_allowed` is still true. This exists to save a lookup
    and record intent, never to grant permission.

    Account numbers are identifying information and permanent once committed,
    which is why they live here rather than in tracked files.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    account_number: str = Field(min_length=4)
    is_cash_account: bool = True
    nickname: str | None = None

    @property
    def masked(self) -> str:
        """All but the last four digits hidden. Use this in anything a human
        might read, copy, or paste — logs, prompts, error messages."""
        return f"****{self.account_number[-4:]}"


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    risk: RiskConfig
    strategies: StrategyConfig
    project_root: Path
    account: AccountConfig | None = None

    @property
    def halt_path(self) -> Path:
        return self.project_root / self.risk.halt_file

    def is_halted(self) -> bool:
        return self.halt_path.exists()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return raw


def risk_fingerprint(config: RiskConfig) -> str:
    """SHA-256 over the *effective* risk values.

    Hashes the validated model dump rather than the file bytes, so comments,
    key order, and CRLF/LF differences are all invisible, while any change to a
    value that actually governs behaviour changes the hash.
    """
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_risk_lock(config: RiskConfig, path: Path) -> str:
    """Record the current risk values as the approved baseline.

    Stores the values alongside the hash so a later mismatch can name exactly
    which limit moved instead of merely reporting that something did.
    """
    fingerprint = risk_fingerprint(config)
    path.write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "fingerprint": fingerprint,
                "values": config.model_dump(mode="json"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return fingerprint


def _describe_risk_drift(locked: dict[str, Any], current: dict[str, Any]) -> list[str]:
    changes: list[str] = []
    for key in sorted(set(locked) | set(current)):
        before, after = locked.get(key, "<absent>"), current.get(key, "<absent>")
        if before != after:
            changes.append(f"  {key}: {before} -> {after}")
    return changes


def verify_risk_lock(config: RiskConfig, lock_path: Path) -> None:
    """Raise unless the risk config matches its recorded baseline.

    This is the half of risk-config protection that survives outside the agent
    harness — a deny hook constrains this agent, while the lock catches any
    edit from any source. Loosening a limit becomes a deliberate two-step act
    (edit, then re-lock) with both steps visible in git history.

    A missing lock file is not an error: it means the baseline has not been
    established yet. Only a *mismatch* is fatal.
    """
    if not lock_path.exists():
        return

    try:
        locked = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Risk lock at {lock_path} is unreadable: {exc}") from exc

    current = risk_fingerprint(config)
    if locked.get("fingerprint") == current:
        return

    drift = _describe_risk_drift(locked.get("values", {}), config.model_dump(mode="json"))
    detail = "\n".join(drift) if drift else "  (no value differences; lock file may be stale)"
    raise ConfigError(
        f"config/risk.yaml does not match its approved baseline ({lock_path.name}).\n"
        f"Changed:\n{detail}\n\n"
        "Risk limits must not change silently. Review the diff, and if the change "
        "is intended, re-lock deliberately:\n"
        "    python -m agentic_trader.cli lock-risk --confirm\n"
        "The trading agent must never run that command."
    )


def load_risk_config(path: Path, lock_path: Path | None = None) -> RiskConfig:
    try:
        config = RiskConfig(**_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"Invalid risk config at {path}:\n{exc}") from exc

    if lock_path is not None:
        verify_risk_lock(config, lock_path)
    return config


def load_strategy_config(path: Path) -> StrategyConfig:
    try:
        return StrategyConfig(**_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"Invalid strategy config at {path}:\n{exc}") from exc


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward for the directory holding `pyproject.toml`."""
    current = (start or Path(__file__)).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise ConfigError("Could not locate project root (no pyproject.toml found)")


def load_account_config(path: Path) -> AccountConfig | None:
    """Load local account identity, or None when the file is absent.

    Absence is normal, not an error — the agent resolves the account through
    `get_accounts` at runtime. A file that exists but is malformed *is* an
    error, since that means someone intended to configure it and it silently
    would not apply.
    """
    if not path.exists():
        return None
    raw = _read_yaml(path)
    if str(raw.get("account_number", "")).strip().upper() == "REPLACE_ME":
        raise ConfigError(
            f"{path} still holds the template placeholder. Fill in a real "
            "account number or delete the file."
        )
    try:
        return AccountConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid account config at {path}:\n{exc}") from exc


def load_config(project_root: Path | None = None) -> AppConfig:
    root = project_root or find_project_root()
    config_dir = root / "config"
    return AppConfig(
        risk=load_risk_config(config_dir / "risk.yaml", config_dir / "risk.lock"),
        strategies=load_strategy_config(config_dir / "strategies.yaml"),
        project_root=root,
        account=load_account_config(config_dir / "account.local.yaml"),
    )
