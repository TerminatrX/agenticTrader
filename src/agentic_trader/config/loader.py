"""Schema-validated config loading.

Risk configuration is the one file in this project that must never fail open.
An unvalidated YAML typo — `max_position_pct: 0.9` where `0.09` was meant, or a
key silently ignored because it was misspelled — reaches the market as real
money. Every field is therefore bounded, unknown keys are rejected outright,
and a malformed file raises at startup rather than defaulting to something
permissive.
"""

from __future__ import annotations

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

    # --- Liquidity --------------------------------------------------------
    min_avg_volume_30d: Decimal = Field(default=Decimal("500000"), ge=0)
    max_spread_pct: Decimal = Field(default=Decimal("0.005"), gt=0, le=Decimal("0.1"))

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


def load_risk_config(path: Path) -> RiskConfig:
    try:
        return RiskConfig(**_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"Invalid risk config at {path}:\n{exc}") from exc


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
        risk=load_risk_config(config_dir / "risk.yaml"),
        strategies=load_strategy_config(config_dir / "strategies.yaml"),
        project_root=root,
        account=load_account_config(config_dir / "account.local.yaml"),
    )
