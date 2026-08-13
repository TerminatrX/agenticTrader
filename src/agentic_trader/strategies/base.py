"""Strategy contract and registry.

A strategy answers exactly one question: given a snapshot and whether we
already hold the name, what would you do? It does not size the trade, consult
the account, or know anything about money. That separation is deliberate — it
keeps strategies cheap to test and makes the risk engine the single place where
capital decisions happen.

Two rules bind every implementation:

1. Never emit ENTER on an unknown condition. If an indicator needed to evaluate
   a rule is missing, the rule failed. Trading on absent data is the one
   mistake that a backtest will never warn you about.
2. Populate `reasons` and `failed_conditions` on every signal, including the
   ones that decline to trade. The critic agent and the performance review read
   those lists, and a signal that cannot explain itself cannot be reviewed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from agentic_trader.models import MarketSnapshot, Position, Signal, SignalStrength


@dataclass(frozen=True)
class StrategyContext:
    """Everything a strategy is allowed to know beyond the snapshot itself."""

    position: Position | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def holds_position(self) -> bool:
        return self.position is not None and self.position.quantity > 0

    def param(self, name: str, default: Any) -> Any:
        """Fetch a tuned parameter, coercing to the default's type.

        YAML happily produces a float where a Decimal is wanted; coercing here
        keeps every strategy from re-implementing the same conversion.
        """
        value = self.params.get(name, default)
        if isinstance(default, Decimal) and not isinstance(value, Decimal):
            return Decimal(str(value))
        return value


class Strategy(ABC):
    """Base class for all strategies."""

    name: str = "unnamed"

    @abstractmethod
    def evaluate(self, snapshot: MarketSnapshot, context: StrategyContext) -> Signal:
        """Return this strategy's opinion on the symbol. Must never raise."""

    def _no_signal(self, snapshot: MarketSnapshot, reason: str) -> Signal:
        return Signal(
            symbol=snapshot.symbol,
            strategy=self.name,
            strength=SignalStrength.NONE,
            reference_price=snapshot.reference_price,
            failed_conditions=[reason],
        )


_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator adding a strategy to the registry under its `name`."""
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"Duplicate strategy name: {cls.name}")
    _REGISTRY[cls.name] = cls
    return cls


def get_strategy(name: str) -> Strategy:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "none registered"
        raise KeyError(f"Unknown strategy {name!r}. Known: {known}")
    return _REGISTRY[name]()


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)
