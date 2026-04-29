"""models.py — Shared data structures for the entire system.

All monetary values are floats in dollar units (0.88 = 88¢).
All probabilities are floats in [0, 1] (0.12 = 12%).

Reference: insurance_writing_guide.md, Section 1.1
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime


class MarketStatus(enum.Enum):
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"

    @classmethod
    def _missing_(cls, value):
        if value == "active":
            return cls.OPEN
        if value == "determined":
            return cls.SETTLED
        return None


class MarketCategory(enum.Enum):
    """Tier classification from insurance_writing_guide.md, Section 2.1."""
    SPORTS_MENTIONS = "sports_mentions"       # Tier 1
    ENTERTAINMENT = "entertainment"           # Tier 1
    AWARDS = "awards"                         # Tier 1
    SPORTS_PROPS = "sports_props"             # Tier 1
    POLITICAL_MENTIONS = "political_mentions" # Tier 2
    SPORTS_OUTCOMES = "sports_outcomes"       # Tier 2
    EARNINGS_MENTIONS = "earnings_mentions"   # Tier 3
    ECONOMIC_DATA = "economic_data"           # Tier 3
    FINANCE = "finance"                       # Tier 3
    UNKNOWN = "unknown"


TIER_1_CATEGORIES = {
    MarketCategory.SPORTS_MENTIONS,
    MarketCategory.ENTERTAINMENT,
    MarketCategory.AWARDS,
    MarketCategory.SPORTS_PROPS,
}

TIER_2_CATEGORIES = {
    MarketCategory.POLITICAL_MENTIONS,
    MarketCategory.SPORTS_OUTCOMES,
}


class Side(enum.Enum):
    YES = "yes"
    NO = "no"


class Role(enum.Enum):
    MAKER = "maker"
    TAKER = "taker"


class Phase(enum.Enum):
    A_PRE_EVENT = "A"
    B_MID_EVENT = "B"
    C_CAPITAL_RECYCLE = "C"


# ---------------------------------------------------------------------------
# Category classifier
# ---------------------------------------------------------------------------

SERIES_TO_CATEGORY: dict[str, MarketCategory] = {
    "KXNFL": MarketCategory.SPORTS_MENTIONS,
    "KXNBA": MarketCategory.SPORTS_MENTIONS,
    "KXMLB": MarketCategory.SPORTS_MENTIONS,
    "KXNHL": MarketCategory.SPORTS_MENTIONS,
    "KXOSC": MarketCategory.AWARDS,
    "KXGRAM": MarketCategory.AWARDS,
    "KXEMMY": MarketCategory.AWARDS,
    "KXSOTU": MarketCategory.POLITICAL_MENTIONS,
    "KXEARN": MarketCategory.EARNINGS_MENTIONS,
    "KXFED": MarketCategory.ECONOMIC_DATA,
    "KXCPI": MarketCategory.ECONOMIC_DATA,
    "KXBTC": MarketCategory.FINANCE,
    "KXSPX": MarketCategory.FINANCE,
    "KXMRB": MarketCategory.ENTERTAINMENT,
    "KXYOU": MarketCategory.ENTERTAINMENT,
}


def classify_market(series_ticker: str) -> MarketCategory:
    """Classify a market by its series ticker prefix (longest-prefix match)."""
    for prefix, category in sorted(
        SERIES_TO_CATEGORY.items(), key=lambda x: -len(x[0])
    ):
        if series_ticker.startswith(prefix):
            return category
    return MarketCategory.UNKNOWN


@dataclass(frozen=True)
class Market:
    """A single Kalshi binary contract."""
    ticker: str
    title: str
    subtitle: str
    yes_bid: float
    yes_ask: float
    volume: int
    open_interest: int
    status: MarketStatus
    close_time: datetime
    result: str | None
    category: MarketCategory = MarketCategory.UNKNOWN
    event_ticker: str = ""
    series_ticker: str = ""

    @property
    def yes_mid(self) -> float:
        return (self.yes_bid + self.yes_ask) / 2

    @property
    def no_mid(self) -> float:
        return 1.0 - self.yes_mid

    @property
    def implied_yes_prob(self) -> float:
        return self.yes_mid

    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid


@dataclass
class BaseRateEstimate:
    """Output of any base-rate provider."""
    source: str
    true_yes_prob: float
    sample_size: int
    confidence: float
    notes: str = ""


@dataclass(frozen=True)
class EdgeResult:
    """Output of the edge calculator."""
    market: Market
    base_rate: BaseRateEstimate
    side: Side
    entry_price: float
    edge_pp: float
    ev_per_contract: float
    raw_kelly_fraction: float
    quarter_kelly_fraction: float
    taker_fee: float
    maker_fee: float
    position_size_contracts: int
    position_size_dollars: float
    passes_filters: bool
    rejection_reason: str = ""


@dataclass
class PortfolioState:
    """Current portfolio snapshot for position-limit enforcement."""
    bankroll: float
    cash: float
    positions: dict[str, float] = field(default_factory=dict)
    category_exposure: dict[str, float] = field(default_factory=dict)
    event_exposure: dict[str, float] = field(default_factory=dict)

    @property
    def total_deployed(self) -> float:
        return sum(self.positions.values())

    @property
    def utilization(self) -> float:
        return self.total_deployed / self.bankroll if self.bankroll > 0 else 0.0
