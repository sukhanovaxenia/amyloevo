"""Taxonomic and phylogenetic structure in APR profiles.

``signal`` is the distribution-free screen: a PERMANOVA per taxonomic rank,
tested inside a paralogue group because paralogue and clade are entangled in any
pooled analysis. It answers whether profiles cluster by clade, which is weaker
than the question §4 of the design asks -- clustering follows from nested
sampling alone -- and is the cheap test that decides whether building the
species tree is worth it.

The tree-based tests the design calls for (Fritz & Purvis D, Pagel's lambda,
phylogenetic logistic regression) are not here yet. ``signal`` is explicit about
standing in for them rather than replacing them.
"""

from .signal import (
    DEFAULT_RANKS,
    MIN_LEVELS,
    MIN_MEMBERS,
    MIN_PER_LEVEL,
    NEGLIGIBLE_WITHIN_SCATTER,
    RankResult,
    SignalResult,
    format_signal_report,
    permanova,
    taxonomic_signal,
)

__all__ = [
    "DEFAULT_RANKS",
    "MIN_LEVELS",
    "MIN_MEMBERS",
    "MIN_PER_LEVEL",
    "NEGLIGIBLE_WITHIN_SCATTER",
    "RankResult",
    "SignalResult",
    "format_signal_report",
    "permanova",
    "taxonomic_signal",
]
