"""Taxonomic and phylogenetic structure in APR profiles.

Three layers, in the order they are meant to be run.

``signal`` is the distribution-free screen: a PERMANOVA per taxonomic rank,
tested inside a paralogue group because paralogue and clade are entangled in any
pooled analysis. It answers whether profiles cluster by clade, which is weaker
than the question §4 of the design asks -- clustering follows from nested
sampling alone -- and is the cheap test that decides whether building the tree
is worth it.

``tree`` is everything between a treefile and a testable tree: rooting on a
declared outgroup *with the monophyly check*, reconciling tips against the trait
table, finding the identical tips whose traits disagree, and checking whether
each paralogue group is monophyletic. The design once called a gene tree "a
later option, not the default"; it is now the choice, and that moves the
duplication inside the tree, which is why the per-clade checks are here.

``traits`` turns block occupancies into the binary trait *D* requires, under one
named threshold, and screens each column for the prevalence that *D* and
``phyloglm`` need before they can say anything.

``estimate`` assembles the call to those estimators and decides what their
p-values mean once there is one per block -- including whether the permutation
count can support the multiplicity adjustment at all.

The estimators themselves -- Fritz & Purvis *D*, Pagel's lambda, phylogenetic
logistic regression -- are in ``workflow/scripts/phylo_signal.R``, called out to
R (``caper``, ``phytools``, ``phylolm``). There is no Python implementation of
*D* worth trusting, and reimplementing one to avoid a dependency would put the
least-tested code in the project at the point where the conclusion is drawn.
"""

from .estimate import (
    LAMBDA_CEILING,
    P_COLUMNS,
    P_UNDERFLOW,
    Q_FOR_P,
    TIE_DIGITS,
    adjust,
    boundary_warnings,
    floor_warning,
    format_phylo_report,
    run_estimators,
)
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
from .traits import (
    MIN_CLADE,
    MIN_MINORITY,
    MIN_PREVALENCE,
    PRESENCE_THRESHOLD,
    WHOLE_TREE,
    TraitCheck,
    binarise,
    continuous,
    covariates,
    format_trait_report,
    screen,
    screen_by_clade,
    tasks,
    usable,
)
from .tree import (
    ALRT_OK,
    MIN_TIP_OCCUPANCY,
    UFBOOT_OK,
    CladeCheck,
    Conflict,
    ConflictReport,
    Reconciliation,
    Rooting,
    SupportSummary,
    clade_checks,
    identical_groups,
    low_occupancy,
    mrca_pair,
    prune_to,
    read_tree,
    reconcile,
    root_on_outgroup,
    root_to_tip,
    support_summary,
    tip_labels,
    tip_occupancy,
    trait_conflicts,
    write_tree,
)

__all__ = [
    # estimate
    "LAMBDA_CEILING",
    "P_COLUMNS",
    "P_UNDERFLOW",
    "Q_FOR_P",
    "TIE_DIGITS",
    "adjust",
    "boundary_warnings",
    "floor_warning",
    "format_phylo_report",
    "run_estimators",
    # signal
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
    # tree
    "ALRT_OK",
    "MIN_TIP_OCCUPANCY",
    "UFBOOT_OK",
    "CladeCheck",
    "Conflict",
    "ConflictReport",
    "Reconciliation",
    "Rooting",
    "SupportSummary",
    "clade_checks",
    "identical_groups",
    "low_occupancy",
    "mrca_pair",
    "prune_to",
    "read_tree",
    "reconcile",
    "root_on_outgroup",
    "root_to_tip",
    "support_summary",
    "tip_labels",
    "tip_occupancy",
    "trait_conflicts",
    "write_tree",
    # traits
    "MIN_CLADE",
    "MIN_MINORITY",
    "MIN_PREVALENCE",
    "PRESENCE_THRESHOLD",
    "WHOLE_TREE",
    "TraitCheck",
    "binarise",
    "continuous",
    "covariates",
    "format_trait_report",
    "screen",
    "screen_by_clade",
    "tasks",
    "usable",
]
