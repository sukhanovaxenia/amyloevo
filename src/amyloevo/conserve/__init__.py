"""The conservation measures, and the features they are measured over.

``scores`` holds the per-column measures of section 3 of the design: sequence
conservation (entropy, Jensen-Shannon divergence against a real background),
APR conservation, and the residual between them that is the headline metric.

``blocks`` holds the feature construction that comes first. A per-column binary
profile over a long alignment is sparse, and clustering or testing it directly
measures mostly shared absence; columns are grouped into the blocks the set
actually uses and each sequence scored by the fraction of each block it covers.
Features matter more than the test applied to them.

Neither module touches an alignment directly -- both consume `project` -- and
neither has an external dependency beyond numpy, so both are testable with
nothing installed.
"""

from .blocks import (
    DEFINE_MAX_GAP,
    DEFINE_MIN_COVERAGE,
    DEFINE_MIN_GROUP,
    DEFINE_MIN_OCCUPANCY,
    DEFINE_MIN_WIDTH,
    Block,
    block_profiles,
    blocks_from_occupancy,
    define_blocks,
    group_occupancy,
    merge_intervals,
    occupancy,
)
from .scores import (
    ColumnConservation,
    aa_type,
    apr_fraction,
    apr_residual,
    jensen_shannon_divergence,
    measure_column,
    shannon_entropy,
)

__all__ = [
    "Block",
    "ColumnConservation",
    "DEFINE_MAX_GAP",
    "DEFINE_MIN_COVERAGE",
    "DEFINE_MIN_GROUP",
    "DEFINE_MIN_OCCUPANCY",
    "DEFINE_MIN_WIDTH",
    "aa_type",
    "apr_fraction",
    "apr_residual",
    "block_profiles",
    "blocks_from_occupancy",
    "define_blocks",
    "group_occupancy",
    "jensen_shannon_divergence",
    "measure_column",
    "merge_intervals",
    "occupancy",
    "shannon_entropy",
]
