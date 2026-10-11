"""The coordinate spine: residue numbers and alignment columns, both ways.

``columns`` holds the mapping classes for per-residue questions and the array
projections for whole-alignment work. Both are built from one rule, and the
0-based/1-based boundary is crossed in exactly one place -- which is the only
reason an off-by-one here is findable rather than endemic.
"""

from .columns import (
    GAP_CHARACTERS,
    AlignmentIndex,
    SequenceMapping,
    build_mapping,
    column_to_residue,
    index_alignment,
    is_gap,
    mask_from_regions,
    project_mask,
    residue_to_column,
)

__all__ = [
    "AlignmentIndex",
    "GAP_CHARACTERS",
    "SequenceMapping",
    "build_mapping",
    "column_to_residue",
    "index_alignment",
    "is_gap",
    "mask_from_regions",
    "project_mask",
    "residue_to_column",
]
