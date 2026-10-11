"""Statistical helpers shared across the package.

``multiple`` holds the multiple-comparison adjustments. They live here rather
than in the module that happens to need them first, because the same family of
tests is built in several places -- one test per protein, one per taxonomic
rank, one per domain -- and an adjustment implemented twice is an adjustment
applied inconsistently.
"""

from .multiple import benjamini_hochberg, holm, n_discoveries

__all__ = ["benjamini_hochberg", "holm", "n_discoveries"]
