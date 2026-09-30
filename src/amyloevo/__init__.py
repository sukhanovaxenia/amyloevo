"""amyloevo -- evolutionary conservatism of amyloidogenic regions.

The question this package exists to answer is not whether APR positions are
conserved, but whether a position stays amyloidogenic while its residues turn
over. That is a contrast between two measures, so both are made and the residual
between them is the result. See docs/DESIGN.md.

Imports are kept light: the modules with no external dependency (`project`,
`conserve`, `io`) are importable and testable without an aligner, a tree
program, or the predictor panel installed.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
