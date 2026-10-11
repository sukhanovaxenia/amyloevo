"""Reading orthologue sets, and naming what is in them.

``headers`` parses the per-protein multifasta an orthologue set arrives as.
``taxonomy`` turns the taxids in those headers into names and lineages, from an
NCBI taxdump or from a table the caller already has. ``annotate`` joins the two
into the metadata table the rest of the package and every figure label reads
from.

Nothing here calls a predictor or an aligner, so it is importable and testable
with nothing installed.
"""

from .annotate import (
    DEFAULT_LABEL_TEMPLATE,
    LABEL_FALLBACKS,
    AnnotationResult,
    annotate,
    label_map,
    make_labels,
    render,
)
from .headers import (
    HeaderParseError,
    OrthologHeader,
    detect_format,
    parse_header,
    parse_headers,
)
from .taxonomy import STANDARD_RANKS, Taxon, TaxonomyDB, load_taxonomy

__all__ = [
    "AnnotationResult",
    "DEFAULT_LABEL_TEMPLATE",
    "HeaderParseError",
    "LABEL_FALLBACKS",
    "OrthologHeader",
    "STANDARD_RANKS",
    "Taxon",
    "TaxonomyDB",
    "annotate",
    "detect_format",
    "label_map",
    "load_taxonomy",
    "make_labels",
    "parse_header",
    "parse_headers",
    "render",
]
