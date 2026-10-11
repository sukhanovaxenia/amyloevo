"""Map residue numbers onto alignment columns, and back.

This is the coordinate spine of the package: every conservation measure is
computed per alignment column, every prediction is made per residue of an
individual sequence, and nothing else in amyloevo is allowed to convert between
the two by hand.

Three facts make the conversion worth isolating rather than inlining.

**It is not a shift.** A gap in one sequence displaces every residue after it in
that sequence alone, so the offset between residue number and column is a
function of position and differs per species. Code that "adds the gap count"
once, at the top of a region, mislocates the end of the region.

**It is partial in both directions.** A column may hold a gap for a given
sequence (no residue number), and every residue has a column, but only within
its own alignment. Returning 0 or -1 for the missing case invites arithmetic on
a sentinel; these functions return ``None`` and the callers must decide.

**The numbering is 1-based on both axes**, matching the predictors' output and
every alignment viewer a biologist will check the result against. The internal
list indices are 0-based and that boundary is crossed here, once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

GAP_CHARACTERS = frozenset("-.~")


def is_gap(character: str) -> bool:
    return character in GAP_CHARACTERS


@dataclass(frozen=True)
class SequenceMapping:
    """The residue/column correspondence for ONE aligned sequence.

    ``column_of`` and ``residue_at`` are built once and then read, because the
    conservation pass walks every column of every species and rebuilding the
    offset each time turns an O(L) analysis into O(L^2).
    """

    sequence_id: str
    aligned: str
    #: 1-based residue number -> 1-based alignment column
    _residue_to_column: dict[int, int] = field(repr=False)
    #: 1-based alignment column -> 1-based residue number (absent where gapped)
    _column_to_residue: dict[int, int] = field(repr=False)

    @property
    def ungapped_length(self) -> int:
        return len(self._residue_to_column)

    @property
    def alignment_length(self) -> int:
        return len(self.aligned)

    def column_of(self, residue_number: int) -> int | None:
        """1-based column holding this 1-based residue, or None if out of range."""
        return self._residue_to_column.get(residue_number)

    def residue_at(self, column: int) -> int | None:
        """1-based residue number at this 1-based column, or None where gapped."""
        return self._column_to_residue.get(column)

    def residue_character(self, column: int) -> str:
        """The aligned character at a 1-based column, gap included."""
        if not 1 <= column <= len(self.aligned):
            raise IndexError(
                f"{self.sequence_id}: column {column} outside 1..{len(self.aligned)}"
            )
        return self.aligned[column - 1]

    def columns_for_region(self, start: int, end: int) -> list[int]:
        """Columns spanned by an inclusive 1-based residue region.

        Returns the columns of the region's own residues only -- gap columns
        that fall between them are NOT included. An APR is a set of residues; a
        gap in this sequence is a residue another sequence has and this one does
        not, so counting it here would inflate the region by another species'
        insertions.
        """
        if start > end:
            raise ValueError(f"{self.sequence_id}: region start {start} > end {end}")
        columns = []
        for residue in range(start, end + 1):
            column = self.column_of(residue)
            if column is not None:
                columns.append(column)
        return columns

    def region_span(self, start: int, end: int) -> tuple[int, int] | None:
        """First and last column of an inclusive 1-based residue region.

        This is the span a viewer would highlight -- it DOES include interior
        gap columns, unlike ``columns_for_region``. Both are needed and they are
        different questions, which is why neither is the other's default.
        """
        columns = self.columns_for_region(start, end)
        return (columns[0], columns[-1]) if columns else None


def build_mapping(sequence_id: str, aligned: str) -> SequenceMapping:
    """Index one aligned sequence. Residue numbering ignores gaps, as it must."""
    residue_to_column: dict[int, int] = {}
    column_to_residue: dict[int, int] = {}
    residue_number = 0
    for index, character in enumerate(aligned, start=1):
        if is_gap(character):
            continue
        residue_number += 1
        residue_to_column[residue_number] = index
        column_to_residue[index] = residue_number
    return SequenceMapping(
        sequence_id=sequence_id,
        aligned=aligned,
        _residue_to_column=residue_to_column,
        _column_to_residue=column_to_residue,
    )


@dataclass(frozen=True)
class AlignmentIndex:
    """Every sequence of one alignment, indexed together."""

    mappings: dict[str, SequenceMapping]

    @property
    def alignment_length(self) -> int:
        return next(iter(self.mappings.values())).alignment_length

    @property
    def sequence_ids(self) -> list[str]:
        return list(self.mappings)

    def __getitem__(self, sequence_id: str) -> SequenceMapping:
        return self.mappings[sequence_id]

    def occupancy(self, column: int) -> int:
        """How many sequences have a residue (not a gap) at this column.

        The denominator of every per-column measure, and the reason a column
        present in three sequences out of two hundred must not be reported as
        conserved: with n = 3 the entropy is near zero whatever the residues are.
        """
        return sum(
            1 for m in self.mappings.values() if m.residue_at(column) is not None
        )

    def column_residues(self, column: int, *, include_gaps: bool = False) -> list[str]:
        """The characters at one column, gaps dropped unless asked for."""
        characters = [m.residue_character(column) for m in self.mappings.values()]
        if include_gaps:
            return characters
        return [c for c in characters if not is_gap(c)]

    def transfer(
        self, from_id: str, residue_number: int, to_id: str
    ) -> int | None:
        """The residue of ``to_id`` aligned to a residue of ``from_id``.

        Returns None when the target is gapped there -- which is the meaningful
        answer, not a failure: the orthologue has no residue at that position,
        and an APR cannot be conserved at a position that does not exist.
        """
        column = self[from_id].column_of(residue_number)
        if column is None:
            return None
        return self[to_id].residue_at(column)


def index_alignment(sequences: dict[str, str]) -> AlignmentIndex:
    """Index an aligned FASTA. Every sequence must share the alignment length."""
    if not sequences:
        raise ValueError("empty alignment")
    lengths = {len(seq) for seq in sequences.values()}
    if len(lengths) != 1:
        report = ", ".join(
            f"{sid}={len(seq)}" for sid, seq in sorted(sequences.items())
        )
        raise ValueError(
            f"sequences are not aligned to a common length ({report}). An "
            f"unaligned FASTA reaching this function would silently produce "
            f"column assignments that mean nothing."
        )
    return AlignmentIndex(
        mappings={sid: build_mapping(sid, seq) for sid, seq in sequences.items()}
    )


# --------------------------------------------------------------------------- #
# Array projections
#
# The mapping classes above answer per-residue questions and are the right shape
# for the conservation walk. The block and occupancy work needs whole arrays --
# 372 sequences x 2,280 columns, read repeatedly -- so these return numpy arrays
# built from the same rule. They are **0-based on the column axis**, unlike the
# classes above, because they index arrays; the function names say so and the
# boundary is crossed only here.
# --------------------------------------------------------------------------- #
def residue_to_column(aligned: str):
    """0-based columns of each residue, in residue order.

    ``out[i]`` is the column of the ``i``-th residue (0-based on both axes), so
    ``len(out)`` is the ungapped length.
    """
    import numpy as np

    return np.array(
        [i for i, c in enumerate(aligned) if not is_gap(c)], dtype=int
    )


def column_to_residue(aligned: str):
    """1-based residue number at each 0-based column, 0 where gapped.

    Zero is the "no residue here" marker rather than -1 or nan, so the array
    stays integer and a falsy test reads naturally. It does mean callers must
    not treat 0 as a residue number -- there is no residue 0 in 1-based
    numbering, which is why the convention is safe.
    """
    import numpy as np

    out = np.zeros(len(aligned), dtype=int)
    n = 0
    for column, character in enumerate(aligned):
        if not is_gap(character):
            n += 1
            out[column] = n
    return out


def project_mask(mask, aligned: str, n_columns: int):
    """Lift a per-residue boolean mask into alignment columns.

    Gap columns stay False for this sequence, which is the correct semantics
    twice over: a sequence that lacks the residue cannot have an APR there, and
    filling the interior gaps of a region would inflate it by other sequences'
    insertions. Compare ``SequenceMapping.region_span``, which deliberately
    does include them because a viewer highlight is a different question.

    The mask and the sequence are allowed to disagree in length: only the
    overlap is projected. A predictor run on a slightly different sequence
    version is a real and recoverable situation, and truncating is better than
    refusing the whole protein -- but it is also how a silent off-by-one hides,
    so callers that can check the lengths should.
    """
    import numpy as np

    out = np.zeros(n_columns, dtype=bool)
    columns = residue_to_column(aligned)
    usable = min(columns.size, len(mask))
    if usable:
        out[columns[:usable]] = np.asarray(mask, dtype=bool)[:usable]
    return out


def mask_from_regions(regions, length: int):
    """Boolean per-residue mask from inclusive 1-based ``(start, end)`` regions.

    Accepts tuples or any object with ``.start`` and ``.end``, so a consensus
    region object from the predictor side needs no unpacking at the call site.
    """
    import numpy as np

    mask = np.zeros(length, dtype=bool)
    for region in regions:
        start, end = (
            (region.start, region.end)
            if hasattr(region, "start")
            else (region[0], region[1])
        )
        mask[max(int(start) - 1, 0) : min(int(end), length)] = True
    return mask
