"""Assign sequences to paralogue groups, without knowing the family.

An "orthologue set" pulled from EggNOG or STRING is routinely nothing of the
kind. The FXR1 set that prompted this module is 372 sequences described as FXR1
orthologues and is in fact the whole FMR1/FXR1/FXR2 family: 116 FXR1, 101 FMR1,
97 FXR2 and 58 single-copy invertebrate genes, across 152 taxa of which 82
contribute exactly three sequences. Every statement of the form "conserved
across orthologues" computed on that set is really a statement about a mixture,
and the paralogue differences are larger than the orthologue ones -- FXR1 and
FMR1 differ twofold in APR load, which swamps anything within either group.

So the group has to be established before conservation is measured, and it has
to be established from the data rather than from the set's name.

**The method, and what it assumes.** The caller supplies reference sequences
with group names -- typically one well-annotated species' paralogues. Every
sequence is scored by percent identity to each reference over the *core
columns*, the alignment columns where every reference carries a residue, and
assigned to its best match. The margin between best and second-best is reported
and is the only honest confidence available: a sequence 48% identical to one
reference and 47% to another has not been assigned, it has been rounded.

This assumes the references span the groups present. They usually do not, which
is the next function's job.

**Single-copy taxa are flagged, not labelled.** A taxon contributing one
sequence where the reference species has three is interesting: its lineage may
predate the duplication, or it may have lost copies, or the set may simply be
incomplete for it. Those are different facts and only a tree can separate them.
:func:`copy_number` reports the count per taxon and marks the deficient ones;
it does not call them ancestral, because "pre-duplication" is an inference and
this module reports observations. The caller that wants the inference should say
so in its own words and own it.

Nothing here is specific to FXR1, to vertebrates, or to three groups. The
references define the groups; two or twelve work the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Below this best-vs-second margin an assignment is not trusted. 0.05 = five
#: percentage points of identity, which on a structured core of a few hundred
#: columns is a handful of residues -- close enough to a tie that the ordering
#: would flip under a different aligner.
DEFAULT_MIN_MARGIN = 0.05

#: Group name for sequences the caller declares to sit outside the reference
#: groups. Deliberately not "pre-duplication" or "ancestral": those are
#: inferences from a tree, and this is a label.
OUTGROUP = "outgroup"

GAPS = frozenset("-.~")


@dataclass(frozen=True)
class Assignment:
    """One sequence's group assignment."""

    sequence_id: str
    group: str
    best_identity: float
    margin: float
    core_coverage: float
    identities: dict[str, float]
    #: Closest reference group by identity, regardless of ``group``. For an
    #: outgroup sequence these differ, and the difference is worth keeping: the
    #: nearest paralogue to a pre-duplication gene is a real observation, it is
    #: just not an assignment.
    nearest: str = ""
    #: Whether the caller declared this sequence outside the reference groups.
    #: A flag rather than a comparison of ``group`` against :data:`OUTGROUP`,
    #: because the label is the caller's to choose -- a family that calls its
    #: outgroup "Invertebrate" or "pre-duplication" must still be excluded from
    #: the group count, and inferring the status from the string silently fails
    #: for every label but the default.
    outgroup: bool = False

    @property
    def is_outgroup(self) -> bool:
        return self.outgroup

    @property
    def confident(self) -> bool:
        return self.margin >= DEFAULT_MIN_MARGIN

    def is_confident(self, min_margin: float = DEFAULT_MIN_MARGIN) -> bool:
        """Whether the best-vs-second margin clears ``min_margin``.

        An outgroup sequence is never confident: it has no group to be confident
        about, and its margin is small for the structural reason that it is
        roughly equidistant from every reference.
        """
        return not self.is_outgroup and self.margin >= min_margin


def core_columns(alignment: dict[str, str], references: dict[str, str]) -> np.ndarray:
    """Columns where every reference carries a residue.

    Identity is measured here and nowhere else. Comparing over all columns would
    score each sequence partly on which indels it shares with a reference, and
    the longest reference would win assignments by having more columns to agree
    on; restricting to the shared core makes the comparison symmetric.
    """
    missing = [r for r in references if r not in alignment]
    if missing:
        raise KeyError(
            f"reference(s) not in the alignment: {missing}. The references must "
            f"be members of the aligned set, not separate sequences -- otherwise "
            f"their columns do not correspond to anything."
        )
    arrays = [
        np.array([c not in GAPS for c in alignment[ref]]) for ref in references
    ]
    return np.flatnonzero(np.logical_and.reduce(arrays))


def assign(
    alignment: dict[str, str],
    references: dict[str, str],
    *,
    min_margin: float = DEFAULT_MIN_MARGIN,
    columns: np.ndarray | None = None,
    outgroup=None,
    outgroup_label: str = OUTGROUP,
) -> list[Assignment]:
    """Assign every sequence in ``alignment`` to its closest reference group.

    ``references`` maps a sequence id present in ``alignment`` to a group name,
    e.g. ``{"9606.ENSP00000350170": "FXR1", ...}``.

    Identity is computed over columns where BOTH sequences have a residue, so a
    fragment is not penalised for the residues it lacks; how much of the core it
    covers is reported separately as ``core_coverage`` and is what a completeness
    filter should use.

    ``outgroup`` names the sequences that sit OUTSIDE the reference groups --
    typically lineages predating the duplication that produced them. It takes a
    set of ids or a predicate on the id. Those sequences keep their identities
    and margin, which stay informative, but their ``group`` is
    ``outgroup_label`` rather than a paralogue name.

    This has to be the caller's declaration, because it is not recoverable from
    the identities. On the FXR1 family the 58 single-copy invertebrate genes all
    fall below the 0.05 margin -- every one of them, which makes a low margin a
    *sensitive* screen -- but so do 32 vertebrate sequences that genuinely
    belong to a paralogue. Reading the low-margin flag as "outgroup" would move
    those 32 out of the group they belong to; reading it as "ambiguous" is what
    it actually means. Which sequences predate the duplication is a statement
    about the tree, and the caller who makes it should own it.
    """
    if len(references) < 2:
        raise ValueError(
            f"need at least two references to assign between groups, got "
            f"{len(references)}. With one reference every sequence is assigned "
            f"to it by construction and the margin is undefined."
        )
    cols = core_columns(alignment, references) if columns is None else columns
    if cols.size == 0:
        raise ValueError(
            "the references share no column carrying a residue in all of them, "
            "so there is no common core to measure identity over. Check that "
            "they are aligned to each other and not merely present."
        )

    if outgroup is None:
        is_outgroup = lambda _sid: False  # noqa: E731
    elif callable(outgroup):
        is_outgroup = outgroup
    else:
        members = set(outgroup)
        is_outgroup = lambda sid: sid in members  # noqa: E731

    ref_arrays = {
        name: np.array(list(alignment[rid]))[cols]
        for rid, name in references.items()
    }

    out: list[Assignment] = []
    for sid, aligned in alignment.items():
        arr = np.array(list(aligned))[cols]
        present = ~np.isin(arr, list(GAPS))
        identities: dict[str, float] = {}
        for name, ref in ref_arrays.items():
            both = present & ~np.isin(ref, list(GAPS))
            identities[name] = (
                float((arr[both] == ref[both]).mean()) if both.any() else float("nan")
            )

        ranked = sorted(
            identities.items(),
            key=lambda kv: (-kv[1] if np.isfinite(kv[1]) else np.inf),
        )
        best_name, best_value = ranked[0]
        second_value = ranked[1][1] if len(ranked) > 1 else float("nan")
        margin = (
            float(best_value - second_value)
            if np.isfinite(best_value) and np.isfinite(second_value)
            else float("nan")
        )
        declared = bool(is_outgroup(sid))
        out.append(
            Assignment(
                sequence_id=sid,
                group=outgroup_label if declared else best_name,
                best_identity=float(best_value),
                margin=margin,
                core_coverage=float(present.mean()),
                identities=identities,
                nearest=best_name,
                outgroup=declared,
            )
        )
    return out


def copy_number(
    assignments: list[Assignment],
    taxid_of,
    *,
    n_groups: int | None = None,
) -> dict:
    """Sequences per taxon, and which taxa carry fewer than the reference set.

    ``taxid_of`` maps a sequence id to a taxon key (a taxid, a species name --
    anything hashable and consistent).

    Returns ``{"per_taxon": {taxon: n}, "deficient": [taxon, ...],
    "complete": [taxon, ...], "n_groups": int}``.

    A deficient taxon is an OBSERVATION: it contributed fewer sequences than
    there are groups. It may predate the duplication, it may have lost a copy,
    or the orthologue set may be incomplete for it -- and those are separable
    only on a tree. Calling the set "pre-duplication" on this evidence alone is
    the inference this function declines to make for you.
    """
    # The outgroup is not one of the groups a complete genome should carry, so
    # it must not raise the bar every taxon is judged against.
    groups = {a.group for a in assignments if not a.is_outgroup}
    n_groups = n_groups if n_groups is not None else len(groups)

    per_taxon: dict[object, int] = {}
    for a in assignments:
        key = taxid_of(a.sequence_id) if callable(taxid_of) else taxid_of.get(a.sequence_id)
        if key in (None, ""):
            continue
        per_taxon[key] = per_taxon.get(key, 0) + 1

    return {
        "per_taxon": per_taxon,
        "deficient": sorted(
            (t for t, n in per_taxon.items() if n < n_groups), key=str
        ),
        "complete": sorted(
            (t for t, n in per_taxon.items() if n >= n_groups), key=str
        ),
        "n_groups": n_groups,
    }


def assignment_table(assignments: list[Assignment], *, min_margin: float = DEFAULT_MIN_MARGIN):
    """Assignments as a DataFrame, one row per sequence."""
    import pandas as pd

    rows = []
    for a in assignments:
        row = {
            "id": a.sequence_id,
            "group": a.group,
            "nearest": a.nearest,
            "best_identity": a.best_identity,
            "margin": a.margin,
            "core_coverage": a.core_coverage,
            "confident": a.is_confident(min_margin),
        }
        for name, value in sorted(a.identities.items()):
            row[f"identity_{name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(assignments: list[Assignment], *, min_margin: float = DEFAULT_MIN_MARGIN) -> str:
    """A short report. The ambiguous count is the one that decides whether a
    paralogue-contrastive claim can be made at all."""
    total = len(assignments)
    if total == 0:
        return "No sequences to assign."
    counts: dict[str, int] = {}
    for a in assignments:
        counts[a.group] = counts.get(a.group, 0) + 1
    n_out = sum(1 for a in assignments if a.is_outgroup)
    # Ambiguity is only meaningful for sequences that were assigned; an outgroup
    # sequence is low-margin by construction and counting it as ambiguous would
    # inflate the number that matters.
    assigned = [a for a in assignments if not a.is_outgroup]
    ambiguous = sum(1 for a in assigned if not a.is_confident(min_margin))

    lines = [
        f"{total} sequence(s) assigned to "
        f"{len(counts) - (1 if n_out else 0)} group(s):",
        *(f"    {g:20s} {n:5d}  ({n / total:.1%})"
          for g, n in sorted(counts.items(), key=lambda kv: -kv[1])),
        "",
    ]
    if n_out:
        lines.append(
            f"{n_out} sequence(s) were declared outside the reference groups "
            f"and carry no paralogue call; their nearest reference is reported "
            f"as an observation, not an assignment."
        )
    if assigned:
        lines.append(
            f"{ambiguous} of {len(assigned)} assigned "
            f"({ambiguous / len(assigned):.1%}) have a best-vs-second margin "
            f"below {min_margin:g} and should be excluded from any contrast "
            f"BETWEEN groups; they are still usable within-set."
        )
    return "\n".join(lines)
