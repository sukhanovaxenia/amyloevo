"""Paralogue and group assignment.

An orthologue set is routinely a paralogue family. Establish the groups before
measuring conservation, or the paralogue differences -- which are usually the
larger ones -- are reported as orthologue variation.
"""

from .paralogues import (
    Assignment,
    assign,
    assignment_table,
    copy_number,
    core_columns,
    summarise,
)

__all__ = [
    "Assignment",
    "assign",
    "assignment_table",
    "copy_number",
    "core_columns",
    "summarise",
]
