"""Linker pass: finds registry entities mentioned in chunk text.

Dictionary match on the human-editable ``entity_aliases`` table. An alias
shared by more than one entity is ambiguous: it is linked only if the LLM
confirms it, and dropped when no LLM is configured.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

# Entity types and the payload field their IDs go in.
PAYLOAD_FIELDS = {
    "asset": "asset_ids",
    "location": "location_ids",
    "work": "work_ids",
    "event": "event_ids",
    "training": "training_ids",
}
# Short aliases ("lab", "CNC") match too much prose to be worth it.
MIN_ALIAS_LENGTH = 4

# (text, alias, entity_type, entity_id) -> does the alias refer to the entity?
Confirm = Callable[[str, str, str, str], bool]


@dataclass
class Links:
    ids: dict[str, list[str]] = field(default_factory=dict)

    def get(self, payload_field: str) -> list[str]:
        return self.ids.get(payload_field, [])


class Linker:
    def __init__(
        self, aliases: Iterable[tuple[str, str, str]], confirm: Confirm | None = None
    ) -> None:
        """``aliases`` are (entity_type, entity_id, alias) rows."""
        by_alias: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for entity_type, entity_id, alias in aliases:
            if entity_type in PAYLOAD_FIELDS and len(alias.strip()) >= MIN_ALIAS_LENGTH:
                by_alias[alias.strip().lower()].add((entity_type, entity_id))
        self._entities = dict(by_alias)
        self._confirm = confirm
        self._pattern = (
            re.compile(
                r"\b(?:"
                + "|".join(re.escape(a) for a in sorted(self._entities, key=len, reverse=True))
                + r")\b",
                re.IGNORECASE,
            )
            if self._entities
            else None
        )

    def link(self, text: str) -> Links:
        if self._pattern is None:
            return Links()
        found: dict[str, set[str]] = defaultdict(set)
        for alias in {m.group().lower() for m in self._pattern.finditer(text)}:
            entities = self._entities.get(alias, set())
            for entity_type, entity_id in sorted(entities):
                if len(entities) > 1 and (
                    self._confirm is None or not self._confirm(text, alias, entity_type, entity_id)
                ):
                    continue
                found[PAYLOAD_FIELDS[entity_type]].add(entity_id)
        return Links({name: sorted(ids) for name, ids in found.items()})
