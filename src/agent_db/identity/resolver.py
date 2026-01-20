"""Identity resolution logic for offline mapping."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from agent_db.identity.models import IdentityLink, MatchRule
from agent_db.semantic.models import MatchStrategy
from agent_db.identity.store import MappingStore
from agent_db.semantic.models import EntityIdentity, IdentityMatchRule, IdentitySource


class IdentityResolver:
    """Resolve identities across sources and persist mappings."""

    def __init__(self, identity: EntityIdentity, store: MappingStore):
        self.identity = identity
        self.store = store
        self._rules = [self._to_match_rule(r) for r in identity.match_rules]

    def resolve_batch(self, source_records: dict[str, list[dict[str, Any]]]) -> list[IdentityLink]:
        """Resolve identities across sources using provided records."""
        if not self.identity.sources:
            return []

        primary = self.identity.sources[0]
        primary_records = source_records.get(primary.database, [])

        canonical_records = self._build_canonical_records(primary, primary_records)
        links: list[IdentityLink] = []

        for canonical_id, record in canonical_records.items():
            source_key = self._extract_value(record, primary.key_column)
            if source_key is None:
                continue
            link = IdentityLink(
                canonical_id=canonical_id,
                source=primary.database,
                source_key=str(source_key),
                match_rule="primary",
                confidence=1.0,
                provenance={"source": primary.database},
            )
            self.store.save_link(link)
            links.append(link)

        for source in self.identity.sources[1:]:
            records = source_records.get(source.database, [])
            for record in records:
                link = self._match_record(source, record, canonical_records)
                if link:
                    self.store.save_link(link)
                    links.append(link)
                else:
                    fallback = self._fallback_link(source, record)
                    if fallback:
                        self.store.save_link(fallback)
                        links.append(fallback)

        return links

    def _match_record(
        self,
        source: IdentitySource,
        record: dict[str, Any],
        canonical_records: dict[str, dict[str, Any]],
    ) -> IdentityLink | None:
        for rule in self._rules:
            if rule.strategy == MatchStrategy.EXACT:
                match = self._match_exact(rule, source, record, canonical_records)
                if match:
                    return match
            if rule.strategy == MatchStrategy.FUZZY:
                match = self._match_fuzzy(rule, source, record, canonical_records)
                if match:
                    return match
        return None

    def _match_exact(
        self,
        rule: MatchRule,
        source: IdentitySource,
        record: dict[str, Any],
        canonical_records: dict[str, dict[str, Any]],
    ) -> IdentityLink | None:
        for canonical_id, canonical in canonical_records.items():
            if self._fields_match(rule.fields, source, record, canonical):
                return self._build_link(rule, source, record, canonical_id, confidence=rule.confidence)
        return None

    def _match_fuzzy(
        self,
        rule: MatchRule,
        source: IdentitySource,
        record: dict[str, Any],
        canonical_records: dict[str, dict[str, Any]],
    ) -> IdentityLink | None:
        threshold = rule.threshold if rule.threshold is not None else 0.85
        for canonical_id, canonical in canonical_records.items():
            score = self._similarity_score(rule.fields, source, record, canonical)
            if score >= threshold:
                return self._build_link(rule, source, record, canonical_id, confidence=score)
        return None

    def _fields_match(
        self,
        fields: list[str],
        source: IdentitySource,
        record: dict[str, Any],
        canonical: dict[str, Any],
    ) -> bool:
        for field in fields:
            source_field = source.field_map.get(field, field)
            source_value = self._extract_value(record, source_field)
            canonical_value = self._extract_value(canonical, field)
            if source_value is None or canonical_value is None:
                return False
            if str(source_value).strip().lower() != str(canonical_value).strip().lower():
                return False
        return True

    def _similarity_score(
        self,
        fields: list[str],
        source: IdentitySource,
        record: dict[str, Any],
        canonical: dict[str, Any],
    ) -> float:
        scores = []
        for field in fields:
            source_field = source.field_map.get(field, field)
            source_value = self._extract_value(record, source_field)
            canonical_value = self._extract_value(canonical, field)
            if source_value is None or canonical_value is None:
                continue
            scores.append(self._value_similarity(source_value, canonical_value))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    @staticmethod
    def _value_similarity(left: Any, right: Any) -> float:
        if left is None or right is None:
            return 0.0
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if left == right:
                return 1.0
            denom = max(abs(left), abs(right), 1.0)
            return max(0.0, 1.0 - abs(left - right) / denom)
        left_text = str(left).strip().lower()
        right_text = str(right).strip().lower()
        if not left_text or not right_text:
            return 0.0
        return SequenceMatcher(None, left_text, right_text).ratio()

    def _fallback_link(self, source: IdentitySource, record: dict[str, Any]) -> IdentityLink | None:
        source_key = self._extract_value(record, source.key_column)
        if source_key is None:
            return None
        canonical_id = self._make_canonical_id(source.database, source_key)
        return IdentityLink(
            canonical_id=canonical_id,
            source=source.database,
            source_key=str(source_key),
            match_rule="fallback",
            confidence=0.5,
            provenance={"source": source.database},
        )

    @staticmethod
    def _make_canonical_id(source: str, source_key: Any) -> str:
        return f"{source}:{source_key}"

    def _build_canonical_records(
        self, source: IdentitySource, records: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        canonical_records: dict[str, dict[str, Any]] = {}
        for record in records:
            source_key = self._extract_value(record, source.key_column)
            if source_key is None:
                continue
            canonical_id = self._make_canonical_id(source.database, source_key)
            canonical_records[canonical_id] = self._canonicalize_record(source, record)
        return canonical_records

    def _canonicalize_record(
        self, source: IdentitySource, record: dict[str, Any]
    ) -> dict[str, Any]:
        canonical = {}
        # Always include the key column
        canonical[source.key_column] = self._extract_value(record, source.key_column)
        for canonical_field, source_field in source.field_map.items():
            canonical[canonical_field] = self._extract_value(record, source_field)
        return canonical

    def _build_link(
        self,
        rule: MatchRule,
        source: IdentitySource,
        record: dict[str, Any],
        canonical_id: str,
        confidence: float | None,
    ) -> IdentityLink | None:
        source_key = self._extract_value(record, source.key_column)
        if source_key is None:
            return None
        return IdentityLink(
            canonical_id=canonical_id,
            source=source.database,
            source_key=str(source_key),
            match_rule=rule.name,
            confidence=confidence if confidence is not None else 1.0,
            provenance={"rule": rule.name, "strategy": rule.strategy.value},
        )

    @staticmethod
    def _extract_value(record: dict[str, Any], field_path: str) -> Any:
        if "." not in field_path:
            return record.get(field_path)
        current: Any = record
        for part in field_path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    @staticmethod
    def _to_match_rule(rule: IdentityMatchRule) -> MatchRule:
        return MatchRule(
            name=rule.name,
            strategy=MatchStrategy(rule.strategy.value),
            fields=rule.fields,
            threshold=rule.threshold,
            confidence=rule.confidence,
        )
