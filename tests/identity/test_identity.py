"""Tests for identity resolution module."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from agent_db.identity.models import IdentityLink, MatchRule, ResolutionJob, ResolutionStatus
from agent_db.identity.store import MappingStore
from agent_db.identity.resolver import IdentityResolver
from agent_db.semantic.models import (
    EntityIdentity,
    IdentitySource,
    IdentityMatchRule,
    MatchStrategy,
)


@pytest.fixture
def temp_store_path() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "identity"


@pytest.fixture
def sample_link() -> IdentityLink:
    return IdentityLink(
        canonical_id="postgresql:1",
        source="postgresql",
        source_key="1",
        match_rule="primary",
        confidence=1.0,
        provenance={"source": "postgresql"},
    )


class TestIdentityLink:
    def test_link_creation(self, sample_link: IdentityLink):
        assert sample_link.canonical_id == "postgresql:1"
        assert sample_link.confidence == 1.0

    def test_link_confidence_validation(self):
        with pytest.raises(ValueError):
            IdentityLink(
                canonical_id="test:1",
                source="test",
                source_key="1",
                match_rule="test",
                confidence=1.5,  # Invalid
            )


class TestMatchRule:
    def test_rule_creation(self):
        rule = MatchRule(
            name="email_match",
            strategy=MatchStrategy.EXACT,
            fields=["email"],
            confidence=0.95,
        )
        assert rule.name == "email_match"
        assert rule.strategy == MatchStrategy.EXACT

    def test_fuzzy_rule(self):
        rule = MatchRule(
            name="name_fuzzy",
            strategy=MatchStrategy.FUZZY,
            fields=["name", "email"],
            threshold=0.85,
        )
        assert rule.threshold == 0.85


class TestMappingStore:
    def test_save_and_get_link(self, temp_store_path: Path, sample_link: IdentityLink):
        store = MappingStore(temp_store_path)
        store.save_link(sample_link)

        links = store.get_links("postgresql:1")
        assert len(links) == 1
        assert links[0].source_key == "1"

    def test_upsert_link(self, temp_store_path: Path, sample_link: IdentityLink):
        store = MappingStore(temp_store_path)
        store.save_link(sample_link)

        updated_link = IdentityLink(
            canonical_id="postgresql:1",
            source="postgresql",
            source_key="1",
            match_rule="primary",
            confidence=0.9,  # Changed
            provenance={"source": "postgresql"},
        )
        store.save_link(updated_link)

        links = store.get_links("postgresql:1")
        assert len(links) == 1
        assert links[0].confidence == 0.9

    def test_list_canonical_ids(self, temp_store_path: Path, sample_link: IdentityLink):
        store = MappingStore(temp_store_path)
        store.save_link(sample_link)

        link2 = IdentityLink(
            canonical_id="postgresql:2",
            source="postgresql",
            source_key="2",
            match_rule="primary",
            confidence=1.0,
        )
        store.save_link(link2)

        ids = store.list_canonical_ids()
        assert len(ids) == 2
        assert "postgresql:1" in ids
        assert "postgresql:2" in ids

    def test_get_source_keys(self, temp_store_path: Path):
        store = MappingStore(temp_store_path)

        # Save links for multiple sources
        store.save_link(IdentityLink(
            canonical_id="user:1",
            source="postgresql",
            source_key="pg_1",
            match_rule="primary",
            confidence=1.0,
        ))
        store.save_link(IdentityLink(
            canonical_id="user:1",
            source="qdrant",
            source_key="qd_1",
            match_rule="email_match",
            confidence=0.95,
        ))

        mapping = store.get_source_keys(["user:1"], "postgresql")
        assert mapping == {"user:1": "pg_1"}

        mapping = store.get_source_keys(["user:1"], "qdrant")
        assert mapping == {"user:1": "qd_1"}

    def test_get_canonical_ids(self, temp_store_path: Path):
        store = MappingStore(temp_store_path)

        store.save_link(IdentityLink(
            canonical_id="user:1",
            source="postgresql",
            source_key="pg_1",
            match_rule="primary",
            confidence=1.0,
        ))

        mapping = store.get_canonical_ids("postgresql", ["pg_1", "pg_2"])
        assert mapping == {"pg_1": "user:1"}

    def test_save_and_get_job(self, temp_store_path: Path):
        store = MappingStore(temp_store_path)

        job = ResolutionJob(
            job_id="job_001",
            entity="user",
            status=ResolutionStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        store.save_job(job)

        loaded = store.get_job("job_001")
        assert loaded is not None
        assert loaded.status == ResolutionStatus.RUNNING


class TestIdentityResolver:
    @pytest.fixture
    def identity(self) -> EntityIdentity:
        return EntityIdentity(
            canonical_id="user_id",
            sources=[
                IdentitySource(
                    database="postgresql",
                    entity="user",
                    key_column="id",
                    field_map={"email": "email", "name": "name"},
                ),
                IdentitySource(
                    database="qdrant",
                    collection="user_vectors",
                    key_column="user_id",
                    field_map={"email": "user_email"},
                ),
            ],
            match_rules=[
                IdentityMatchRule(
                    name="email_match",
                    strategy=MatchStrategy.EXACT,
                    fields=["email"],
                    confidence=0.95,
                ),
            ],
        )

    def test_resolve_batch_exact_match(self, temp_store_path: Path, identity: EntityIdentity):
        store = MappingStore(temp_store_path)
        resolver = IdentityResolver(identity, store)

        source_records = {
            "postgresql": [
                {"id": 1, "email": "alice@example.com", "name": "Alice"},
                {"id": 2, "email": "bob@example.com", "name": "Bob"},
            ],
            "qdrant": [
                {"user_id": "q1", "user_email": "alice@example.com"},
                {"user_id": "q2", "user_email": "charlie@example.com"},
            ],
        }

        links = resolver.resolve_batch(source_records)

        # Should have links for:
        # - 2 primary records from postgresql
        # - 1 matched record from qdrant (alice)
        # - 1 fallback record from qdrant (charlie)
        assert len(links) == 4

        # Verify alice matched
        alice_links = [l for l in links if "alice" in str(l.provenance) or l.source_key == "q1"]
        assert any(l.source == "qdrant" and l.match_rule == "email_match" for l in links)

    def test_resolve_batch_fallback(self, temp_store_path: Path, identity: EntityIdentity):
        store = MappingStore(temp_store_path)
        resolver = IdentityResolver(identity, store)

        source_records = {
            "postgresql": [
                {"id": 1, "email": "alice@example.com", "name": "Alice"},
            ],
            "qdrant": [
                {"user_id": "q999", "user_email": "unknown@example.com"},
            ],
        }

        links = resolver.resolve_batch(source_records)

        # Should have fallback link for unmatched qdrant record
        fallback_links = [l for l in links if l.match_rule == "fallback"]
        assert len(fallback_links) == 1
        assert fallback_links[0].confidence == 0.5
