"""Tests for YAML schema loader."""

from pathlib import Path

import pytest

from agent_db.semantic.loader import SchemaLoader, SemanticSchema


@pytest.fixture
def sample_schema_path() -> Path:
    return Path(__file__).parent.parent / "fixtures" / "sample_schema.yaml"


class TestSchemaLoader:
    def test_load_from_file(self, sample_schema_path: Path):
        loader = SchemaLoader()
        schema = loader.load(sample_schema_path)

        assert isinstance(schema, SemanticSchema)
        assert len(schema.entities) == 1
        assert schema.entities[0].name == "user"

    def test_load_entity_details(self, sample_schema_path: Path):
        loader = SchemaLoader()
        schema = loader.load(sample_schema_path)

        user = schema.entities[0]
        assert user.table == "users"
        assert user.lifecycle is not None
        assert user.lifecycle.created == "created_at"
        assert len(user.states) == 2
        assert len(user.attributes) == 1

    def test_load_cross_database_mappings(self, sample_schema_path: Path):
        loader = SchemaLoader()
        schema = loader.load(sample_schema_path)

        assert len(schema.cross_database_mappings) == 1
        mapping = schema.cross_database_mappings[0]
        assert mapping.name == "user_unified_view"
        assert len(mapping.sources) == 2

    def test_get_entity_by_name(self, sample_schema_path: Path):
        loader = SchemaLoader()
        schema = loader.load(sample_schema_path)

        user = schema.get_entity("user")
        assert user is not None
        assert user.name == "user"

        unknown = schema.get_entity("unknown")
        assert unknown is None

    def test_load_from_string(self):
        loader = SchemaLoader()
        yaml_content = """
entities:
  - name: order
    table: orders
    description: "Customer orders"
    semantic_type: event
"""
        schema = loader.load_from_string(yaml_content)
        assert len(schema.entities) == 1
        assert schema.entities[0].name == "order"

    def test_load_entity_identity(self, sample_schema_path: Path):
        loader = SchemaLoader()
        schema = loader.load(sample_schema_path)

        user = schema.entities[0]
        assert user.identity is not None
        assert user.identity.canonical_id == "user_id"
        assert len(user.identity.sources) == 2
        assert user.identity.sources[0].database == "postgresql"
        assert user.identity.sources[0].key_column == "id"
        assert len(user.identity.match_rules) == 2
        assert user.identity.match_rules[0].name == "email_match"
