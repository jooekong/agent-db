"""Tests for semantic layer models."""

import pytest
from agent_db.semantic.models import (
    Entity,
    Attribute,
    EntityState,
    Lifecycle,
    SemanticType,
    AttributeSemanticType,
    EnumValue,
    CrossDatabaseMapping,
    DataSource,
    QueryRouting,
    DatabaseRole,
)


class TestEntity:
    def test_entity_basic_creation(self):
        entity = Entity(
            name="user",
            table="users",
            description="System users",
            semantic_type=SemanticType.ACTOR,
        )
        assert entity.name == "user"
        assert entity.table == "users"
        assert entity.semantic_type == SemanticType.ACTOR

    def test_entity_with_lifecycle(self):
        lifecycle = Lifecycle(
            created="created_at",
            updated="updated_at",
            deleted="deleted_at",
        )
        entity = Entity(
            name="user",
            table="users",
            description="System users",
            semantic_type=SemanticType.ACTOR,
            lifecycle=lifecycle,
        )
        assert entity.lifecycle.created == "created_at"

    def test_entity_with_states(self):
        states = [
            EntityState(name="active", condition="last_login_at > NOW() - INTERVAL '30d'"),
            EntityState(name="dormant", condition="last_login_at < NOW() - INTERVAL '30d'"),
        ]
        entity = Entity(
            name="user",
            table="users",
            description="System users",
            semantic_type=SemanticType.ACTOR,
            states=states,
        )
        assert len(entity.states) == 2
        assert entity.states[0].name == "active"


class TestAttribute:
    def test_attribute_basic(self):
        attr = Attribute(
            column="subscription_tier",
            semantic_type=AttributeSemanticType.DIMENSION,
        )
        assert attr.column == "subscription_tier"
        assert attr.semantic_type == AttributeSemanticType.DIMENSION

    def test_attribute_with_enum(self):
        enum_values = [
            EnumValue(value="free", meaning="Free user with limits", business_priority="low"),
            EnumValue(value="pro", meaning="Paid user", business_priority="high"),
        ]
        attr = Attribute(
            column="subscription_tier",
            semantic_type=AttributeSemanticType.DIMENSION,
            enum_values=enum_values,
        )
        assert len(attr.enum_values) == 2
        assert attr.enum_values[0].value == "free"


class TestCrossDatabaseMapping:
    def test_mapping_basic(self):
        sources = [
            DataSource(
                database="postgresql",
                entity="user",
                role=DatabaseRole.MASTER,
            ),
            DataSource(
                database="qdrant",
                collection="user_behavior_vectors",
                role=DatabaseRole.ENRICHMENT,
                provides=["User behavior vectors for similarity search"],
            ),
        ]
        mapping = CrossDatabaseMapping(
            name="user_unified_view",
            sources=sources,
        )
        assert mapping.name == "user_unified_view"
        assert len(mapping.sources) == 2

    def test_mapping_with_routing(self):
        routing = [
            QueryRouting(pattern="similar users", prefer="qdrant"),
            QueryRouting(pattern="user relationships", prefer="neo4j"),
        ]
        mapping = CrossDatabaseMapping(
            name="user_unified_view",
            sources=[],
            query_routing=routing,
        )
        assert len(mapping.query_routing) == 2
