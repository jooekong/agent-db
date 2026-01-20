"""Factory for creating database adapters."""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from agent_db.adapters.protocol import ConnectionConfig, DatabaseAdapter, DatabaseType


class DatabasesConfig(BaseModel):
    """Configuration for multiple databases."""

    databases: dict[str, ConnectionConfig] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "DatabasesConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatabasesConfig":
        """Load configuration from dictionary."""
        return cls.model_validate(data)


# Registry of adapter classes
_ADAPTER_REGISTRY: dict[DatabaseType, type[DatabaseAdapter]] = {}


def register_adapter(db_type: DatabaseType):
    """Decorator to register an adapter class."""
    def decorator(cls: type[DatabaseAdapter]) -> type[DatabaseAdapter]:
        _ADAPTER_REGISTRY[db_type] = cls
        return cls
    return decorator


def create_adapter(config: ConnectionConfig) -> DatabaseAdapter:
    """Create adapter instance from configuration."""
    adapter_class = _ADAPTER_REGISTRY.get(config.database_type)
    if not adapter_class:
        raise ValueError(f"No adapter registered for: {config.database_type}")
    return adapter_class(config)


def create_adapters(configs: DatabasesConfig) -> dict[str, DatabaseAdapter]:
    """Create multiple adapter instances from configuration."""
    adapters = {}
    for name, config in configs.databases.items():
        config.name = name  # Ensure name matches config key
        adapters[name] = create_adapter(config)
    return adapters


async def connect_all(adapters: dict[str, DatabaseAdapter]) -> None:
    """Connect all adapters."""
    for adapter in adapters.values():
        await adapter.connect()


async def disconnect_all(adapters: dict[str, DatabaseAdapter]) -> None:
    """Disconnect all adapters."""
    for adapter in adapters.values():
        await adapter.disconnect()
