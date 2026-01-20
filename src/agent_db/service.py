"""Main query service orchestrating all components."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from agent_db.semantic.loader import SchemaLoader, SemanticSchema
from agent_db.metadata.store import MetadataStore
from agent_db.metadata.models import ProfilingConfig
from agent_db.engine.intent import IntentParser
from agent_db.engine.planner import QueryPlanner
from agent_db.engine.executor import QueryExecutor
from agent_db.engine.interpreter import ResultInterpreter, InterpretedResult
from agent_db.llm.provider import LLMConfig
from agent_db.adapters.protocol import DatabaseAdapter
from agent_db.profiling.service import ProfilingService
from agent_db.identity.store import MappingStore


class ServiceConfig(BaseModel):
    """Service configuration."""

    schema_path: Path
    metadata_path: Path
    llm_model: str = "gpt-4"
    llm_api_key: Optional[str] = None
    profiling_config: Optional[ProfilingConfig] = None
    identity_path: Optional[Path] = None


class QueryService:
    """Main service orchestrating query flow."""

    def __init__(self, config: ServiceConfig):
        self.config = config

        # Load schema
        loader = SchemaLoader()
        self.schema: SemanticSchema = loader.load(config.schema_path)

        # Initialize metadata store
        self.metadata_store = MetadataStore(config.metadata_path)

        # Initialize identity mapping store
        identity_path = config.identity_path or (config.metadata_path / "identity")
        self.mapping_store = MappingStore(identity_path)

        # Initialize profiling service
        profiling_config = config.profiling_config or ProfilingConfig()
        self.profiling_service = ProfilingService(
            self.metadata_store,
            config=profiling_config,
        )

        # Initialize LLM config
        llm_config = LLMConfig(
            model=config.llm_model,
            api_key=config.llm_api_key,
        )

        # Initialize engine components
        self.intent_parser = IntentParser(llm_config)
        self.planner = QueryPlanner(llm_config, self.schema)
        self.executor = QueryExecutor(
            {}, schema=self.schema, mapping_store=self.mapping_store
        )  # Adapters added separately
        self.interpreter = ResultInterpreter(llm_config)

    def add_adapter(self, name: str, adapter: DatabaseAdapter) -> None:
        """Add database adapter."""
        self.executor.adapters[name] = adapter
        self.profiling_service.add_adapter(name, adapter)

    async def query(self, question: str) -> InterpretedResult:
        """Process natural language query."""
        # Parse intent
        intent = await self.intent_parser.parse(question)

        # Inject profile context into planner
        profile_context = self.profiling_service.format_profiles_for_llm()
        self.planner.set_profile_context(profile_context)

        # Generate plan
        plan = await self.planner.plan(intent)

        # Execute plan
        execution_result = await self.executor.execute(plan)

        # Interpret results
        interpreted = await self.interpreter.interpret(intent, execution_result)

        return interpreted
