# Agent-DB

AI-Native Data Interaction Platform - 将数据库访问能力直接暴露给 LLM，实现人与数据的自然语言交互。

## 特性

- **静态语义层** - YAML 配置定义实体、属性和跨数据库映射
- **动态元数据层** - 数据摘要、关联洞察、查询模式库
- **多数据库支持** - PostgreSQL、Qdrant、Neo4j、InfluxDB
- **LLM 抽象层** - 通过 LiteLLM 支持多种 LLM 提供商
- **查询引擎** - 意图解析 → 查询规划 → 执行 → 结果解释

## 安装

```bash
pip install -e ".[dev]"
```

## 快速开始

### 1. 定义语义 Schema

创建 `schema.yaml`:

```yaml
entities:
  - name: user
    table: users
    description: "System users"
    semantic_type: actor
    lifecycle:
      created: created_at
      updated: updated_at
    attributes:
      - column: subscription_tier
        semantic_type: dimension
        enum_values:
          - value: free
            meaning: "Free user"
          - value: pro
            meaning: "Paid user"

cross_database_mappings:
  - name: user_unified_view
    sources:
      - database: postgresql
        entity: user
        role: master
      - database: qdrant
        collection: user_vectors
        role: enrichment
```

### 2. 使用服务

```python
from pathlib import Path
from agent_db.service import QueryService, ServiceConfig
from agent_db.adapters import PostgreSQLAdapter, ConnectionConfig, DatabaseType

# 配置
config = ServiceConfig(
    schema_path=Path("schema.yaml"),
    metadata_path=Path("metadata/"),
    llm_model="gpt-4",
    llm_api_key="your-api-key",
)

# 创建服务
service = QueryService(config)

# 添加数据库适配器
pg_adapter = PostgreSQLAdapter(ConnectionConfig(
    database_type=DatabaseType.POSTGRESQL,
    host="localhost",
    port=5432,
    database="mydb",
    user="user",
    password="pass",
))
await pg_adapter.connect()
service.add_adapter("postgresql", pg_adapter)

# 查询
result = await service.query("How many users do we have?")
print(result.summary)
```

### 3. 启动 API 服务

```bash
python -m agent_db
```

API 将在 `http://localhost:8000` 启动。

## 开发

```bash
# 运行测试
pytest

# 类型检查
mypy src/

# 代码检查
ruff check src/
```

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    用户层 (API/CLI)                      │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                   查询引擎 (Engine)                      │
│   意图解析 → 查询规划 → 执行 → 结果解释                  │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                统一数据访问层 (Adapters)                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │PostgreSQL│  │  Qdrant │  │  Neo4j  │  │InfluxDB │    │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘    │
└─────────────────────────────────────────────────────────┘
```

## License

MIT
