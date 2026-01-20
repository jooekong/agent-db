"""SQL templates for profiling queries across databases."""

from string import Template
from typing import Any


class SQLTemplates:
    """SQL/Cypher/Flux query templates for data profiling."""

    # PostgreSQL Templates
    POSTGRESQL = {
        "row_count": "SELECT COUNT(*) AS cnt FROM {table}",
        "column_info": """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
        """,
        "numeric_stats": """
            SELECT
                MIN({column}) AS min_val,
                MAX({column}) AS max_val,
                AVG({column}::numeric) AS avg_val,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column}) AS median_val,
                COUNT(DISTINCT {column}) AS distinct_count,
                SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END)::float / COUNT(*) AS null_ratio
            FROM {table}
        """,
        "numeric_percentiles": """
            SELECT
                PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY {column}) AS p5,
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY {column}) AS p10,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) AS p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {column}) AS p50,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) AS p75,
                PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY {column}) AS p90,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {column}) AS p95,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY {column}) AS p99
            FROM {table}
        """,
        "text_stats": """
            SELECT
                MIN(LENGTH({column})) AS min_length,
                MAX(LENGTH({column})) AS max_length,
                AVG(LENGTH({column})) AS avg_length,
                COUNT(DISTINCT {column}) AS distinct_count,
                SUM(CASE WHEN {column} IS NULL OR {column} = '' THEN 1 ELSE 0 END)::float / COUNT(*) AS empty_ratio,
                SUM(CASE WHEN {column} IS NULL THEN 1 ELSE 0 END)::float / COUNT(*) AS null_ratio
            FROM {table}
        """,
        "top_values": """
            SELECT {column}, COUNT(*) AS cnt
            FROM {table}
            WHERE {column} IS NOT NULL
            GROUP BY {column}
            ORDER BY cnt DESC
            LIMIT {limit}
        """,
        "sample_random": """
            SELECT * FROM {table}
            ORDER BY RANDOM()
            LIMIT {sample_size}
        """,
        "sample_tablesample": """
            SELECT * FROM {table} TABLESAMPLE SYSTEM({percentage})
        """,
        "sample_column": """
            SELECT {column} FROM {table}
            WHERE {column} IS NOT NULL
            ORDER BY RANDOM()
            LIMIT {sample_size}
        """,
        "schema_hash": """
            SELECT MD5(STRING_AGG(column_name || data_type, ',' ORDER BY ordinal_position)) AS hash
            FROM information_schema.columns
            WHERE table_name = '{table}'
        """,
    }

    # Neo4j Cypher Templates
    NEO4J = {
        "row_count": "MATCH (n:{label}) RETURN COUNT(n) AS cnt",
        "property_keys": """
            MATCH (n:{label})
            WITH DISTINCT keys(n) AS key_list
            UNWIND key_list AS key
            RETURN DISTINCT key
        """,
        "numeric_stats": """
            MATCH (n:{label})
            WHERE n.{property} IS NOT NULL
            RETURN
                MIN(n.{property}) AS min_val,
                MAX(n.{property}) AS max_val,
                AVG(n.{property}) AS avg_val,
                COUNT(DISTINCT n.{property}) AS distinct_count
        """,
        "sample_random": """
            MATCH (n:{label})
            RETURN n
            ORDER BY rand()
            LIMIT {sample_size}
        """,
        "sample_property": """
            MATCH (n:{label})
            WHERE n.{property} IS NOT NULL
            RETURN n.{property} AS value
            ORDER BY rand()
            LIMIT {sample_size}
        """,
        "null_ratio": """
            MATCH (n:{label})
            WITH COUNT(n) AS total,
                 SUM(CASE WHEN n.{property} IS NULL THEN 1 ELSE 0 END) AS nulls
            RETURN nulls * 1.0 / total AS null_ratio
        """,
    }

    # InfluxDB Flux Templates
    INFLUXDB = {
        "row_count": """
            from(bucket: "{bucket}")
                |> range(start: {start}, stop: {stop})
                |> filter(fn: (r) => r._measurement == "{measurement}")
                |> count()
        """,
        "field_keys": """
            import "influxdata/influxdb/schema"
            schema.fieldKeys(bucket: "{bucket}", predicate: (r) => r._measurement == "{measurement}")
        """,
        "numeric_stats": """
            from(bucket: "{bucket}")
                |> range(start: {start}, stop: {stop})
                |> filter(fn: (r) => r._measurement == "{measurement}" and r._field == "{field}")
                |> group()
                |> reduce(fn: (r, accumulator) => ({{
                    min_val: if r._value < accumulator.min_val then r._value else accumulator.min_val,
                    max_val: if r._value > accumulator.max_val then r._value else accumulator.max_val,
                    sum: accumulator.sum + r._value,
                    count: accumulator.count + 1
                }}), identity: {{min_val: 999999999.0, max_val: -999999999.0, sum: 0.0, count: 0}})
        """,
        "sample_random": """
            from(bucket: "{bucket}")
                |> range(start: {start}, stop: {stop})
                |> filter(fn: (r) => r._measurement == "{measurement}" and r._field == "{field}")
                |> sample(n: {sample_size})
        """,
        "percentiles": """
            from(bucket: "{bucket}")
                |> range(start: {start}, stop: {stop})
                |> filter(fn: (r) => r._measurement == "{measurement}" and r._field == "{field}")
                |> quantile(q: {quantile})
        """,
    }

    @classmethod
    def get_template(cls, database_type: str, template_name: str) -> str:
        """Get template by database type and name."""
        templates = {
            "postgresql": cls.POSTGRESQL,
            "neo4j": cls.NEO4J,
            "influxdb": cls.INFLUXDB,
        }

        db_templates = templates.get(database_type.lower())
        if not db_templates:
            raise ValueError(f"Unknown database type: {database_type}")

        template = db_templates.get(template_name)
        if not template:
            raise ValueError(f"Unknown template: {template_name} for {database_type}")

        return template.strip()

    @classmethod
    def render(
        cls, database_type: str, template_name: str, **params: Any
    ) -> str:
        """Render template with parameters."""
        template = cls.get_template(database_type, template_name)
        return template.format(**params)
