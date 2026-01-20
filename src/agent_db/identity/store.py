"""File-based identity mapping store."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from agent_db.identity.models import IdentityLink, ResolutionJob


class MappingStore:
    """Store canonical ID mappings on disk."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._links_path = base_path / "links"
        self._jobs_path = base_path / "jobs"

        for path in [self._links_path, self._jobs_path]:
            path.mkdir(parents=True, exist_ok=True)

    def save_link(self, link: IdentityLink) -> None:
        """Upsert a single link."""
        links = self.get_links(link.canonical_id)
        updated = False
        for idx, existing in enumerate(links):
            if existing.source == link.source and existing.source_key == link.source_key:
                link.created_at = existing.created_at
                link.updated_at = datetime.utcnow()
                links[idx] = link
                updated = True
                break
        if not updated:
            links.append(link)
        self._write_links(link.canonical_id, links)

    def save_links(self, canonical_id: str, links: list[IdentityLink]) -> None:
        """Overwrite links for a canonical ID."""
        self._write_links(canonical_id, links)

    def get_links(self, canonical_id: str) -> list[IdentityLink]:
        """Get all links for a canonical ID."""
        path = self._links_path / f"{canonical_id}.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        return [IdentityLink(**item) for item in data]

    def list_canonical_ids(self) -> list[str]:
        """List all canonical IDs."""
        return [p.stem for p in self._links_path.glob("*.json")]

    def get_source_keys(
        self, canonical_ids: list[str], source: str
    ) -> dict[str, str]:
        """Map canonical IDs to source keys for a specific source."""
        mapping: dict[str, str] = {}
        for canonical_id in canonical_ids:
            for link in self.get_links(canonical_id):
                if link.source == source:
                    mapping[canonical_id] = link.source_key
        return mapping

    def get_canonical_ids(
        self, source: str, source_keys: list[str]
    ) -> dict[str, str]:
        """Map source keys to canonical IDs."""
        target_keys = set(source_keys)
        mapping: dict[str, str] = {}
        for canonical_id in self.list_canonical_ids():
            for link in self.get_links(canonical_id):
                if link.source == source and link.source_key in target_keys:
                    mapping[link.source_key] = canonical_id
        return mapping

    def save_job(self, job: ResolutionJob) -> None:
        """Save resolution job state."""
        path = self._jobs_path / f"{job.job_id}.json"
        path.write_text(job.model_dump_json(indent=2))

    def get_job(self, job_id: str) -> Optional[ResolutionJob]:
        """Get resolution job state."""
        path = self._jobs_path / f"{job_id}.json"
        if not path.exists():
            return None
        return ResolutionJob(**json.loads(path.read_text()))

    def _write_links(self, canonical_id: str, links: list[IdentityLink]) -> None:
        path = self._links_path / f"{canonical_id}.json"
        path.write_text(
            json.dumps(
                [link.model_dump(mode="json") for link in links],
                indent=2,
            )
        )
