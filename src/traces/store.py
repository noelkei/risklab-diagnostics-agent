from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app import ProjectSettings, load_settings
from schemas import TraceRecord


@dataclass(frozen=True)
class JsonlTraceStore:
    path: Path

    @classmethod
    def from_settings(cls, *, settings: ProjectSettings | None = None) -> "JsonlTraceStore":
        resolved_settings = settings or load_settings()
        return cls(path=resolved_settings.paths.trace_runs_path)

    def append(self, record: TraceRecord) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized_record = json.dumps(record.model_dump(mode="json"), ensure_ascii=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(serialized_record + "\n")
        return self.path
