"""Phase 2 ingestion helpers for the frozen MVP."""

from importlib import import_module

__all__ = ["Phase2Artifacts", "build_phase2_artifacts", "preview_phase2_build", "write_phase2_artifacts"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(".phase2", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
