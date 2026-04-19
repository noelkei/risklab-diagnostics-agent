"""Phase 6 evaluation utilities for the frozen MVP."""

from .runner import EvalDatasetError, EvalRunnerError

__all__ = ["EvalDatasetError", "EvalRunnerError", "load_eval_questions", "main", "run_eval"]


def load_eval_questions(*args, **kwargs):
    from .runner import load_eval_questions as _load_eval_questions

    return _load_eval_questions(*args, **kwargs)


def run_eval(*args, **kwargs):
    from .runner import run_eval as _run_eval

    return _run_eval(*args, **kwargs)


def main(*args, **kwargs):
    from .runner import main as _main

    return _main(*args, **kwargs)
