"""Compatibility imports; paper scoring lives in evaluation.metrics."""
from evaluation.metrics import aggregate, compare_results, score_depth

__all__ = ["aggregate", "compare_results", "score_depth"]
