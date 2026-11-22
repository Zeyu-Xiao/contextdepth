"""Compatibility imports; evaluation-grid projection lives in evaluation.geometry."""
from evaluation.geometry import EvaluationPrediction, camera_observation, project_prediction

__all__ = ["EvaluationPrediction", "camera_observation", "project_prediction"]
