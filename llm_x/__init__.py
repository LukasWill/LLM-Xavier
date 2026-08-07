"""Public API for the LLM-X offline evaluator."""

from .data import Episode, EpisodeValidationError
from .evaluation import EvaluationConfig, EvaluationResult, evaluate_episode

__all__ = [
    "Episode",
    "EpisodeValidationError",
    "EvaluationConfig",
    "EvaluationResult",
    "evaluate_episode",
]

__version__ = "0.1.0"
