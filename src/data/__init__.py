"""Market data adapters, scenario generators, and published benchmark datasets."""

from src.data.paper_datasets import (
    COVARIANCE,
    INSTRUMENTS,
    MEAN_RETURNS,
    MIN_VARIANCE_VARIANCE,
    MIN_VARIANCE_WEIGHTS,
    REFERENCE_CVAR,
    REFERENCE_VAR,
    TARGET_RETURN,
    sample_normal_returns,
)

__all__ = [
    "COVARIANCE",
    "INSTRUMENTS",
    "MEAN_RETURNS",
    "MIN_VARIANCE_VARIANCE",
    "MIN_VARIANCE_WEIGHTS",
    "REFERENCE_CVAR",
    "REFERENCE_VAR",
    "TARGET_RETURN",
    "sample_normal_returns",
]
