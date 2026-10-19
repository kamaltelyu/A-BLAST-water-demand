"""Backward-compatible preprocessing imports for final v3.2."""
from .pipeline import chronological_split, stl_history_only, make_supervised, validate_daily_index
from .data import acquire_datasets, dataset_profile, sha256_file
__all__ = ["chronological_split", "stl_history_only", "make_supervised", "validate_daily_index", "acquire_datasets", "dataset_profile", "sha256_file"]
