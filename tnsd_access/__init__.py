"""tnsd_access — versioned EEG epoch loading utilities."""

from .trial_loader import TrialHandler
from .utilities import init_dataset

__all__ = ["TrialHandler", "build_trial_metadata"]
