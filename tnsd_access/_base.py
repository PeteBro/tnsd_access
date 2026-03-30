"""Abstract base classes for modality data handlers."""

from abc import ABC, abstractmethod
from typing import Iterator

import pandas as pd


class DerivativesHandler(ABC):
    """Base class for versioned derivative data handlers (EEG epochs, fMRI betas, etc.).

    Parameters
    ----------
    root : Path
        Dataset root directory.
    version : str
        Derivative version to load (e.g. ``'v1'``).
    """

    def __init__(self, root, version: str = 'v1'):
        self.root = root
        self.version = version

    @abstractmethod
    def lookup_trials(self, **filters) -> pd.DataFrame:
        """Return a metadata DataFrame for matching trials or items."""
        ...

    @abstractmethod
    def get_data(self, trials: pd.DataFrame = None, **kwargs) -> dict:
        """Load data into memory.

        Returns
        -------
        dict
            ``{'data': np.ndarray, 'metadata': pd.DataFrame}``
        """
        ...

    @abstractmethod
    def iter_data(self, trials: pd.DataFrame = None,
                  batch_size: int = 1000, **kwargs) -> Iterator[dict]:
        """Iterate over data in memory-friendly batches.

        Yields
        ------
        dict
            Same structure as :meth:`get_data`.
        """
        ...


class RawHandler(ABC):
    """Base class for raw (unprocessed) recording handlers.

    Parameters
    ----------
    root : Path
        Dataset root directory.
    """

    def __init__(self, root):
        self.root = root

    @abstractmethod
    def get_recording(self, subject: int, session: int, run: int):
        """Load a raw recording for the given subject / session / run."""
        ...
