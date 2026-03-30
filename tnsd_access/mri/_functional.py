"""fMRI derivative and raw BOLD handlers."""

from typing import Iterator

import pandas as pd

from .._base import DerivativesHandler, RawHandler


class FMRIRawHandler(RawHandler):
    """Load raw BOLD NIfTI recordings.

    Raw BOLD files are expected at::

        <root>/sub-<XX>/ses-mri<YY>/func/sub-<XX>_ses-mri<YY>_task-<task>_run-<ZZ>_bold.nii.gz

    Parameters
    ----------
    root : Path
        Dataset root directory.
    """

    def get_recording(self, subject: int, session: int, run: int):
        """Load a raw BOLD NIfTI file.

        Parameters
        ----------
        subject : int
        session : int
        run : int

        Returns
        -------
        nibabel.Nifti1Image
        """
        raise NotImplementedError


class FMRIHandler(DerivativesHandler):
    """Load fMRI derivatives (beta maps, preprocessed BOLD) from versioned Zarr datastores.

    The dataset is expected to follow this layout::

        dataset_root/
        └── derivatives/fmri/<version>/
            ├── metadata.tsv
            └── sub-XX/
                └── chunk-XX/   ← zarr stores

    Access raw BOLD timeseries via :attr:`raw`.

    Parameters
    ----------
    root : Path
        Dataset root directory.
    version : str
        Derivative version to load (default ``'v1'``).
    """

    def __init__(self, root, version: str = 'v1'):
        super().__init__(root, version)
        self.datastore = root / 'derivatives' / 'fmri' / self.version / 'datastore'

    @property
    def raw(self) -> FMRIRawHandler:
        """Access raw BOLD recordings (:class:`FMRIRawHandler`)."""
        if not hasattr(self, '_raw'):
            self._raw = FMRIRawHandler(self.root)
        return self._raw

    def lookup_trials(self, **filters) -> pd.DataFrame:
        """Return fMRI derivative items matching the given metadata criteria.

        Mirrors the interface of :meth:`tnsd_access.eeg.EEGHandler.lookup_trials`.
        """
        raise NotImplementedError

    def get_data(self, trials: pd.DataFrame = None, **kwargs) -> dict:
        """Load fMRI derivative data (beta maps / preprocessed BOLD) into memory.

        Returns
        -------
        dict
            ``{'data': np.ndarray, 'metadata': pd.DataFrame}``
        """
        raise NotImplementedError

    def iter_data(self, trials: pd.DataFrame = None,
                  batch_size: int = 1000, **kwargs) -> Iterator[dict]:
        """Iterate over fMRI derivative data in memory-friendly batches.

        Yields
        ------
        dict
            Same structure as :meth:`get_data`.
        """
        raise NotImplementedError
