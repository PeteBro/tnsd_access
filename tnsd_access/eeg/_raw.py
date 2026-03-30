"""EEG raw recording handler (.bdf format via MNE)."""

from .._base import RawHandler


class EEGRawHandler(RawHandler):
    """Load raw continuous EEG recordings in BDF format via MNE.

    Raw recordings are expected at::

        <root>/sub-<XX>/ses-eeg<YY>/eeg/sub-<XX>_ses-eeg<YY>_task-<task>_run-<ZZ>_eeg.bdf

    Parameters
    ----------
    root : Path
        Dataset root directory.
    """

    def get_recording(self, subject: int, session: int, run: int):
        """Load a raw BDF recording.

        Parameters
        ----------
        subject : int
        session : int
        run : int

        Returns
        -------
        mne.io.Raw
        """
        raise NotImplementedError
