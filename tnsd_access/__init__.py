"""tnsd_access — multimodal dataset access for the Temporal Natural Scenes Dataset."""

from pathlib import Path

from ._mixins import EEGMixin, MRIMixin, StimuliMixin, RDMMixin
from ._utils import init_dataset


class Dataset(EEGMixin, MRIMixin, StimuliMixin, RDMMixin):
    """Top-level entry point for the TNSD dataset.

    Modality handlers are instantiated lazily on first access.  If *root*
    does not exist, the user is prompted to initialise the dataset from S3,
    which downloads seed metadata for all modalities.

    Parameters
    ----------
    root : str or Path
        Path to the dataset root directory.

    Examples
    --------
    >>> ds = Dataset('/path/to/temporal-natural-scenes-dataset')

    >>> # EEG epochs (derivatives)
    >>> trials = ds.eeg.lookup_trials(subject=1, shared=True)
    >>> result  = ds.eeg.get_data(trials)

    >>> # EEG raw recordings
    >>> raw = ds.eeg.raw.get_recording(subject=1, session=1, run=1)

    >>> # fMRI derivatives (versioned)
    >>> result = ds.mri.fmri.get_data(subject=1)

    >>> # fMRI raw BOLD
    >>> bold = ds.mri.fmri.raw.get_recording(subject=1, session=1, run=1)

    >>> # Anatomical
    >>> t1 = ds.mri.anat.get_t1(subject=1)

    >>> # Stimuli
    >>> images = ds.stimuli.get_images(category='face')

    >>> # RDMs (version tracks the corresponding modality handler)
    >>> store  = ds.rdm.eeg.load(subject=1, complete_only=True)
    >>> traj   = ds.rdm.eeg.get_pair(subject=1, stim_a=101, stim_b=204)
    >>> matrix = ds.rdm.eeg.get_matrix(subject=1, time=0.1)
    >>> store  = ds.rdm.fmri.load(subject=1)
    """

    def __init__(self, root: str):
        self.root = Path(root)
        if not self.root.exists():
            ans = input(
                f"Dataset not found at '{self.root}'. "
                f"Initialise from S3? [y/n] "
            )
            if ans.strip().lower() == 'y':
                init_dataset(self.root)
            else:
                raise RuntimeError(f"Dataset root not found: '{self.root}'")


__all__ = ['Dataset', 'init_dataset']
