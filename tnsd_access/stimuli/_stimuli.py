"""Stimuli handler — images, captions, and categories.

Images are stored as a Zarr array of shape (n_stims, H, W, C).
Captions and categories are stored as TSV metadata files.

# TODO: This is a natural access point for cross-modal stimulus alignment.
# The 'shared' flag in EEG/fMRI trial metadata identifies the stimulus set
# shown to all subjects. Consider adding a get_shared() method or a helper
# that returns stimuli aligned to a given trial metadata table.
"""

import pandas as pd
import numpy as np


class StimuliHandler:
    """Load stimuli: images (Zarr), captions, and categories (TSV).

    Parameters
    ----------
    root : str or Path
        Dataset root directory.
    """

    def __init__(self, root):
        self.root = root

    def get_images(self, **filters) -> np.ndarray:
        """Load stimulus images.

        Parameters
        ----------
        **filters
            Metadata column / value pairs to filter stimuli (e.g.
            ``category='face'``).

        Returns
        -------
        np.ndarray
            Array of shape ``(n_stims, H, W, C)``, dtype ``uint8``.
        """
        raise NotImplementedError

    def get_captions(self, stim_id=None) -> pd.DataFrame:
        """Load captions for stimuli.

        Parameters
        ----------
        stim_id : list of int, optional
            Stimulus IDs to fetch. If ``None``, returns all captions.

        Returns
        -------
        pd.DataFrame
        """
        raise NotImplementedError

    def get_categories(self) -> pd.DataFrame:
        """Load stimulus category labels.

        Returns
        -------
        pd.DataFrame
        """
        raise NotImplementedError
