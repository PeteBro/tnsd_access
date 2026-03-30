"""MRI data access — fMRI derivatives, raw BOLD, and anatomical images."""

from ._functional import FMRIHandler
from ._anatomical import AnatHandler


class MRIHandler:
    """MRI modality accessor.

    Sub-modalities are accessed via :attr:`fmri` and :attr:`anat`.

    Parameters
    ----------
    root : str or Path
        Dataset root directory.
    """

    def __init__(self, root):
        self.root = root

    @property
    def fmri(self) -> FMRIHandler:
        """fMRI derivatives and raw BOLD (:class:`FMRIHandler`)."""
        if not hasattr(self, '_fmri'):
            self._fmri = FMRIHandler(self.root)
        return self._fmri

    @property
    def anat(self) -> AnatHandler:
        """Anatomical images (:class:`AnatHandler`)."""
        if not hasattr(self, '_anat'):
            self._anat = AnatHandler(self.root)
        return self._anat


__all__ = ['MRIHandler']
