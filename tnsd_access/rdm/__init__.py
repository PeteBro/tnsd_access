"""RDM data access — modality-specific and cross-modal representational geometry.

Access via ``ds.rdm.eeg`` or ``ds.rdm.fmri``.  Each sub-accessor is a
:class:`~tnsd_access.rdm._rdm.RDMHandler` tied to the corresponding modality's
current derivative version.

Version tracking
----------------
``ds.rdm.eeg`` resolves ``ds.eeg.version`` on every access and returns a
cached :class:`~tnsd_access.rdm._rdm.RDMHandler` for that version.  Changing
``ds.eeg.version`` automatically routes subsequent accesses to a new handler;
call ``ds.rdm.reset()`` to also discard any previously cached handlers and
free their Zarr store caches.
"""

from ._rdm import RDMHandler


class RDMAccessor:
    """Top-level RDM accessor attached to a :class:`~tnsd_access.Dataset`.

    Provides ``.eeg`` and ``.fmri`` sub-accessors, each a
    :class:`~tnsd_access.rdm._rdm.RDMHandler` for the corresponding modality.
    Handlers are cached per version — changing a modality version and calling
    :meth:`reset` creates a fresh handler with a clean store cache.

    Parameters
    ----------
    dataset : Dataset
        The parent dataset instance.
    """

    def __init__(self, dataset):
        self._ds = dataset
        self._eeg_cache:  dict[str, RDMHandler] = {}
        self._fmri_cache: dict[str, RDMHandler] = {}

    def reset(self):
        """Clear cached handlers so version changes take effect."""
        self._eeg_cache.clear()
        self._fmri_cache.clear()

    @property
    def eeg(self) -> RDMHandler:
        """EEG RDMs — version follows ``ds.eeg.version``."""
        version = self._ds.eeg.version
        if version not in self._eeg_cache:
            self._eeg_cache[version] = RDMHandler(self._ds.root, version, modality='eeg')
        return self._eeg_cache[version]

    @property
    def fmri(self) -> RDMHandler:
        """fMRI RDMs — version follows ``ds.mri.fmri.version``."""
        version = self._ds.mri.fmri.version
        if version not in self._fmri_cache:
            self._fmri_cache[version] = RDMHandler(self._ds.root, version, modality='fmri')
        return self._fmri_cache[version]


__all__ = ['RDMAccessor', 'RDMHandler']
