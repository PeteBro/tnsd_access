"""Modality mixins that compose the Dataset class.

Each mixin contributes one lazily-initialised accessor property.  Adding a new
modality in the future is as simple as writing a new mixin and adding it to the
Dataset class line in __init__.py.
"""


class EEGMixin:
    """Adds the ``.eeg`` accessor to Dataset."""

    @property
    def eeg(self):
        if not hasattr(self, '_eeg'):
            from .eeg import EEGHandler
            self._eeg = EEGHandler(self.root)
        return self._eeg


class MRIMixin:
    """Adds the ``.mri`` accessor to Dataset."""

    @property
    def mri(self):
        if not hasattr(self, '_mri'):
            from .mri import MRIHandler
            self._mri = MRIHandler(self.root)
        return self._mri


class StimuliMixin:
    """Adds the ``.stimuli`` accessor to Dataset."""

    @property
    def stimuli(self):
        if not hasattr(self, '_stimuli'):
            from .stimuli import StimuliHandler
            self._stimuli = StimuliHandler(self.root)
        return self._stimuli


class RDMMixin:
    """Adds the ``.rdm`` accessor to Dataset.

    Sub-accessors ``.rdm.eeg`` and ``.rdm.fmri`` each return an
    :class:`~tnsd_access.rdm.RDMHandler` whose version tracks the
    corresponding modality handler.  Call ``ds.rdm.reset()`` after changing
    a modality version to refresh the cached handlers.
    """

    @property
    def rdm(self):
        if not hasattr(self, '_rdm'):
            from .rdm import RDMAccessor
            self._rdm = RDMAccessor(self)
        return self._rdm
