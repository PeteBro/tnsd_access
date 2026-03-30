"""Anatomical MRI handler (T1w, T2w).

Anatomicals are loaded directly from BIDS rawdata — there is no versioned
derivatives layer for anatomical images.
"""


class AnatHandler:
    """Load anatomical MRI images (T1w, T2w).

    Files are expected at::

        <root>/sub-<XX>/ses-mri<YY>/anat/sub-<XX>_ses-mri<YY>_T1w.nii.gz
        <root>/sub-<XX>/ses-mri<YY>/anat/sub-<XX>_ses-mri<YY>_T2w.nii.gz

    Parameters
    ----------
    root : Path
        Dataset root directory.
    """

    def __init__(self, root):
        self.root = root

    def get_t1(self, subject: int):
        """Load the T1w NIfTI image for a subject.

        Parameters
        ----------
        subject : int

        Returns
        -------
        nibabel.Nifti1Image
        """
        raise NotImplementedError

    def get_t2(self, subject: int):
        """Load the T2w NIfTI image for a subject.

        Parameters
        ----------
        subject : int

        Returns
        -------
        nibabel.Nifti1Image
        """
        raise NotImplementedError
