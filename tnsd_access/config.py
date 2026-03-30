
BUCKET = 'temporal-natural-scenes-dataset'

# Top-level BIDS dataset files, independent of any modality.
_DATASET = [
    'dataset_description.json',
    'CITATION.cff',
    'LICENSE',
    'README.md',
    'CHANGES',
]

_EEG = [
    'derivatives/eeg/v1/datastore/manifest.json',
    'derivatives/eeg/v1/datastore/metadata.tsv',
    'sub-01/sub-01_sessions.tsv',
    'sub-01/sub-02_sessions.tsv',
    'sub-01/sub-03_sessions.tsv',
    'sub-01/sub-04_sessions.tsv',
    'sub-01/sub-05_sessions.tsv',
    'sub-01/sub-06_sessions.tsv',
    'sub-01/sub-07_sessions.tsv',
    'sub-01/sub-08_sessions.tsv',
]

_MRI = []      # TODO: add MRI seed files

_STIMULI = []  # TODO: add stimuli seed files

SEED_FILES = {
    'dataset': _DATASET,
    'eeg':     _EEG,
    'mri':     _MRI,
    'stimuli': _STIMULI,
}
