# eeg_access

[![Documentation Status](https://readthedocs.org/projects/nsd-access/badge/?version=latest)](https://nsd-access.readthedocs.io/en/latest/)

Python package for easy reading of the NSD-EEG processed data.

Full documentation: https://nsd-access.readthedocs.io/en/latest/

---

## Setup

**1. Clone the repo**
```bash
git clone git@github.com:PeteBro/eeg_access.git
cd eeg_access
```

**2. Install**
```bash
pip install -e .
```

---

## Data access

The dataset is hosted on AWS S3. You need AWS credentials to access it.

### 1. Install the AWS CLI and configure credentials

Install the AWS CLI by following the [official instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), then run:

```bash
aws configure
```

Enter your AWS Access Key ID, Secret Access Key, and preferred region when prompted. These credentials are used by the package internally via `boto3`.

### 2. Get the dataset

**Option A — copy the full dataset with the AWS CLI**

If you want to mirror the entire dataset locally upfront:

```bash
aws s3 cp s3://temporal-natural-scenes-dataset/ /path/to/local/dataset/ --recursive
```

**Option B — initialise with the Python package (recommended)**

`init_dataset` downloads only the lightweight seed files needed to get started (metadata, manifests, session tables). Individual data stores are then fetched on demand the first time you query them.

```python
from eeg_access import init_dataset

init_dataset('/path/to/local/dataset')
```

This creates the dataset directory if it doesn't exist and pulls the metadata required for trial lookups. Once complete, point `TrialHandler` at the same root — any data stores not yet present locally will prompt you to download them automatically when queried.

---

## Usage

### Load trials for a subject

```python
from eeg_access import TrialHandler

loader = TrialHandler('/path/to/local/dataset')

# Look up trials for subject 6, filtering to conditions 5 and 2951
trials = loader.lookup_trials(subject=6, condition=[5, 2951])

# Load all matching trials into memory — returns {'data': ndarray, 'metadata': DataFrame}
result = loader.get_data(trials)
```

### Average by condition (ERP-style)

```python
from eeg_access import TrialHandler

loader = TrialHandler('/path/to/local/dataset')

# Look up all shared trials across subjects
trials = loader.lookup_trials(shared=True)

# Load data and average across trials within each condition
# result['data'] shape: (n_conditions, n_channels, n_samples)
# result['metadata'] has one row per condition
result = loader.get_data(trials, average_by='condition')
```

### Batch iteration (memory-efficient)

```python
from eeg_access import TrialHandler

loader = TrialHandler('/path/to/local/dataset')

# Look up trials for subject 6
trials = loader.lookup_trials(subject=6)

# Iterate in batches of 32 trials — useful for large datasets
for batch in loader.iter_data(trials, batch_size=32):
    data = batch['data']      # (batch_size, n_channels, n_samples)
    meta = batch['metadata']  # DataFrame aligned to data's first axis
```
