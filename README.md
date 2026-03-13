# tnsd_access

[![Documentation Status](https://readthedocs.org/projects/nsd-access/badge/?version=latest)](https://nsd-access.readthedocs.io/en/latest/)

Python package for easy reading of the NSD-EEG processed data.

Full documentation: https://nsd-access.readthedocs.io/en/latest/

---

## Setup

**1. Clone the repo**
```bash
git clone git@github.com:PeteBro/tnsd_access.git
cd tnsd_access
```

**2. Install**
```bash
pip install -e .
```

---

## Usage

### Load trials for a subject

```python
from tnsd_access import TrialHandler

loader = TrialHandler()

# Look up trials for subject 6, filtering to conditions 5 and 2951
trials = loader.lookup_trials(subject=6, condition=[5, 2951])

# Load all matching trials into memory — returns {'data': ndarray, 'metadata': DataFrame}
result = loader.get_data(trials)
```

### Average by condition (ERP-style)

```python
from tnsd_access import TrialHandler

loader = TrialHandler()

# Look up all shared trials across subjects
trials = loader.lookup_trials(shared=True)

# Load data and average across trials within each condition
# result['data'] shape: (n_conditions, n_channels, n_samples)
# result['metadata'] has one row per condition
result = loader.get_data(trials, average_by='condition')
```

### Batch iteration (memory-efficient)

```python
from tnsd_access import TrialHandler

loader = TrialHandler()

# Look up trials for subject 6
trials = loader.lookup_trials(subject=6)

# Iterate in batches of 32 trials — useful for large datasets
for batch in loader.iter_data(trials, batch_size=32):
    data = batch['data']      # (batch_size, n_channels, n_samples)
    meta = batch['metadata']  # DataFrame aligned to data's first axis
```
