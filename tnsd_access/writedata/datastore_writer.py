"""
Writes a preprocessing endpoint (MNE epoch files) into a partitioned zarr
datastore for fast random access.

Typical usage
-------------
writer = DatastoreWriter(
    root='/path/to/epo_runs',
    out='/path/to/output_epochs',
)
writer.write()
"""

import glob
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import mne
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from tqdm import tqdm

mne.set_log_level('ERROR')

# Path to the stimulus table, relative to root
STIM_RELPATH = 'stimuli/subject_stimuli.tsv'


class DatastoreWriter:
    """Convert a preprocessing endpoint of MNE epoch files into a zarr datastore.

    Parameters
    ----------
    root : str
        Root directory containing ``*_epo.fif`` files (searched recursively).
        A stimulus table is expected at ``<root>/<STIM_RELPATH>``.
    out : str
        Output directory where zarr chunks and metadata will be written.
    chunk_size : float
        Target size in GB for each zarr chunk (default 2.0).
    inner_chunk_size : int
        Number of epochs per zarr inner chunk along the first axis (default 500).
    """

    def __init__(self, root, out, chunk_size=2.0, inner_chunk_size=500):
        self.root = root
        self.out = out
        self.chunk_size = chunk_size
        self.inner_chunk_size = inner_chunk_size

        self._n_channels = None
        self._n_samples = None
        self._batches = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_store(self, path, shape):
        return zarr.open(
            path,
            zarr_format=2,
            mode='w',
            shape=shape,
            chunks=(self.inner_chunk_size, self._n_channels, self._n_samples),
            dtype='float32',
            compressor=Blosc(cname='lz4', clevel=1, shuffle=Blosc.BITSHUFFLE),
        )

    @staticmethod
    def _get_size(path):
        size = sum(os.path.getsize(f.path) for f in os.scandir(path))
        return size * 1e-9

    def _flush_batch(self, current_batch, subject, chunk):
        path = os.path.join(
            self.out,
            f"sub-{str(subject).zfill(2)}",
            f"chunk-{str(chunk).zfill(2)}",
        )
        batch_df = pd.concat(current_batch)
        batch_df['path'] = path
        batch_df['array_index'] = np.arange(len(batch_df))
        self._batches.append(batch_df)

    @staticmethod
    def _read_mne(path):
        return mne.read_epochs(path, preload=False, verbose=False)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def write(self):
        """Run the full write pipeline."""
        os.makedirs(self.out, exist_ok=True)

        # --- Discover epoch files and infer data dimensions ---
        all_paths = sorted(
            glob.glob(os.path.join(self.root, '**', '*_epo.fif'), recursive=True)
        )
        if not all_paths:
            raise FileNotFoundError(
                f"No *_epo.fif files found under root: {self.root}"
            )

        epo_files = [
            mne.read_epochs(path, preload=False, verbose=False)
            for path in tqdm(all_paths, desc='Reading epoch files')
        ]

        _ref = epo_files[0].load_data()
        self._n_channels = _ref._data.shape[1]
        self._n_samples = _ref._data.shape[2]

        # --- Build unified metadata table ---
        all_metadata = pd.concat(
            [epochs.metadata for epochs in epo_files], ignore_index=True
        )
        all_metadata['condition'] = np.concatenate(
            [epochs.events[:, -1] for epochs in epo_files]
        )
        all_metadata['mne_path'] = np.concatenate(
            [[all_paths[idx]] * len(epo) for idx, epo in enumerate(epo_files)]
        )

        stim_path = os.path.join(self.root, STIM_RELPATH)
        if os.path.exists(stim_path):
            stim_table = pd.read_csv(stim_path, sep='\t')
            s1000 = stim_table.query('s1000 == True')['nsd-id'].to_numpy()
            all_metadata['shared'] = np.isin(all_metadata['condition'], s1000)
            sort_cols = ['shared', 'subject', 'mne_path', 'condition',
                         'session', 'run', 'trial_instance']
        else:
            sort_cols = ['subject', 'mne_path', 'condition',
                         'session', 'run', 'trial_instance']

        write_order = all_metadata.sort_values(sort_cols)
        del epo_files

        # --- Estimate rows per chunk from a temporary store ---
        tmp_path = os.path.join(self.out, '_tmp_zarr')
        _tmp = self._init_store(tmp_path, _ref._data.shape)
        _tmp[:] = _ref._data
        size_per_epoch = self._get_size(tmp_path) / len(_ref)
        rows_per_chunk = max(1, int(self.chunk_size / size_per_epoch))
        shutil.rmtree(tmp_path)

        # --- Partition epochs into size-bounded batches ---
        self._batches = []
        for subject, subj_df in write_order.groupby('subject', sort=True):
            chunk = 0
            current_batch, current_size = [], 0
            for _, group in subj_df.groupby('condition', sort=False):
                current_batch.append(group)
                current_size += len(group)
                if current_size >= rows_per_chunk:
                    chunk += 1
                    self._flush_batch(current_batch, subject, chunk)
                    current_batch, current_size = [], 0
            if current_batch:
                chunk += 1
                self._flush_batch(current_batch, subject, chunk)

        # --- Write zarr datastores ---
        status = tqdm(bar_format='{desc}', position=1, leave=False)
        for batch in tqdm(self._batches, desc='Filling Datastores', position=0):
            path = batch['path'].unique()[0]
            filestore = self._init_store(path, shape=(len(batch), self._n_channels, self._n_samples))

            mne_files = np.sort(batch['mne_path'].unique())
            batch = batch.sort_values(['mne_path', 'epoch'])

            status.set_description_str('Reading runs...')
            with ThreadPoolExecutor() as pool:
                epoch_data = list(pool.map(self._read_mne, mne_files))

            status.set_description_str('Sorting metadata...')
            epoch_metadata = pd.concat([e.metadata for e in epoch_data])
            epoch_metadata['idx'] = np.concatenate(
                [[idx] * len(f) for idx, f in enumerate(epoch_data)]
            )

            metadata_events = epoch_metadata['event_id'].to_numpy()
            sorter = np.argsort(metadata_events)
            trial_positions = sorter[
                np.searchsorted(metadata_events, batch['event_id'].to_numpy(), sorter=sorter)
            ]
            fileidcs = epoch_metadata['idx'].iloc[trial_positions]
            epoidcs = epoch_metadata['epoch'].iloc[trial_positions] - 1

            lookup_idcs = defaultdict(list)
            for fileidx, epoidx in zip(fileidcs, epoidcs):
                lookup_idcs[fileidx].append(epoidx)

            status.set_description_str('Getting epochs...')
            inner = tqdm(lookup_idcs.items(), desc='Fetching Epoch Arrays', position=2, leave=False)
            data_array = np.concatenate(
                [epoch_data[key][value].get_data() for key, value in inner]
            )

            data_array = data_array[np.argsort(batch['array_index'].to_numpy())]
            filestore[:] = data_array
            batch = batch.drop(columns='mne_path')
            filestore.attrs.put(batch.reset_index(drop=True).to_dict(orient='list'))

        # --- Write global metadata ---
        metadata = (
            write_order
            .sort_values(['subject', 'session', 'run', 'epoch'], kind='mergesort')
            .drop(columns='mne_path')
        )
        metadata.to_csv(os.path.join(self.out, 'epoch_metadata.tsv'), sep='\t', index=False)
