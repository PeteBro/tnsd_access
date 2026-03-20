"""Trial metadata lookup and data loader."""

import os
import glob
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm
from pathlib import Path
from ..utilities import build_trial_metadata, resolve_dir, check_islocal, check_stale, fetch_remote

BUCKET = 'temporal-natural-scenes-dataset'

class TrialHandler:

    """Load and selected EEG trial data from a zarr datastores.

    Point this class at your dataset root folder and tell it which data version
    you want to work with.  It will find the matching metadata table
    automatically, giving you a simple interface to select and load trials.

    The dataset is expected to follow this layout::

        dataset_root/
        └── .../ (any depth)
            └── <version>/
                ├── *metadata.tsv
                └── sub-XX/
                    └── chunk-XX/   ← zarr stores

    Parameters
    ----------
    dataset_root : str
        Top-level folder of your dataset (e.g. ``'/data/nsdBIDS'``).
    version : str
        Name of the data version directory to load (e.g. ``'preproc_1'``).
        The directory can sit anywhere under *dataset_root* — it will be 
        located automatically.

    Examples
    --------
    >>> loader = TrialHandler('/data/temporal-natural-scenes-dataset', version='v1')

    >>> # Load a specific subset by filtering inline
    >>> result = loader.get_data(subject=1, condition=[5, 2951])

    >>> # Or look up a trial table first, then load
    >>> trials = loader.lookup_trials(subject=1)
    >>> result = loader.get_data(trials)
    >>> result['data'].shape   # (n_trials, n_channels, n_samples)
    """

#
    def __init__(self, dataset_root: str = 'temporal-natural-scenes-dataset', version: str = 'v1'):
        """Resolve paths for reading datastore and initialize store cache for fast reading."""
#
        global BUCKET

        print('Resolving path...')
        self.root = resolve_dir(dataset_root, makedir=True)
        self.datastore = self.root / 'derivatives' / version / 'datastore'
        print('Reading metadata...')
        self.metadata = pd.read_csv(self.datastore / 'metadata.tsv', sep='\t', index_col=False)
        self.metadata['path'] = self.metadata['path'].apply(lambda p: (self.datastore / p).resolve())
        self.store_cache = {}
        print('Done.')

#
    def lookup_trials(self, cond='and', **filters) -> pd.DataFrame:
        """Return trials matching the given metadata criteria.

        Pass any metadata column as a keyword argument to filter trials.
        Multiple filters are combined with ``cond='and'`` (all criteria must
        match) or ``cond='or'`` (any criterion is enough).  The returned
        DataFrame can be passed directly to :meth:`get_data` or
        :meth:`iter_data`, or inspected before loading.

        Parameters
        ----------
        cond : {'and', 'or'}, optional
            How to combine multiple filters.  ``'and'`` (default) keeps only
            trials that satisfy **all** filters; ``'or'`` keeps trials that
            satisfy **at least one**.
        **filters
            Column name / value pairs.  The value can be a single item or a
            list.  For example ``condition=[1, 2, 3]`` keeps only trials whose
            ``condition`` column is 1, 2, or 3.

        Other Parameters
        ----------------
        subject : int or list of int
            Subject number(s) to include.  Valid values are integers 1–18.
        condition : int or list of int
            NSD stimulus ID(s) to include.
        session : int or list of int
            Session number(s) to include.
        run : int or list of int
            Run number(s) within a session to include.
        epoch : int or list of int
            Epoch number(s) within a run to include.
        trial_instance : int or list of int
            Repetition index for a stimulus within a session.  Useful for
            selecting first-presentation trials only (``trial_instance=1``).
        trial_type : str or list of str
            Trial type label(s) to include.
        shared : bool
            If ``True``, keep only the 1000 stimuli shared across all subjects.
            If ``False``, keep only subject-unique stimuli.
        onset : float or list of float
            Trial onset time(s) in seconds.
        duration : float or list of float
            Trial duration(s) in seconds.
        response_time : float or list of float
            Participant response time(s) in seconds.
        response : str or list of str
            Participant response value(s).
        stim_file : str or list of str
            Stimulus filename(s).
        trigger_value : int or list of int
            Trigger value(s) sent at trial onset.
        event_id : int or list of int
            MNE event ID(s).
        datetime : str or list of str
            Datetime string(s) of trial onset.

        Returns
        -------
        pd.DataFrame
            Filtered metadata table, sorted by the filter columns and with a
            fresh integer index.  Pass this directly to :meth:`get_data`.

        Examples
        --------
        >>> # All shared trials for subject 1
        >>> trials = loader.lookup_trials(subject=1, shared=True)

        >>> # Trials from subject 1 OR subject 2
        >>> trials = loader.lookup_trials(cond='or', subject=[1, 2])
        """
        mask = pd.DataFrame(
            np.full(self.metadata.shape, True), columns=self.metadata.columns
        )
        for col, vals in filters.items():
            if not isinstance(vals, (list, tuple, np.ndarray)):
                vals = [vals]
            mask[col] &= self.metadata[col].isin(vals)
        mask = mask.to_numpy()
        if cond == 'or':
            mask = np.any(mask, axis=1)
        elif cond == 'and':
            mask = np.all(mask, axis=1)
        else:
            raise ValueError('Invalid cond: must be "and" or "or"')

        trials = self.metadata[mask].sort_values(list(filters.keys())).reset_index(drop=True)

        local_status = check_islocal(trials['path'].unique())
        missing = [p for p, local in local_status.items() if not local]
        present = [p for p, local in local_status.items() if local]

        if missing:
            ans = input(f'{len(missing)} data store(s) not found locally. Download from remote? [y/n] ')
            if ans.strip().lower() == 'y':
                fetch_remote(missing, BUCKET, self.root)
            else:
                trials = trials[~trials['path'].isin(missing)].reset_index(drop=True)

        #stale = check_stale(present, BUCKET, self.root)
        #if stale:
        #    ans = input(f'{len(stale)} data store(s) are inconsistent with remote. Update from remote? [y/n] ')
        #    if ans.strip().lower() == 'y':
        #        fetch_remote(stale, BUCKET, self.root)

        return trials

#
    def get_data( # TODO: Add sample index option as well as times - also maybe return an object or something / mne
        self,
        trials: pd.DataFrame = None,
        channels=None,
        tmin: float = None,
        tmax: float = None,
        step = None,
        sample_idcs=None,
        average_by=None,
        verbose=True,
        cond='and',
        **filters,
    ) -> dict:
        """Load EEG data into memory, with optional inline trial filtering.

        Reads the EEG arrays from disk and returns them as a NumPy array
        together with the corresponding metadata.  Zarr stores are cached
        after the first open, so repeated calls for trials from the same
        store are fast.

        You can supply trials three ways:

        * Pass a pre-built ``trials`` DataFrame (e.g. from
          :meth:`lookup_trials`).
        * Pass filter keyword arguments directly — :meth:`lookup_trials` is
          called internally.
        * Pass neither — all trials in the metadata table are loaded.

        Parameters
        ----------
        trials : pd.DataFrame, optional
            Trial metadata table.  Must contain ``path`` and ``array_index``
            columns.  When omitted, ``**filters`` (if any) are used to build
            the table automatically via :meth:`lookup_trials`.
        channels : list, optional
            Channels to load.  Can be a list of integer indices or channel
            name strings.  When omitted, all channels are returned.
        tmin : float, optional
            Start of the time window in seconds.  Trials are cropped to
            ``[tmin, tmax]`` before being returned.  When omitted, the full
            epoch is returned.
        tmax : float, optional
            End of the time window in seconds.  See ``tmin``.
        step : int, optional
            Sample step size for downsampling.  ``step=2`` returns every other
            sample, halving the time resolution.  Default is ``None`` (no
            downsampling).
        average_by : str or list of str, optional
            Metadata column(s) to average over.  For example
            ``average_by='condition'`` returns one averaged waveform per
            condition instead of one waveform per trial.
        verbose : bool, optional
            Show a progress bar while loading.  Default ``True``.
        cond : {'and', 'or'}, optional
            How to combine multiple ``**filters`` (passed to
            :meth:`lookup_trials`).  Ignored when ``trials`` is provided
            explicitly.  Default ``'and'``.
        **filters
            Column / value pairs forwarded to :meth:`lookup_trials` when
            ``trials`` is not provided.

        Returns
        -------
        dict
            A dictionary with two keys:

            ``'data'``
                NumPy array of shape ``(n_trials, n_channels, n_samples)``
                (or ``(n_groups, n_channels, n_samples)`` when
                ``average_by`` is set), dtype ``float32``.
            ``'metadata'``
                DataFrame with one row per trial (or per group when
                ``average_by`` is set), aligned to the first axis of
                ``'data'``.

        Examples
        --------
        >>> # Inline filtering — no separate lookup_trials call needed
        >>> result = loader.get_data(subject=1, shared=True)
        >>> eeg = result['data']    # shape: (n_trials, n_channels, n_samples)

        >>> # Pass a pre-built trial table
        >>> trials = loader.lookup_trials(conditions=[1, 2, 3])
        >>> result = loader.get_data(trials)

        >>> # Average across trials, grouped by condition
        >>> result = loader.get_data(shared=True, average_by='condition')
        """
        if trials is None:
            trials = self.lookup_trials(cond=cond, **filters) if filters else self.metadata.copy()

        stores = trials['path'].unique()
        for path in stores:
            if path not in self.store_cache.keys():
                self.store_cache[path] = zarr.open(path, mode='r')

        store0 = self.store_cache[stores[0]]
        info = store0.attrs['info']
        times = np.asarray(store0.attrs['times'])
        channel_names = info['ch_names']

        if channels is None:
            channels = slice(None)
        else:
            channels = np.array([channel_names.index(c) if isinstance(c, str) else c for c in channels])
            if channels.size > 1 and np.all(np.diff(channels) == channels[1] - channels[0]):
                channels = slice(channels[0], channels[-1] + 1, channels[1] - channels[0])

        if sample_idcs is not None:
            samples = np.asarray(sample_idcs)
        else:
            tmin_idx = 0 if tmin is None else np.abs(times - tmin).argmin()
            tmax_idx = len(times) if tmax is None else np.abs(times - tmax).argmin()
            samples = slice(tmin_idx, tmax_idx, step)

        # Sort by store then array_index for sequential chunk access;
        # record original row position so output order matches input trials
        ordered = trials[['path', 'array_index']].copy()
        ordered['out_row'] = np.arange(len(trials))
        ordered = ordered.sort_values(['path', 'array_index'])
    
        samplearr = store0.oindex[0, channels, samples]
        n_channels, n_samples = samplearr.shape
        data_array = np.empty((len(trials), n_channels, n_samples), dtype='float32')
    
        with tqdm(total=len(trials), desc='Loading Trials', disable=not verbose) as prog:
            for path, group in ordered.groupby('path', sort=False):
                store = self.store_cache[path]
                arr_idcs = group['array_index'].to_numpy()
                out_rows = group['out_row'].to_numpy()
                data_array[out_rows] = store.oindex[arr_idcs, channels, samples]
                prog.update(len(out_rows))
    #
        meta = trials.reset_index(drop=True)
    #
        if average_by is not None:
            keys = [average_by] if isinstance(average_by, str) else list(average_by)
            groups = meta.groupby(keys, sort=False)
            data_array = np.stack([data_array[grp.index].mean(axis=0) for _, grp in groups])
            meta = (groups.agg(lambda col: col.iloc[0] if col.nunique() == 1 else np.nan)
                         .drop(columns=['path', 'array_index'], errors='ignore')
                         .dropna(axis=1)
                         .reset_index())
    #
        return {"data": data_array, "metadata": meta}

#
    def iter_data(
        self,
        trials: pd.DataFrame = None,
        batch_size: int = 64,
        channels=None,
        tmin: float = None,
        tmax: float = None,
        average_by=None,
        sort_lookup=True,
        cond='and',
        **filters,
    ):
        """Iterate over trials in memory-friendly batches.

        Yields successive chunks of loaded EEG data instead of loading
        everything at once.  Useful when your full trial set is too large to
        fit in RAM, or when you want to feed a model batch-by-batch.

        Each yielded item has the same structure as the dict returned by
        :meth:`get_data`: a ``'data'`` array and a ``'metadata'`` DataFrame.

        When ``average_by`` is set, the iterator guarantees that all trials
        belonging to the same group are included in the same batch before
        averaging — groups are never split across batches.

        As with :meth:`get_data`, you can pass a pre-built ``trials`` table,
        supply ``**filters`` to build one inline, or omit both to iterate
        over all trials.

        Parameters
        ----------
        trials : pd.DataFrame, optional
            Trial metadata table.  When omitted, ``**filters`` (if any) are
            used to build it via :meth:`lookup_trials`, or all trials are
            used if no filters are given.
        batch_size : int, optional
            Maximum number of trials (or group rows) to load per batch.
            Default is 64.
        channels : list, optional
            Channels to load (integer indices or name strings).  All channels
            are loaded when omitted.
        tmin : float, optional
            Start of the time window in seconds.  When omitted, the full epoch
            is returned.
        tmax : float, optional
            End of the time window in seconds.  See ``tmin``.
        average_by : str or list of str, optional
            Metadata column(s) to average over within each batch.
        sort_lookup : bool, optional
            Sort trials by store path and array index before iterating for
            more efficient sequential disk reads.  Default ``True``.
        cond : {'and', 'or'}, optional
            How to combine multiple ``**filters``.  Ignored when ``trials``
            is provided explicitly.  Default ``'and'``.
        **filters
            Column / value pairs forwarded to :meth:`lookup_trials` when
            ``trials`` is not provided.

        Yields
        ------
        dict
            Same structure as :meth:`get_data`: ``{'data': np.ndarray,
            'metadata': pd.DataFrame}``.

        Examples
        --------
        >>> # Inline filtering
        >>> for batch in loader.iter_data(subject=1, batch_size=32):
        ...     eeg = batch['data']   # shape: (<=32, n_channels, n_samples)
        ...     meta = batch['metadata']
        ...     process(eeg, meta)

        >>> # Iterate with per-stimulus averaging
        >>> trials = loader.lookup_trials(shared=True)
        >>> for batch in loader.iter_data(trials, batch_size=64, average_by='subject'):
        ...     process(batch['data'], batch['metadata'])
        """
        if trials is None:
            trials = self.lookup_trials(cond=cond, **filters) if filters else self.metadata.copy()

        keys = ([average_by] if isinstance(average_by, str) else list(average_by)) if average_by else None
#
        if sort_lookup:
            trials = trials.sort_values(['path', 'array_index'])
#
        if keys:
            # accumulate complete groups into batches, never splitting a group
            batch, count = [], 0
            for _, grp in trials.groupby(keys, sort=False):
                if count + len(grp) > batch_size and batch:
                    yield self.get_data(pd.concat(batch), channels=channels,
                                        tmin=tmin, tmax=tmax, average_by=keys,
                                        verbose=False)
                    batch, count = [], 0
                batch.append(grp)
                count += len(grp)
            if batch:
                yield self.get_data(pd.concat(batch), channels=channels,
                                    tmin=tmin, tmax=tmax, average_by=keys,
                                    verbose=False)
        else:
            for start in range(0, len(trials), batch_size):
                yield self.get_data(
                    trials.iloc[start:start + batch_size],
                    channels=channels, tmin=tmin, tmax=tmax, verbose=False
                )
