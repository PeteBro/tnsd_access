"""Trial metadata lookup and data loader."""

import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
from pathlib import Path
from .utilities import resolve_dir, check_islocal, fetch_remote

BUCKET = 'temporal-natural-scenes-dataset'

class TrialHandler:

    """Load and select EEG trial data from mne Epochs (.fif) datastores.

    Point this class at your dataset root folder and tell it which data version
    you want to work with.  It will find the matching metadata table
    automatically, giving you a simple interface to select and load trials.

    The dataset is expected to follow this layout::

        dataset_root/
        └── .../ (any depth)
            └── <version>/
                └── epochs/
                    ├── *metadata.tsv
                    └── sub-XX/
                        └── ses-XX-epo.fif   ← mne Epochs files

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
    >>> loader = TrialHandler('/data/temporal-natural-scenes-dataset', version='V0')

    >>> # Load a specific subset by filtering inline
    >>> result = loader.get_data(subject=1, condition=[5, 2951])

    >>> # Or look up a trial table first, then load
    >>> trials = loader.lookup_trials(subject=1)
    >>> epochs = loader.get_data(trials)
    >>> epochs.get_data().shape   # (n_trials, n_channels, n_samples)
    """


    def __init__(self, dataset_root: str = 'temporal-natural-scenes-dataset', version: str = 'V0'):
        """Resolve paths for reading datastore and initialize store cache for fast reading."""

        global BUCKET

        print('Resolving path...')
        self.root = resolve_dir(dataset_root, makedir=True)
        self.datastore = self.root / 'derivatives' / version / 'epochs'
        print('Reading metadata...')
        self.metadata = pd.read_csv(self.datastore / 'metadata.tsv', sep='\t', index_col=False)

        # Resolve only the handful of unique parent directories, not every one of
        # ~1M rows individually — the filename component is never a symlink, only
        # the directory can be, so per-row .resolve() was pure syscall overhead.
        parts = self.metadata['path'].str.rsplit('/', n=1, expand=True)
        parts.columns = ['dir', 'name']
        resolved_dirs = {d: str((self.datastore / d).resolve()) for d in parts['dir'].unique()}
        self.metadata['path'] = parts['dir'].map(resolved_dirs) + '/' + parts['name']

        self.store_cache = {}
        self.zscore_cache = {}
        print('Done.')


    @staticmethod
    def _stats_paths(epoch_path, level):
        """Map an epochs file path to its (mean, std) per-channel stats tsvs, e.g.
        ``sub-01/ses-eeg01-epo.fif`` -> ``sub-01/info/ses-eeg01_{level}_{mean,std}.tsv``."""
        epoch_path = Path(epoch_path)
        session = epoch_path.stem.removesuffix('-epo')
        info_dir = epoch_path.parent / 'info'
        return (info_dir / f'{session}_{level}_mean.tsv',
                info_dir / f'{session}_{level}_std.tsv')


    def _ensure_zscore_stats(self, paths, level):
        """Load per-channel zscore stats for the given store paths into self.zscore_cache, fetching any missing files."""
        needed = [p for p in paths if (p, level) not in self.zscore_cache]
        if not needed:
            return

        stats_files = [f for p in needed for f in self._stats_paths(p, level)]
        local_status = check_islocal(stats_files)
        missing = [p for p, local in local_status.items() if not local]

        if missing:
            ans = input(f'{len(missing)} {level}-level stats file(s) not found locally. Download from remote? [y/n] ')
            if ans.strip().lower() == 'y':
                workers = input('Multithreading possible, how many workers would you like to download with? ')
                fetch_remote(missing, BUCKET, self.root, max_workers=int(workers))
            else:
                raise RuntimeError(f"Missing {level}-level stats files, cannot apply zscore='{level}'.")

        for p in needed:
            mean_path, std_path = self._stats_paths(p, level)
            mean_df = pd.read_csv(mean_path, sep='\t')
            std_df = pd.read_csv(std_path, sep='\t')
            if 'run' in mean_df.columns:
                mean_df = mean_df.set_index('run')
                std_df = std_df.set_index('run')
            self.zscore_cache[(p, level)] = (mean_df, std_df)


    def _apply_zscore(self, piece, path, level, runs):
        """Standardize a loaded piece in place using cached per-channel mean/std stats.

        Stats only cover the EEG channels (not EOG/stim/etc), so align by
        channel name rather than assuming the stats and the piece have the
        same channel set/order."""
        mean_df, std_df = self.zscore_cache[(path, level)]
        ch_idx = [piece.ch_names.index(ch) for ch in mean_df.columns]
        if level == 'session':
            mean = mean_df.to_numpy()[0][None, :, None]
            std = std_df.to_numpy()[0][None, :, None]
        else:
            if isinstance(mean_df.index, pd.RangeIndex):
                mean = mean_df.to_numpy()[runs - 1][:, :, None]
                std = std_df.to_numpy()[runs - 1][:, :, None]
            else:
                mean = mean_df.loc[runs].to_numpy()[:, :, None]
                std = std_df.loc[runs].to_numpy()[:, :, None]
        # single fancy-index read + write (not two, via -= then /=) — the fancy
        # index copies on every access, so this halves the passes over the data
        piece._data[:, ch_idx, :] = (piece._data[:, ch_idx, :] - mean) / std


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

        if missing:
            ans = input(f'{len(missing)} data store(s) not found locally. Download from remote? [y/n] ')
            if ans.strip().lower() == 'y':
                workers = input(f'Multithreading possible, how many workers would you like to download with? ')
                fetch_remote(missing, BUCKET, self.root, max_workers=int(workers))
            else:
                trials = trials[~trials['path'].isin(missing)].reset_index(drop=True)

        return trials


    def get_data(
        self,
        trials: pd.DataFrame = None,
        channels=None,
        tmin: float = None,
        tmax: float = None,
        average_by=None,
        drop_bads: bool = True,
        zscore: str = None,
        verbose=True,
        cond='and',
        return_as='mne',
        **filters,
    ):
        """Load EEG data, with optional inline trial filtering.

        Reads the requested trials from disk as an :class:`mne.Epochs`
        object.  mne Epochs files are opened lazily and cached after the
        first open, so repeated calls for trials from the same file are
        fast.

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
        average_by : str or list of str, optional
            Metadata column(s) to average over.  For example
            ``average_by='condition'`` returns one averaged waveform per
            condition instead of one waveform per trial.  The returned
            metadata keeps ``average_by`` plus any other column that is
            constant within every group (e.g. ``subject``); columns that
            vary within a group (e.g. ``epoch``, ``onset``) are dropped.  An
            ``n_trials`` column is added recording how many trials went
            into each average.
        drop_bads : bool, optional
            Drop trials flagged ``bad`` in the metadata before loading, so
            they're excluded from the returned data and — when
            ``average_by`` is set — from the average itself.  Default
            ``True``.  Has no effect if the metadata has no ``bad`` column.
        zscore : {'run', 'session', None}, optional
            Standardize each channel using precomputed per-channel mean/std
            stats before cropping or averaging.  ``'run'`` uses stats
            computed within each trial's run, ``'session'`` uses stats
            computed across the whole session.  Missing stats files are
            fetched from remote the same way missing ``.fif`` stores are.
            Default ``None`` (no standardization).
        verbose : bool, optional
            Show a progress bar while loading.  Default ``True``.
        cond : {'and', 'or'}, optional
            How to combine multiple ``**filters`` (passed to
            :meth:`lookup_trials`).  Ignored when ``trials`` is provided
            explicitly.  Default ``'and'``.
        return_as : {'mne', 'numpy'}, optional
            ``'mne'`` (default) returns a single :class:`mne.Epochs` object.
            ``'numpy'`` skips building that object entirely — no
            :func:`mne.concatenate_epochs`, no re-wrapping into
            :class:`mne.EpochsArray` for ``average_by`` — and instead
            returns the raw ``(data, metadata)`` arrays directly.  Faster
            when you don't need mne's plotting/metadata machinery.
        **filters
            Column / value pairs forwarded to :meth:`lookup_trials` when
            ``trials`` is not provided.

        Returns
        -------
        mne.Epochs
            When ``return_as='mne'``.  One epoch per trial (or per group
            when ``average_by`` is set), in the same order as ``trials``.
            Trial metadata is attached as ``epochs.metadata``.
        (numpy.ndarray, pandas.DataFrame)
            When ``return_as='numpy'``.  ``data`` has shape
            ``(n_trials, n_channels, n_samples)`` (or ``(n_groups, ...)``
            when ``average_by`` is set); ``metadata`` has one row per
            entry in ``data``, in the same order.

        Examples
        --------
        >>> # Inline filtering — no separate lookup_trials call needed
        >>> epochs = loader.get_data(subject=1, shared=True)   # mne.Epochs, n_trials epochs

        >>> # Pass a pre-built trial table
        >>> trials = loader.lookup_trials(conditions=[1, 2, 3])
        >>> epochs = loader.get_data(trials)

        >>> # Average across trials, grouped by condition
        >>> epochs = loader.get_data(shared=True, average_by='condition')

        >>> # Skip the mne wrapper entirely
        >>> data, metadata = loader.get_data(subject=1, shared=True, return_as='numpy')
        """
        if return_as not in ('mne', 'numpy'):
            raise ValueError(f"Invalid return_as: {return_as!r}, must be 'mne' or 'numpy'")
        if zscore not in (None, 'run', 'session'):
            raise ValueError(f"Invalid zscore: {zscore!r}, must be 'run', 'session', or None")
        if trials is None:
            trials = self.lookup_trials(cond=cond, **filters) if filters else self.metadata.copy()

        if drop_bads and 'bad' in trials.columns:
            trials = trials[~trials['bad']].reset_index(drop=True)

        stores = trials['path'].unique()
        for path in stores:
            if path not in self.store_cache.keys():
                self.store_cache[path] = mne.read_epochs(path, preload=False, verbose=False)
        if zscore is not None:
            self._ensure_zscore_stats(stores, zscore)

        # Sort by store then array_index for sequential file access;
        # record original row position so output order matches input trials
        cols = ['path', 'array_index'] + (['run'] if zscore == 'run' else [])
        ordered = trials[cols].copy()
        ordered['out_row'] = np.arange(len(trials))
        ordered = ordered.sort_values(['path', 'array_index'])

        pieces, data_pieces, meta_pieces, out_rows = [], [], [], []
        with tqdm(total=len(trials), desc='Fetching trials', disable=not verbose) as prog, \
             mne.use_log_level('ERROR'):
            for path, group in ordered.groupby('path', sort=False):
                prog.set_description(f'Fetching trials [{Path(path).name}]')
                store = self.store_cache[path]
                arr_idcs = group['array_index'].to_numpy()
                piece = store[arr_idcs]
                piece.load_data()
                if zscore is not None:
                    runs = group['run'].to_numpy() if zscore == 'run' else None
                    self._apply_zscore(piece, path, zscore, runs)
                    # mne.concatenate_epochs re-reads file-backed epochs from disk,
                    # silently dropping in-place edits — rebuild as an in-memory
                    # EpochsArray so the zscored data actually survives concatenation.
                    piece = mne.EpochsArray(piece.get_data(), piece.info, events=piece.events,
                                             event_id=piece.event_id, tmin=piece.tmin,
                                             metadata=piece.metadata, verbose=False)
                if channels is not None:
                    piece.pick(channels)
                if tmin is not None or tmax is not None:
                    piece.crop(tmin=tmin, tmax=tmax)
                if return_as == 'numpy':
                    data_pieces.append(piece.get_data())
                    meta_pieces.append(piece.metadata)
                else:
                    pieces.append(piece)
                out_rows.append(group['out_row'].to_numpy())
                prog.update(len(group))

        if return_as == 'numpy':
            return self._get_numpy(data_pieces, meta_pieces, out_rows, average_by, verbose)
        order = np.argsort(np.concatenate(out_rows))
        return self._get_mne(pieces, order, average_by, verbose)

    @staticmethod
    def _aggregate_metadata(groups, keys):
        """Collapse grouped metadata to one row per group, keeping only columns constant within each group."""
        meta = (groups.agg(lambda col: col.iloc[0] if col.nunique() == 1 else np.nan)
                      .dropna(axis=1)
                      .reset_index())
        meta.insert(len(keys), 'n_trials', groups.size().to_numpy())
        return meta

    def _get_numpy(self, data_pieces, meta_pieces, out_rows, average_by, verbose):
        """Stack per-file arrays directly into (data, metadata), skipping mne's Epochs machinery entirely."""
        if verbose:
            print(f'Stacking {len(data_pieces)} file(s) worth of trials...')
        data = np.empty((sum(len(r) for r in out_rows), *data_pieces[0].shape[1:]), dtype=data_pieces[0].dtype)
        for piece, rows in zip(data_pieces, out_rows):
            data[rows] = piece

        order = np.argsort(np.concatenate(out_rows))
        metadata = pd.concat(meta_pieces, ignore_index=True).iloc[order].reset_index(drop=True)

        if average_by is not None:
            keys = [average_by] if isinstance(average_by, str) else list(average_by)
            if verbose:
                print(f'Averaging trials by {keys}...')
            groups = metadata.groupby(keys, sort=False)
            data = np.stack([data[grp.index.to_numpy()].mean(axis=0) for _, grp in groups])
            metadata = self._aggregate_metadata(groups, keys)

        if verbose:
            print('Done.')
        return data, metadata

    def _get_mne(self, pieces, order, average_by, verbose):
        """Concatenate per-file Epochs into one mne.Epochs object, averaging groups if requested."""
        if verbose:
            print(f'Concatenating {len(pieces)} file(s) worth of trials...')
        combined = mne.concatenate_epochs(pieces, verbose=False)

        if verbose:
            print('Reordering to match requested trial order...')
        combined = combined[order]

        if average_by is not None:
            keys = [average_by] if isinstance(average_by, str) else list(average_by)
            if verbose:
                print(f'Averaging trials by {keys}...')
            meta = combined.metadata.reset_index(drop=True)
            groups = meta.groupby(keys, sort=False)
            with mne.use_log_level('ERROR'):
                avg_data = np.stack([combined[grp.index.to_numpy()].get_data().mean(axis=0) for _, grp in groups])
                avg_meta = self._aggregate_metadata(groups, keys)
                combined = mne.EpochsArray(avg_data, combined.info, tmin=combined.tmin, verbose=False)
                combined.metadata = avg_meta

        if verbose:
            print('Done.')
        return combined


    def iter_data(
        self,
        trials: pd.DataFrame = None,
        batch_size: int = 1000,
        channels=None,
        tmin: float = None,
        tmax: float = None,
        average_by=None,
        drop_bads: bool = True,
        zscore: str = None,
        sort_lookup=True,
        verbose=True,
        cond='and',
        return_as='mne',
        **filters,
    ):
        """Iterate over trials in memory-friendly batches.

        Yields successive chunks of loaded EEG data instead of loading
        everything at once.  Useful when your full trial set is too large to
        fit in RAM, or when you want to feed a model batch-by-batch.

        Each yielded item is whatever :meth:`get_data` returns for the given
        ``return_as`` — an :class:`mne.Epochs` object (with metadata attached
        as ``epochs.metadata``), or a ``(data, metadata)`` tuple.

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
            Default is 1000.
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
        drop_bads : bool, optional
            Drop trials flagged ``bad`` in the metadata before loading each
            batch.  Forwarded to :meth:`get_data`.  Default ``True``.
        zscore : {'run', 'session', None}, optional
            Standardize each channel using precomputed per-channel mean/std
            stats before cropping or averaging.  Forwarded to
            :meth:`get_data` for every batch.  Default ``None``.
        sort_lookup : bool, optional
            Sort trials by store path and array index before batching, so
            each batch tends to draw from fewer distinct files.  Default
            ``True``.
        verbose : bool, optional
            Show a progress bar while loading each batch.  Forwarded to
            :meth:`get_data`.  Default ``True``.
        cond : {'and', 'or'}, optional
            How to combine multiple ``**filters``.  Ignored when ``trials``
            is provided explicitly.  Default ``'and'``.
        return_as : {'mne', 'numpy'}, optional
            Forwarded to :meth:`get_data` for every batch.  Default ``'mne'``.
        **filters
            Column / value pairs forwarded to :meth:`lookup_trials` when
            ``trials`` is not provided.

        Yields
        ------
        mne.Epochs or (numpy.ndarray, pandas.DataFrame)
            Same as :meth:`get_data` returns, per ``return_as``.

        Examples
        --------
        >>> # Inline filtering
        >>> for epochs in loader.iter_data(subject=1, batch_size=32):
        ...     process(epochs, epochs.metadata)   # mne.Epochs, <=32 epochs

        >>> # Iterate with per-stimulus averaging
        >>> trials = loader.lookup_trials(shared=True)
        >>> for epochs in loader.iter_data(trials, batch_size=64, average_by='subject'):
        ...     process(epochs, epochs.metadata)
        """
        if trials is None:
            trials = self.lookup_trials(cond=cond, **filters) if filters else self.metadata.copy()

        keys = ([average_by] if isinstance(average_by, str) else list(average_by)) if average_by else None

        if sort_lookup:
            trials = trials.sort_values(['path', 'array_index'])

        if keys:
            # accumulate complete groups into batches, never splitting a group
            batch, count = [], 0
            for _, grp in trials.groupby(keys, sort=False):
                if count + len(grp) > batch_size and batch:
                    yield self.get_data(pd.concat(batch), channels=channels,
                                        tmin=tmin, tmax=tmax, average_by=keys,
                                        drop_bads=drop_bads, zscore=zscore, verbose=verbose,
                                        return_as=return_as)
                    batch, count = [], 0
                batch.append(grp)
                count += len(grp)
            if batch:
                yield self.get_data(pd.concat(batch), channels=channels,
                                    tmin=tmin, tmax=tmax, average_by=keys,
                                    drop_bads=drop_bads, zscore=zscore, verbose=verbose,
                                    return_as=return_as)
        else:
            for start in range(0, len(trials), batch_size):
                yield self.get_data(
                    trials.iloc[start:start + batch_size],
                    channels=channels, tmin=tmin, tmax=tmax,
                    drop_bads=drop_bads, zscore=zscore, verbose=verbose,
                    return_as=return_as
                )
