"""EEG epoch derivatives handler."""

import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

from .._base import DerivativesHandler
from .._utils import resolve_dir, check_islocal, fetch_remote
from ..config import BUCKET


class EEGHandler(DerivativesHandler):
    """Load and select EEG epoch data from versioned Zarr datastores.

    Point this at your dataset root and specify a derivative version.  It will
    locate the matching metadata table automatically, giving you a simple
    interface for selecting and loading trials.

    The dataset is expected to follow this layout::

        dataset_root/
        └── derivatives/eeg/<version>/datastore/
            ├── metadata.tsv
            └── sub-XX/
                └── chunk-XX/   ← zarr stores

    Access raw continuous recordings via :attr:`raw`.

    Parameters
    ----------
    root : str or Path
        Top-level folder of your dataset (e.g. ``'/data/tnsd'``).
    version : str
        Derivative version directory to load (e.g. ``'v1'``).

    Examples
    --------
    >>> ds = Dataset('/data/temporal-natural-scenes-dataset')
    >>> trials = ds.eeg.lookup_trials(subject=1, shared=True)
    >>> result  = ds.eeg.get_data(trials)
    >>> result['data'].shape   # (n_trials, n_channels, n_samples)
    """

    def __init__(self, root: str = 'temporal-natural-scenes-dataset',
                 version: str = 'v1'):
        print('Resolving path...')
        root = resolve_dir(root, makedir=True)
        super().__init__(root, version)
        self.datastore = self.root / 'derivatives' / 'eeg' / self.version / 'datastore'
        print('Reading metadata...')
        self.metadata = pd.read_csv(self.datastore / 'metadata.tsv', sep='\t', index_col=False)
        self.metadata['path'] = self.metadata['path'].apply(
            lambda p: (self.datastore / p).resolve()
        )
        self.store_cache = {}
        print('Done.')

    @property
    def raw(self):
        """Access raw continuous EEG recordings (:class:`EEGRawHandler`)."""
        if not hasattr(self, '_raw'):
            from ._raw import EEGRawHandler
            self._raw = EEGRawHandler(self.root)
        return self._raw

    # ------------------------------------------------------------------

    def lookup_trials(self, cond='and', **filters) -> pd.DataFrame:
        """Return trials matching the given metadata criteria.

        Pass any metadata column as a keyword argument to filter trials.
        Multiple filters are combined with ``cond='and'`` (all criteria must
        match) or ``cond='or'`` (any criterion is enough).  The returned
        DataFrame can be passed directly to :meth:`get_data` or
        :meth:`iter_data`.

        Parameters
        ----------
        cond : {'and', 'or'}, optional
            How to combine multiple filters.  Default ``'and'``.
        **filters
            Column name / value pairs.  A value can be a single item or a
            list.  For example ``condition=[1, 2, 3]`` keeps only trials whose
            ``condition`` column is 1, 2, or 3.

        Other Parameters
        ----------------
        subject : int or list of int
        condition : int or list of int
        session : int or list of int
        run : int or list of int
        epoch : int or list of int
        trial_instance : int or list of int
        trial_type : str or list of str
        shared : bool
        onset : float or list of float
        duration : float or list of float
        response_time : float or list of float
        response : str or list of str
        stim_file : str or list of str
        trigger_value : int or list of int
        event_id : int or list of int
        datetime : str or list of str

        Returns
        -------
        pd.DataFrame
            Filtered metadata table with a fresh integer index.

        Examples
        --------
        >>> trials = ds.eeg.lookup_trials(subject=1, shared=True)
        >>> trials = ds.eeg.lookup_trials(cond='or', subject=[1, 2])
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
                workers = input('Multithreading possible, how many workers would you like to download with? ')
                fetch_remote(missing, BUCKET, self.root, max_workers=int(workers))
            else:
                trials = trials[~trials['path'].isin(missing)].reset_index(drop=True)

        return trials

    # ------------------------------------------------------------------

    def get_data(
        self,
        trials: pd.DataFrame = None,
        channels=None,
        tmin: float = None,
        tmax: float = None,
        step=None,
        sample_idcs=None,
        average_by=None,
        verbose=True,
        cond='and',
        **filters,
    ) -> dict:
        """Load EEG epoch data into memory.

        Zarr stores are cached after the first open, so repeated calls for
        trials from the same store are fast.

        You can supply trials three ways:

        * Pass a pre-built ``trials`` DataFrame (e.g. from :meth:`lookup_trials`).
        * Pass filter keyword arguments directly.
        * Pass neither — all trials in the metadata table are loaded.

        Parameters
        ----------
        trials : pd.DataFrame, optional
            Trial metadata table with ``path`` and ``array_index`` columns.
        channels : list, optional
            Integer indices or channel name strings to load.  All channels
            when omitted.
        tmin : float, optional
            Start of the time window in seconds.  Ignored when ``sample_idcs``
            is provided.
        tmax : float, optional
            End of the time window in seconds.  Ignored when ``sample_idcs``
            is provided.
        step : int, optional
            Sample step for downsampling (``step=2`` → every other sample).
        sample_idcs : array-like of int, optional
            Explicit sample indices.  Takes precedence over ``tmin``/``tmax``/``step``.
        average_by : str or list of str, optional
            Metadata column(s) to average over (e.g. ``average_by='condition'``).
        verbose : bool, optional
            Show a progress bar (default ``True``).
        cond : {'and', 'or'}, optional
            Filter combination mode, forwarded to :meth:`lookup_trials`.
        **filters
            Column / value pairs forwarded to :meth:`lookup_trials`.

        Returns
        -------
        dict
            ``{'data': np.ndarray, 'metadata': pd.DataFrame}``
            Data shape: ``(n_trials, n_channels, n_samples)``.

        Examples
        --------
        >>> result = ds.eeg.get_data(subject=1, shared=True)
        >>> result = ds.eeg.get_data(trials, average_by='condition')
        """
        if trials is None:
            trials = self.lookup_trials(cond=cond, **filters) if filters else self.metadata.copy()

        stores = trials['path'].unique()
        for path in stores:
            if path not in self.store_cache:
                self.store_cache[path] = zarr.open(path, mode='r')

        store0 = self.store_cache[stores[0]]
        info = store0.attrs['info']
        times = np.asarray(store0.attrs['times'])
        channel_names = info['ch_names']

        if channels is None:
            channels = slice(None)
        else:
            channels = np.array([
                channel_names.index(c) if isinstance(c, str) else c
                for c in channels
            ])
            if channels.size > 1 and np.all(np.diff(channels) == channels[1] - channels[0]):
                channels = slice(channels[0], channels[-1] + 1, channels[1] - channels[0])

        if sample_idcs is not None:
            samples = np.asarray(sample_idcs)
        else:
            tmin_idx = 0 if tmin is None else np.abs(times - tmin).argmin()
            tmax_idx = len(times) if tmax is None else np.abs(times - tmax).argmin()
            samples = slice(tmin_idx, tmax_idx, step)

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

        meta = trials.reset_index(drop=True)

        if average_by is not None:
            keys = [average_by] if isinstance(average_by, str) else list(average_by)
            groups = meta.groupby(keys, sort=False)
            data_array = np.stack([data_array[grp.index].mean(axis=0) for _, grp in groups])
            meta = (groups.agg(lambda col: col.iloc[0] if col.nunique() == 1 else np.nan)
                         .drop(columns=['path', 'array_index'], errors='ignore')
                         .dropna(axis=1)
                         .reset_index())

        return {'data': data_array, 'metadata': meta}

    # ------------------------------------------------------------------

    def iter_data(
        self,
        trials: pd.DataFrame = None,
        batch_size: int = 1000,
        channels=None,
        tmin: float = None,
        tmax: float = None,
        average_by=None,
        sort_lookup=True,
        cond='and',
        **filters,
    ):
        """Iterate over trials in memory-friendly batches.

        Each yielded item has the same structure as the dict returned by
        :meth:`get_data`.  When ``average_by`` is set, groups are never split
        across batches.

        Parameters
        ----------
        trials : pd.DataFrame, optional
            Trial metadata table.  When omitted, ``**filters`` are used or all
            trials are iterated.
        batch_size : int, optional
            Maximum number of trials per batch (default 1000).
        channels : list, optional
            Integer indices or channel name strings.
        tmin : float, optional
            Start of time window in seconds.
        tmax : float, optional
            End of time window in seconds.
        average_by : str or list of str, optional
            Metadata column(s) to average within each batch.
        sort_lookup : bool, optional
            Sort by store path and array index for sequential reads (default True).
        cond : {'and', 'or'}, optional
            Filter combination mode.
        **filters
            Column / value pairs forwarded to :meth:`lookup_trials`.

        Yields
        ------
        dict
            ``{'data': np.ndarray, 'metadata': pd.DataFrame}``

        Examples
        --------
        >>> for batch in ds.eeg.iter_data(subject=1, batch_size=32):
        ...     process(batch['data'], batch['metadata'])
        """
        if trials is None:
            trials = self.lookup_trials(cond=cond, **filters) if filters else self.metadata.copy()

        keys = ([average_by] if isinstance(average_by, str) else list(average_by)) if average_by else None

        if sort_lookup:
            trials = trials.sort_values(['path', 'array_index'])

        if keys:
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
                    channels=channels, tmin=tmin, tmax=tmax, verbose=False,
                )
