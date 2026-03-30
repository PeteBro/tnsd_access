"""Representational Dissimilarity Matrix (RDM) handler.

RDMs are stored per-subject as Zarr groups under::

    derivatives/<modality>/<version>/rdms/sub-XX.zarr

Each store contains three arrays:

    stim_ids  — int32,   (n_stims,)          sorted stimulus IDs for this subject
    pairs     — int32,   (n_pairs, 2)         upper-triangle pairs, no diagonal
    rdm       — float32, (n_windows, n_pairs) pairwise decoding accuracy per time window

Dissimilarity values are pairwise linear SVC decoding accuracies (0.5 = chance,
1.0 = perfectly discriminable).  NaN means that window/pair has not yet been
computed — computation is distributed across SLURM jobs and filled in
incrementally.

Chunks are (1, 50000): one window per chunk row, optimised for writing one
window at a time.  Window times (in seconds, centre of each window) are stored
as a Zarr attribute ``'times'``.
"""

from pathlib import Path

import numpy as np
import zarr


class RDMHandler:
    """Load RDMs for a given modality and derivative version.

    Instantiated via ``ds.rdm.eeg`` or ``ds.rdm.fmri`` — do not construct
    directly.

    Parameters
    ----------
    root : str or Path
        Dataset root directory.
    version : str
        Derivative version (e.g. ``'v1'``).
    modality : str
        Source modality — ``'eeg'`` or ``'fmri'``.
    """

    def __init__(self, root, version: str, modality: str):
        self.root = Path(root)
        self.version = version
        self.modality = modality
        self._rdm_root = self.root / 'derivatives' / modality / version / 'rdms'
        self._store_cache: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_store(self, subject: int) -> zarr.Group:
        if subject not in self._store_cache:
            path = self._rdm_root / f'sub-{subject:02d}.zarr'
            self._store_cache[subject] = zarr.open(str(path), mode='r')
        return self._store_cache[subject]

    @staticmethod
    def _get_times(store: zarr.Group):
        return np.array(store.attrs['times']) if 'times' in store.attrs else None

    @staticmethod
    def _time_mask(times, tmin, tmax):
        mask = np.ones(len(times), dtype=bool)
        if tmin is not None:
            mask &= times >= tmin
        if tmax is not None:
            mask &= times <= tmax
        return mask

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, subject: int, complete_only: bool = False,
             tmin: float = None, tmax: float = None,
             compute: bool = False) -> dict:
        """Load the full RDM store for a subject.

        By default returns raw Zarr arrays (lazy, zero-copy, cloud-friendly).
        Any filtering option or ``compute=True`` materialises to NumPy.

        Parameters
        ----------
        subject : int
        complete_only : bool
            If True, drop pairs that contain any NaN across time windows.
        tmin : float, optional
            Start of time window in seconds (inclusive).
        tmax : float, optional
            End of time window in seconds (inclusive).
        compute : bool
            If True, materialise all arrays to NumPy before returning.

        Returns
        -------
        dict
            ``{'rdm': array (n_windows, n_pairs), 'pairs': (n_pairs, 2),``
            ``'stim_ids': (n_stims,), 'times': (n_windows,) or None}``

        Examples
        --------
        >>> store = ds.rdm.eeg.load(subject=1)
        >>> store['rdm']          # Zarr array — slice as needed
        >>> store['rdm'][10, :]   # all pairs at window 10

        >>> # Materialise a time-cropped, complete-pairs-only subset
        >>> store = ds.rdm.eeg.load(subject=1, tmin=0.0, tmax=0.3,
        ...                         complete_only=True, compute=True)
        """
        store = self._get_store(subject)
        times = self._get_times(store)
        needs_load = complete_only or tmin is not None or tmax is not None or compute

        if not needs_load:
            return {
                'rdm':      store['rdm'],
                'pairs':    store['pairs'],
                'stim_ids': store['stim_ids'],
                'times':    times,
            }

        # Materialise
        rdm_arr      = store['rdm'][:]       # (n_windows, n_pairs)
        pairs_arr    = store['pairs'][:]
        stim_ids_arr = store['stim_ids'][:]

        if times is not None and (tmin is not None or tmax is not None):
            t_mask   = self._time_mask(times, tmin, tmax)
            rdm_arr  = rdm_arr[t_mask]
            times    = times[t_mask]

        if complete_only:
            complete  = ~np.any(np.isnan(rdm_arr), axis=0)
            rdm_arr   = rdm_arr[:, complete]
            pairs_arr = pairs_arr[complete]

        return {
            'rdm':      rdm_arr,
            'pairs':    pairs_arr,
            'stim_ids': stim_ids_arr,
            'times':    times,
        }

    # ------------------------------------------------------------------

    def get_pair(self, subject: int, stim_a: int, stim_b: int,
                 tmin: float = None, tmax: float = None) -> dict:
        """Load the full temporal trajectory for a single stimulus pair.

        The pair is canonicalised to ``(min, max)`` automatically.

        Parameters
        ----------
        subject : int
        stim_a : int
            First stimulus ID.
        stim_b : int
            Second stimulus ID.
        tmin : float, optional
        tmax : float, optional

        Returns
        -------
        dict
            ``{'trajectory': (n_windows,), 'times': (n_windows,) or None}``

        Examples
        --------
        >>> result = ds.rdm.eeg.get_pair(subject=1, stim_a=101, stim_b=204)
        >>> result['trajectory']   # (n_windows,) — decoding accuracy over time
        """
        a, b = min(stim_a, stim_b), max(stim_a, stim_b)
        store  = self._get_store(subject)
        pairs  = store['pairs'][:]
        hits   = np.where((pairs[:, 0] == a) & (pairs[:, 1] == b))[0]
        if len(hits) == 0:
            raise KeyError(f'Pair ({a}, {b}) not found in subject {subject} store.')
        col    = int(hits[0])

        times  = self._get_times(store)
        traj   = store['rdm'][:, col]   # (n_windows,)

        if times is not None and (tmin is not None or tmax is not None):
            mask = self._time_mask(times, tmin, tmax)
            traj  = traj[mask]
            times = times[mask]

        return {'trajectory': traj, 'times': times}

    # ------------------------------------------------------------------

    def get_stimulus(self, subject: int, stim_id: int,
                     tmin: float = None, tmax: float = None,
                     complete_only: bool = False) -> dict:
        """Load all pairs involving a single stimulus.

        Parameters
        ----------
        subject : int
        stim_id : int
        tmin : float, optional
        tmax : float, optional
        complete_only : bool
            Drop pairs with any NaN in their temporal trajectory.

        Returns
        -------
        dict
            ``{'rdm': (n_windows, n_pairs), 'pairs': (n_pairs, 2),``
            ``'times': (n_windows,) or None}``
        """
        store = self._get_store(subject)
        pairs = store['pairs'][:]
        mask  = (pairs[:, 0] == stim_id) | (pairs[:, 1] == stim_id)
        cols  = np.where(mask)[0]
        if len(cols) == 0:
            raise KeyError(f'Stimulus {stim_id} not found in subject {subject} store.')

        times   = self._get_times(store)
        rdm_arr = store['rdm'].oindex[:, cols.tolist()]  # (n_windows, n_pairs_for_stim)

        if times is not None and (tmin is not None or tmax is not None):
            t_mask  = self._time_mask(times, tmin, tmax)
            rdm_arr = rdm_arr[t_mask]
            times   = times[t_mask]

        if complete_only:
            complete = ~np.any(np.isnan(rdm_arr), axis=0)
            rdm_arr  = rdm_arr[:, complete]
            cols     = cols[complete]

        return {'rdm': rdm_arr, 'pairs': pairs[cols], 'times': times}

    # ------------------------------------------------------------------

    def get_matrix(self, subject: int, time: float = None,
                   complete_only: bool = False) -> dict:
        """Reconstruct the square symmetric RDM at a given time point.

        Parameters
        ----------
        subject : int
        time : float, optional
            Time in seconds.  The nearest window is selected.  If ``None``,
            values are averaged across all windows (ignoring NaN).
        complete_only : bool
            If True, leave incomplete pairs as NaN in the output matrix.
            If False (default), NaN cells appear wherever computation is
            still pending.

        Returns
        -------
        dict
            ``{'matrix': (n_stims, n_stims), 'stim_ids': (n_stims,)}``
            Diagonal is 0.0; uncomputed cells are NaN.

        Examples
        --------
        >>> result = ds.rdm.eeg.get_matrix(subject=1, time=0.1)
        >>> result['matrix']   # (n_stims, n_stims)
        """
        store    = self._get_store(subject)
        stim_ids = store['stim_ids'][:]
        pairs    = store['pairs'][:]
        times    = self._get_times(store)
        n        = len(stim_ids)
        id_to_idx = {int(sid): i for i, sid in enumerate(stim_ids)}

        if time is not None and times is not None:
            w_idx   = int(np.abs(times - time).argmin())
            rdm_row = store['rdm'][w_idx, :]
        elif time is not None:
            rdm_row = store['rdm'][int(time), :]
        else:
            rdm_row = np.nanmean(store['rdm'][:], axis=0)

        matrix = np.full((n, n), np.nan)
        np.fill_diagonal(matrix, 0.0)

        for col, (a, b) in enumerate(pairs):
            val = rdm_row[col]
            if complete_only and np.isnan(val):
                continue
            i, j = id_to_idx[int(a)], id_to_idx[int(b)]
            matrix[i, j] = matrix[j, i] = val

        return {'matrix': matrix, 'stim_ids': stim_ids}
