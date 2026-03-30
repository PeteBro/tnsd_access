"""Shared utilities for dataset discovery and remote access."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from tqdm import tqdm

from .config import BUCKET, SEED_FILES


def init_dataset(root, modalities=('dataset', 'eeg', 'mri', 'stimuli')):
    """Download seed files for all (or selected) modalities from S3.

    Parameters
    ----------
    root : str
        Local path where the dataset will be initialised.
    modalities : tuple of str
        Which seed files to fetch. Any combination of
        ``'dataset'`` (top-level BIDS files), ``'eeg'``, ``'mri'``,
        ``'stimuli'``. Defaults to all modalities.
    """
    root = Path(root)
    os.makedirs(root, exist_ok=True)

    files = []
    for m in modalities:
        files.extend(SEED_FILES.get(m, []))

    if not files:
        print('No seed files defined for the requested modalities.')
        return
    print('Fetching seed files...')
    fetch_remote([root / f for f in files], BUCKET, root)


def _s3_download(bucket, key, dest, progress):
    dest.parent.mkdir(parents=True, exist_ok=True)
    boto3.client('s3').download_file(bucket, key, str(dest),
                                     Callback=lambda n: progress.update(n))


def fetch_remote(paths, bucket, root, max_workers=16, verbose=True):
    """Download files from S3 with parallel workers and progress tracking.

    Parameters
    ----------
    paths : list of Path
        Local destination paths (used to derive S3 keys relative to *root*).
    bucket : str
        S3 bucket name.
    root : str or Path
        Dataset root; used to compute S3 key prefixes.
    max_workers : int
        Number of parallel download threads (default 16).
    verbose : bool
        Show a tqdm progress bar (default True).
    """
    s3 = boto3.client('s3')
    root = Path(root)

    objects = []
    for local_path in paths:
        prefix = str(Path(local_path).relative_to(root))
        for page in s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                objects.append((obj['Key'], obj['Size']))

    total_bytes = sum(size for _, size in objects)
    failed = []
    with tqdm(total=total_bytes, unit='B', unit_scale=True, desc='Downloading', disable=not verbose) as progress:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_s3_download, bucket, key, root / key, progress): key
                for key, _ in objects
            }
            for future in as_completed(futures):
                if future.exception():
                    failed.append(futures[future])

    if failed:
        print(f'S3 fetch failed for {len(failed)} file(s), check filenames or credentials.')


def resolve_dir(path, start=None, makedir=False):
    """Locate a directory by searching upward then downward from *start*.

    Parameters
    ----------
    path : str or Path
        Directory name or relative path to find.
    start : str or Path, optional
        Search root. Defaults to ``os.getcwd()``.
    makedir : bool
        If True and no match is found, create the directory rather than raising.

    Returns
    -------
    Path
        Resolved absolute path to the directory.
    """
    start = Path(start).resolve() if start else Path(os.getcwd())
    path = Path(path)
    upward = [p / path for p in [start, *start.parents] if (p / path).is_dir()]
    downward = ([p for p in start.rglob(str(path)) if p.is_dir()]
                if not upward and not path.is_absolute() else [])

    matches = list({p.resolve() for p in upward + downward})

    if len(matches) == 0:
        if makedir:
            os.makedirs(path)
            print('Could not find specified path, created instead.')
        else:
            raise RuntimeError(f"Could not find directory '{path}'")
    if len(matches) > 1:
        matches_str = "\n  ".join(str(m) for m in matches)
        raise RuntimeError(f"Ambiguous — multiple '{path}' directories found:\n  {matches_str}")

    return matches[0]


def check_islocal(paths):
    """Return a dict mapping each path to whether it exists locally."""
    return {p: os.path.exists(p) for p in paths}
