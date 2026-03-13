"""Shared utilities for dataset discovery and metadata construction."""

import os
#import glob
#import json
#import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import boto3
#import zarr

from .config import BUCKET, SEED_FILES

def init_dataset(root):

    root = Path(root)
    os.makedirs(root, exist_ok=True)

    print('Fetching seed files...')
    fetch_remote([root / f for f in SEED_FILES], BUCKET, root)


def _s3_download(bucket, key, dest, progress):

    dest.parent.mkdir(parents=True, exist_ok=True)
    boto3.client('s3').download_file(bucket, key, str(dest),
                                     Callback=lambda n: progress.update(n))


def fetch_remote(paths, bucket, root, max_workers=16, verbose=True):
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

    start = Path(start).resolve() if start else Path(os.getcwd())
    path = Path(path)
    upward = [p / path for p in [start, *start.parents] if (p / path).is_dir()]
    downward = [p for p in start.rglob(str(path)) if p.is_dir()] if not upward and not path.is_absolute() else []

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

    return {p: os.path.exists(p) for p in paths}


#def get_hash(path):
#    hasher = hashlib.sha256()
#    allfiles = sorted(glob.glob(os.path.join(path, '**'), recursive=True))
#    for fname in allfiles:
#        if os.path.isfile(fname):
#            hasher.update(open(fname, 'rb').read())
#    return hasher.hexdigest()


#def check_stale(paths, bucket, root):
#    """Return paths whose checksum differs from the data_manifest.json on S3.
#
#    TODO: also verify local file checksums match the local manifest, to catch
#    manual edits or corruption.
#    """
#    if not paths:
#        return []
#
#    root = Path(root)
#    datastore = Path(paths[0]).parent.parent
#    manifest_key = str(datastore.relative_to(root) / 'manifest.json')
#    local_manifest_path = datastore / 'manifest.json'
#
#    try:
#        response = boto3.client('s3').get_object(Bucket=bucket, Key=manifest_key)
#        s3_manifest = json.loads(response['Body'].read())
#    except Exception:
#        return []
#
#    if not local_manifest_path.exists():
#        return list(paths)
#
#    with open(local_manifest_path) as f:
#        local_manifest = json.load(f)
#
#    return [
#        p for p in paths
#        if s3_manifest.get(str(Path(p).relative_to(datastore))) != local_manifest.get(str(Path(p).relative_to(datastore)))
#    ]


#def build_trial_metadata(epochs_root: str) -> pd.DataFrame:
#
#    records = []
#    for subject_dir in sorted(glob.glob(os.path.join(epochs_root, "sub-*"))):
#        for chunk_dir in sorted(glob.glob(os.path.join(subject_dir, "chunk-*"))):
#            z = zarr.open(chunk_dir, mode="r")
#            df = pd.DataFrame(dict(z.attrs))
#            df["path"] = chunk_dir
#            records.append(df)
#    return pd.concat(records, ignore_index=True)