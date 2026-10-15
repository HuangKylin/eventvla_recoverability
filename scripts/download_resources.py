"""Download pinned official resources and verify sizes plus upstream object hashes."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / 'cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '300')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '30')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
from huggingface_hub import snapshot_download

GROUPS = {
    'policy': ('ganlinyang/EventVLA', 'model', ROOT / 'checkpoints/EventVLA'),
    'backbone': ('Qwen/Qwen3-VL-4B-Instruct', 'model', ROOT / 'checkpoints/Qwen3-VL-4B-Instruct'),
    'assets': ('ganlinyang/RoboTwin-MeM', 'dataset', ROOT),
}


def selected(group, filename):
    if group == 'assets':
        return filename.startswith('assets/')
    if group == 'policy':
        return filename in {
            'RoboTwin-MeM/final_model/pytorch_model.pt', 'RoboTwin-MeM/config.yaml',
            'RoboTwin-MeM/dataset_statistics.json', 'RoboTwin-MeM/summary.jsonl',
            'RoboTwin-MeM/run_eventvla_train.sh'}
    return True


def state(group, **values):
    p = ROOT / 'reports' / ('download_' + group + '_status.json')
    values.update(group=group, pid=os.getpid(),
                  updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(values, indent=2))
    tmp.replace(p)
    print(json.dumps(values), flush=True)


def download(group):
    rid, rtype, destination = GROUPS[group]
    meta = json.loads((ROOT / 'references' / (rid.replace('/', '--') + '_remote_metadata.json')).read_text())
    entries = [x for x in meta['siblings'] if selected(group, x['rfilename'])]
    assert entries, 'No files selected: ' + group
    names = [e['rfilename'] for e in entries]
    revision = meta['sha']
    total = sum(e.get('size', 0) for e in entries)
    state(group, status='downloading', repo=rid, revision=revision,
          destination=str(destination), files=len(entries), expected_bytes=total)
    snapshot_download(rid, repo_type=rtype, revision=revision,
                      allow_patterns=names, local_dir=str(destination), max_workers=4)
    state(group, status='verifying', repo=rid, revision=revision,
          destination=str(destination), files=len(entries), expected_bytes=total)
    manifest = []
    for entry in entries:
        path = destination / entry['rfilename']
        assert path.resolve().is_relative_to(ROOT.resolve())
        size = path.stat().st_size
        assert size == entry['size'], f'Size mismatch: {entry["rfilename"]}'
        sha = hashlib.sha256()
        blob = hashlib.sha1(('blob ' + str(size) + '\0').encode())
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                sha.update(block)
                blob.update(block)
        lfs = entry.get('lfs')
        if lfs:
            expected = lfs.get('sha256') or lfs.get('oid')
            assert expected and sha.hexdigest() == expected, 'LFS hash mismatch: ' + str(path)
        elif entry.get('blobId'):
            assert blob.hexdigest() == entry['blobId'], 'Git blob mismatch: ' + str(path)
        manifest.append(dict(path=str(path.relative_to(ROOT)), bytes=size, sha256=sha.hexdigest()))
    (ROOT / 'reports' / ('download_' + group + '_manifest.json')).write_text(
        json.dumps(dict(repo=rid, revision=revision, files=manifest), indent=2))
    state(group, status='complete', repo=rid, revision=revision,
          destination=str(destination), files=len(entries), verified_bytes=total)


def guarded(group):
    for attempt in range(1, 13):
        try:
            download(group)
            return
        except Exception as exc:
            if attempt == 12:
                state(group, status='failed', error=traceback.format_exc())
                raise
            state(group, status='retrying', attempt=attempt,
                  error_type=type(exc).__name__, resume_partial_files=True)
            time.sleep(2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--group', choices=['all', 'models', *GROUPS], default='all')
    args = p.parse_args()
    groups = (list(GROUPS) if args.group == 'all' else
              ['policy', 'backbone'] if args.group == 'models' else [args.group])
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as pool:
        list(pool.map(guarded, groups))
