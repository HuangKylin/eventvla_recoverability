"""Validate completed artifacts before inheriting a bounded branch prefix."""
import hashlib
import json
from pathlib import Path


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_resume(root, old_run, source, manifest):
    root, old_run, source = Path(root).resolve(), Path(old_run).resolve(), Path(source).resolve()
    runs = root/'runs'
    if not old_run.is_relative_to(runs):
        raise ValueError('Previous run outside project')
    old = json.loads((old_run/'status.json').read_text())
    if not old.get('process_cleanup_completed') or old['status'] not in ('failed','complete'):
        raise ValueError('Previous batch may still be active')
    if Path(old['source']).resolve() != source:
        raise ValueError('Different parent source')
    frozen = old_run/'selection_manifest.json'
    if file_sha(frozen) != old['manifest_sha256'] or json.loads(frozen.read_text()) != manifest:
        raise ValueError('Changed branch selection manifest')
    for name, value in old['source_hashes'].items():
        # Only orchestration may change to add resume; policy/worker are separately
        # pinned by the native census and input audit/pacing must remain unchanged.
        if name not in ('scripts/run_evidence_failure_branches.py', 'src/recoverability/evidence_branch_resume.py'):
            if file_sha(root/name) != value:
                raise ValueError('Changed paired audit or pacing code: '+name)
    expected = [b['name'] for b in manifest['branches']]
    actual = [b['name'] for b in old['episodes']]
    if actual != expected[:len(actual)] or len(set(actual)) != len(actual):
        raise ValueError('Completed branches are not a unique ordered prefix')
    if len(actual) == len(expected):
        raise ValueError('All branches already complete')
    inherited = []
    for row in old['episodes']:
        directory = Path(row.get('branch_dir', str(old_run/row['name']))).resolve()
        if not directory.is_relative_to(runs):
            raise ValueError('Inherited artifact outside project runs')
        result = json.loads((directory/'result.json').read_text())
        if file_sha(directory/'result.json') != row['result_sha256'] or result['status'] != 'complete':
            raise ValueError('Changed or incomplete inherited result')
        audit = json.loads((directory/'paired_audit.json').read_text())
        if audit != row['audit'] or not audit.get('first_request_exact_except_declared_images'):
            raise ValueError('Missing or changed actual input audit')
        spec = next(x for x in manifest['branches'] if x['name'] == row['name'])
        job = json.loads(directory.with_suffix('.job.json').read_text())
        if Path(job['source']).resolve() != source or job['replacements'] != spec['replacements']:
            raise ValueError('Inherited job does not match fixed branch')
        if row['name'] == 'unchanged':
            checks = audit.get('unchanged_replay', {})
            required = ('actions_exact','final_physics_exact','final_evaluator_exact','outcome_exact')
            if any(checks.get(k) is not True for k in required):
                raise ValueError('Unchanged replay not verified')
        inherited.append(dict(row, branch_dir=str(directory), inherited_from=str(old_run)))
    ancestors = set()
    cursor = old_run
    while True:
        if cursor in ancestors or not cursor.is_relative_to(runs):
            raise ValueError('Invalid resume lineage')
        ancestors.add(cursor)
        previous = json.loads((cursor/'status.json').read_text())
        if not previous.get('resumed_from'):
            break
        cursor = Path(previous['resumed_from']).resolve()
    return inherited, ancestors
