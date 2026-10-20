"""CPU-only upstream input/token audit. Old native parents are regression fixtures.

No policy forward, action proposal, simulator rollout, or research sample is made.
Only load this project's own trusted pickle snapshots (pickle is executable).
"""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import subprocess
import time
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party/EventVLA/EventVLA'
sys.path[:0] = [str(ROOT / 'src'), str(UPSTREAM)]


def build_client_fixtures(destination):
    import numpy as np
    from PIL import Image
    from recoverability.archive_readout import ObservedArchive, select_public
    from recoverability.vla_intervention import image_hash, purge_evidence
    client_file = UPSTREAM / 'examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py'
    spec = importlib.util.spec_from_file_location('archive_audit_client', client_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixtures = []
    fixture_runs = {410000: '20261016T072851Z_native_parent_collection',
                    410001: '20261016T073957Z_native_parent_collection',
                    410002: '20261016T074712Z_native_parent_collection'}
    for seed, run in fixture_runs.items():
        source = ROOT / 'runs' / run / f'{seed}_prepare'
        snapshot_path = source / 'client_state.pkl'
        snapshot_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        with snapshot_path.open('rb') as stream:
            state = pickle.load(stream)
        parent = json.loads((source / 'parent.json').read_text())
        step = int(parent['step'])
        client = module.ModelClient.__new__(module.ModelClient)
        client.__dict__.update(copy.deepcopy(state))
        archive = ObservedArchive()

        def add_record(record_step, images):
            return [archive.append(int(record_step), str(client.temporal_view_names[i]), image)
                    for i, image in enumerate(images)]

        add_record(client.initial_step, client.initial_images)
        for record in client.image_history:
            add_record(record['step'], record['images'])
        candidates, entry_by_key = [], {}
        for entry in client.keyframe_image_memory:
            assert len(entry['images']) == len(entry['metas']) == 1
            key = archive.append(int(entry['step']), str(entry['metas'][0]['view']), entry['images'][0])
            candidates.append(key)
            entry_by_key[key] = entry
        current = [client._resize_image(np.asarray(Image.open(source / f'parent_{view}.png')))
                   for view in ('head_camera', 'left_camera', 'right_camera')]
        add_record(step, current)
        client._record_temporal_images(current, step)
        anchors, metas = client._build_temporal_anchor_images(step)
        anchor_keys = []
        for pixels, meta in zip(anchors, metas):
            target = (int(client.initial_step) + int(meta['absolute_index'])
                      if meta['absolute_index'] is not None else step + int(meta['delta']))
            matches = [r for r in archive.metadata(target)
                       if r.view == meta['view'] and r.sha256 == image_hash(pixels)]
            assert matches, 'anchor must refer to an actually archived observation'
            anchor_keys.append(matches[-1].key)
        fingerprint = archive.fingerprint(step)
        instruction = parent['native_instruction']

        def prepare(selected, stage):
            variant = module.ModelClient.__new__(module.ModelClient)
            variant.__dict__.update(copy.deepcopy(client.__dict__))
            # Equal reset, including proposals. Archive remains independent/unchanged.
            purge_evidence(variant, [])
            variant.keyframe_image_memory = [copy.deepcopy(entry_by_key[key]) for key in selected]
            example = dict(image=anchors, image_metas=metas, lang=instruction)
            variant._inject_keyframe_memory_fields(example, step)
            assert variant.raw_actions is None and variant.pending_commit_done
            return dict(stage=stage, selected=selected, example=example)

        payloads = [prepare((), 'proposal'), prepare(tuple(candidates), 'single_pass_all_events')]
        for budget in (1, 2, 4):
            for strategy in ('recent', 'uniform'):
                selected = select_public(archive, candidates, budget, step, strategy)
                payloads.append(prepare(selected, 'final') | dict(method=strategy, event_read_cap=budget))
        assert hashlib.sha256(snapshot_path.read_bytes()).hexdigest() == snapshot_sha
        fixtures.append(dict(seed=seed, step=step, archive=archive, candidates=candidates,
            anchor_keys=anchor_keys, payloads=payloads, source=str(source), snapshot_sha256=snapshot_sha,
            archive_fingerprint=fingerprint, archived_images=len(archive.metadata(step)),
            archive_scope='snapshot_available_history_only', selectable_scope='four_saved_event_slots',
            event_candidates=len(candidates), anchor_images=len(anchor_keys)))
    with destination.open('wb') as stream:
        pickle.dump(fixtures, stream)


def main():
    import numpy as np
    from PIL import Image
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from eventvla.model.framework.EventVLA import EventVLA
    from eventvla.model.modules.vlm.QWen3 import _QWen3_VL_Interface
    from eventvla.training.trainer_utils.trainer_tools import resize_images
    from recoverability.archive_readout import InputTrace, ReadLedger, ReadBudget, Proposal, validate_proposal, digest
    from recoverability.vla_intervention import image_hash

    torch.set_num_threads(2)
    run_dir = ROOT / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_archive_readout_audit')
    run_dir.mkdir(parents=True, exist_ok=False)
    fixtures_file = run_dir/'trusted_client_fixtures.pkl'
    # Keep the existing sim/policy dependencies separate; no installation needed.
    subprocess.run([str(ROOT/'envs/sim/bin/python'), str(Path(__file__)),
                    '--client-fixtures', str(fixtures_file)], check=True, timeout=120)
    with fixtures_file.open('rb') as stream:
        fixtures = pickle.load(stream)
    config_path = ROOT / 'checkpoints/runtime/EventVLA-RoboTwin-MeM/config.yaml'
    cfg = OmegaConf.load(config_path)
    processor = AutoProcessor.from_pretrained(cfg.framework.qwenvl.base_vlm, local_files_only=True)
    processor.tokenizer.padding_side = 'left'
    qwen = _QWen3_VL_Interface.__new__(_QWen3_VL_Interface)
    torch.nn.Module.__init__(qwen)
    qwen.processor, qwen.config = processor, cfg
    qwen.model = SimpleNamespace(device=torch.device('cpu'))
    framework = EventVLA.__new__(EventVLA)
    torch.nn.Module.__init__(framework)
    framework.config = cfg
    injection = cfg.framework.memory_buffer.qwen_memory_injection
    framework.memory_injection_enabled = injection.enabled
    framework.memory_injection_mode = injection.mode
    framework.keyframe_image_position = injection.keyframe_image_position
    framework.max_keyframe_images = injection.max_keyframe_images
    framework.chunk_len = (cfg.framework.action_model.future_action_window_size +
                           cfg.framework.action_model.past_action_window_size + 1)
    framework.action_token = '\U0001f50d'
    checks, rows, parents = {}, [], []
    for fixture in fixtures:
        seed, step = fixture['seed'], fixture['step']
        archive, anchor_keys = fixture['archive'], fixture['anchor_keys']
        fingerprint = fixture['archive_fingerprint']

        def prepare(payload):
            example = payload['example']
            anchors, metas = example['image'], example['image_metas']
            expected = tuple(anchor_keys) + tuple(payload['selected'])
            images = resize_images([[Image.fromarray(a) for a in anchors]],
                                   target_size=list(cfg.datasets.vla_data.image_size))
            images, image_metas = framework._augment_batch_images_with_keyframe_images(
                images, [metas], [example])
            assert [image_hash(np.asarray(im)) for im in images[0]] == [
                image_hash(im) for im in archive.read(expected, step)], 'unexpected resize/order/truncation'
            start = time.perf_counter()
            inputs = qwen.build_qwenvl_inputs(images, framework._build_vla_prompts([example['lang']]),
                image_metas=image_metas, use_image_role_text=bool(injection.use_image_role_text))
            elapsed = time.perf_counter() - start
            grid = inputs['image_grid_thw'].tolist()
            merge = int(processor.image_processor.merge_size)
            visual = sum(int(np.prod(g)) // (merge ** 2) for g in grid)
            image_token = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
            actual = int((inputs['input_ids'] == image_token).sum())
            assert visual == actual and len(grid) == len(expected)
            trace = InputTrace(payload['stage'], fingerprint, step, expected, visual,
                               int(inputs['attention_mask'].sum()), elapsed,
                               digest(dict(prompt=framework._build_vla_prompts([example['lang']]),
                                   config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                                   extra_policy_state=None)))
            trace.validate(archive)
            return trace, grid

        proposal_input, _ = prepare(fixture['payloads'][0])
        # Input prepared only: unknown action/timing must block a live action arm.
        blocked = False
        try:
            validate_proposal(Proposal(proposal_input, 0, '0' * 64, None, 'input_only'),
                              archive, step, anchor_keys, False, proposal_input.context_sha256)
        except ValueError:
            blocked = True
        checks[f'{seed}_unexecuted_proposal_rejected'] = blocked
        full, _ = prepare(fixture['payloads'][1])
        for payload in fixture['payloads'][2:]:
                budget, strategy = payload['event_read_cap'], payload['method']
                final, grid = prepare(payload)
                cost = ReadBudget(len(proposal_input.observation_keys) + len(final.observation_keys),
                                  proposal_input.visual_tokens + final.visual_tokens,
                                  proposal_input.sequence_tokens + final.sequence_tokens, 2)
                ledger = ReadLedger(archive, step, cost)
                ledger.charge(proposal_input)
                checks[f'{seed}_{strategy}_{budget}_sum'] = ledger.charge(final) == cost
                checks[f'{seed}_{strategy}_{budget}_archive_unchanged'] = archive.fingerprint(step) == fingerprint
                rows.append(dict(fixture_seed=seed, method=strategy, event_read_cap=budget,
                    selected_steps=payload['example']['memory_keyframe_steps'],
                    proposal_input=proposal_input.__dict__, final_input=final.__dict__,
                    final_image_grid_thw=grid, two_pass_input_cost=cost.__dict__,
                    single_pass_all_event_input=full.__dict__,
                    two_pass_visual_ratio_to_all_events=cost.visual_tokens/full.visual_tokens,
                    policy_seconds=None, encoder_seconds=None, retrieval_seconds=None,
                    end_to_end_seconds=None, policy_forward_executed=False))
        parents.append({k: v for k, v in fixture.items() if k not in ('archive', 'payloads', 'candidates', 'anchor_keys')})
    client_file = UPSTREAM/'examples/RoboTwin-Mem/eval_files/model2robotwin_mem_interface.py'
    sources = [Path(__file__), ROOT/'src/recoverability/archive_readout.py', config_path,
               client_file, UPSTREAM/'eventvla/model/framework/EventVLA.py',
               UPSTREAM/'eventvla/model/modules/vlm/QWen3.py',
               Path(cfg.framework.qwenvl.base_vlm)/'preprocessor_config.json']
    report = dict(status='passed' if all(checks.values()) else 'failed', checks=checks, variants=rows,
        parents=parents, run_dir=str(run_dir), actual_upstream_client_and_input_builders=True,
        processor=type(processor).__name__, torch_version=torch.__version__,
        cpu_threads=2, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        trusted_fixture_sha256=hashlib.sha256(fixtures_file.read_bytes()).hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        policy_inference_executed=False, new_independent_research_samples=0,
        cost_scope='prepared_inputs_for_proposed_calls', actual_policy_calls=0,
        formal_training_eligible=False, empirical_method_advantage_established=False,
        limitation='Processor accounting only; no encoder/policy latency, decision quality, or complete online archive claim.')
    raw = json.dumps(report, indent=2) + '\n'
    (run_dir/'report.json').write_text(raw)
    (ROOT/'reports/latest_archive_readout.json').write_text(raw)
    print(json.dumps(dict(status=report['status'], checks=len(checks), variants=len(rows),
        run_dir=str(run_dir), first_variant=rows[0]['two_pass_input_cost'])))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--client-fixtures':
        build_client_fixtures(Path(sys.argv[2]))
    else:
        main()
