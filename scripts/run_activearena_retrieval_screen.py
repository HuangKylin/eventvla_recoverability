"""Bounded CPU retrieval screen on three public demonstrations. No VLA rollout.

Oracle labels and scheduling annotations are confined to evaluation. All actual
retrieval methods read images, global instruction and causal gripper state only.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
from datetime import datetime,timezone
from pathlib import Path
from io import BytesIO
import json,hashlib,time

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_query_kind(states):
    # Closing either hand indicates transport. The extra progress state records
    # which can remains without consulting semantic phase labels or actions.
    left_closed_seen=False
    left_reopened=False
    for state in states:
        if state[7] <= 0.2:left_closed_seen=True
        if left_closed_seen and state[7] >= 0.8:left_reopened=True
    state=states[-1]
    carrying=(state[15]<=0.2) if left_reopened else (state[7]<=0.2)
    return ('box' if carrying else 'can'),dict(left_closed_seen=left_closed_seen,left_reopened=left_reopened,
        current_left_gripper=float(state[7]),current_right_gripper=float(state[15]))


def main():
    import numpy as np
    import pyarrow.parquet as pq
    from PIL import Image,ImageDraw
    import torch
    from transformers import AutoModel,AutoProcessor
    torch.set_num_threads(2)
    contract_path=ROOT/'configs/activearena_can_retrieval_pilot_v1.json'
    contract=json.loads(contract_path.read_text())
    base=ROOT/'data/public_pilots/activearena_can_transfer_v1'
    task=base/contract['task']
    model_dir=ROOT/'checkpoints/siglip2-base-patch16-224'
    manifest=json.loads((model_dir/'local_download_manifest.json').read_text())
    assert manifest['revision']==contract['encoder_revision']
    downloads=json.loads((base/'pilot_download_manifest.json').read_text())
    assert downloads['revision']==contract['dataset_revision']
    for record in downloads['files']:assert sha(base/record['path'])==record['sha256']
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_activearena_retrieval_screen')
    run.mkdir(parents=True,exist_ok=False)
    (run/'protocol.json').write_text(json.dumps(contract,indent=2)+'\n')
    loading=time.perf_counter()
    model=AutoModel.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float32,
                                    trust_remote_code=False).eval().to('cpu')
    processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
    assert not type(processor.image_processor).__name__.endswith('Fast'), 'revalidate if image preprocessing changes'
    load_seconds=time.perf_counter()-loading
    tasks={x['task_index']:x['task'] for x in map(json.loads,(task/'meta/tasks.jsonl').read_text().splitlines())}
    queries=[contract['query_texts']['can'],contract['query_texts']['box']]
    data={}
    for ep in contract['episodes']:
        d=pq.read_table(task/f'data/chunk-000/episode_{ep:06d}.parquet').to_pydict()
        data[ep]=d
        queries.append(tasks[int(np.asarray(d['task_index'][0]).item())])
    start=time.perf_counter()
    with torch.inference_mode():
        text_input=processor(text=queries,padding='max_length',truncation=True,max_length=64,return_tensors='pt')
        text=model.get_text_features(**text_input)
        text=torch.nn.functional.normalize(text,dim=-1).cpu().numpy()
    text_seconds=time.perf_counter()-start
    records=[];episodes=[];checks={}
    for ep,d in data.items():
        indices=list(range(0,len(d['frame_index']),contract['observation_stride']))
        start=time.perf_counter()
        images=[Image.open(BytesIO(d['observation.images.camera_head'][i]['bytes'])).convert('RGB') for i in indices]
        decode_seconds=time.perf_counter()-start
        start=time.perf_counter();features=[]
        with torch.inference_mode():
            for offset in range(0,len(images),4):
                inputs=processor(images=images[offset:offset+4],return_tensors='pt')
                encoded=model.get_image_features(**inputs)
                features.append(torch.nn.functional.normalize(encoded,dim=-1).cpu().numpy())
        features=np.concatenate(features);encode_seconds=time.perf_counter()-start
        np.savez_compressed(run/f'episode_{ep}_features.npz',indices=np.asarray(indices),features=features)
        subtask=np.asarray(d['subtask_index']).reshape(-1)
        # Privileged labels only choose fixed probe positions and evaluate hits.
        annotation=contract['visual_annotation']['episodes'][str(ep)]
        state=np.asarray(d['observation.state'])
        for phase in (2,3,4):
            transition=int(np.flatnonzero(subtask==phase)[0])
            t=next(i for i in indices if i>=transition)
            past=[i for i in indices if i<t]
            pos=[indices.index(i) for i in past]
            target='second_can' if phase==3 else 'box'
            evidence=set(annotation[target+'_frames'])
            relevant=sorted(set(past)&evidence)
            kind,public_state=public_query_kind(state[indices[:indices.index(t)+1]])
            oracle_kind='can' if phase==3 else 'box'
            checks[f'ep{ep}_phase{phase}_causal_archive']=all(i<t for i in past)
            strategies={'recent':None,'siglip_global_instruction':2+ep,
                        'siglip_public_gripper_query':0 if kind=='can' else 1,
                        'oracle_subtask_query_diagnostic_only':0 if oracle_kind=='can' else 1}
            for method,column in strategies.items():
                start=time.perf_counter()
                scores=None if column is None else features[pos]@text[column]
                ordered=(list(reversed(past)) if column is None else
                    [past[j] for j in sorted(range(len(past)),key=lambda j:(-float(scores[j]),-past[j]))])
                search_seconds=time.perf_counter()-start
                selected=ordered[:2]
                assert set(selected)<=set(past)
                records.append(dict(episode=ep,decision_frame=t,evaluator_subtask=phase,method=method,
                    query_kind=kind if method=='siglip_public_gripper_query' else None,
                    public_gripper_state=public_state if method=='siglip_public_gripper_query' else None,
                    evaluator_target=target,history_images=len(past),relevant_past_frames=relevant,
                    historically_answerable=bool(relevant),current_view_sufficient=t in evidence,
                    selected_frames=selected,hit_at_1=(bool(set(selected[:1])&evidence) if relevant else None),
                    hit_at_2=(bool(set(selected)&evidence) if relevant else None),
                    search_dot_and_sort_seconds=search_seconds,oracle_query=method.startswith('oracle')))
        episodes.append(dict(episode=ep,raw_frames=len(d['frame_index']),delivered_frames=len(indices),
            decode_seconds=decode_seconds,archive_encode_seconds=encode_seconds,
            decoded_pixel_sha256=[hashlib.sha256(np.asarray(im).tobytes()).hexdigest() for im in images],
            raw_parquet_sha256=sha(task/f'data/chunk-000/episode_{ep:06d}.parquet')))
        # Diagnostic figure: current view + each method's top retrieved frame.
        for phase in (2,3,4):
            group=[r for r in records if r['episode']==ep and r['evaluator_subtask']==phase]
            canvas=Image.new('RGB',(320*5,272),'white');draw=ImageDraw.Draw(canvas)
            selections=[('CURRENT',group[0]['decision_frame'])]+[(r['method'],r['selected_frames'][0]) for r in group]
            for j,(name,i) in enumerate(selections):
                canvas.paste(images[indices.index(i)].resize((320,240)),(j*320,32))
                draw.text((j*320+3,3),f'{name}\nframe={i}',fill='black')
            canvas.save(run/f'ep{ep}_phase{phase}_retrieval.png')
        print(json.dumps(dict(episode=ep,frames=len(indices),encode_seconds=encode_seconds)),flush=True)
    summary={}
    for method in contract['methods']:
        group=[r for r in records if r['method']==method]
        answerable=[r for r in group if r['historically_answerable']]
        summary[method]=dict(probes=len(group),historically_answerable=len(answerable),
            historical_evidence_absent=len(group)-len(answerable),
            hit_at_1=sum(r['hit_at_1'] for r in answerable),hit_at_2=sum(r['hit_at_2'] for r in answerable),
            current_view_sufficient=sum(r['current_view_sufficient'] for r in group),
            history_needed_probes=sum(not r['current_view_sufficient'] for r in answerable),
            history_needed_hit_at_1=sum(r['hit_at_1'] for r in answerable if not r['current_view_sufficient']),
            history_needed_hit_at_2=sum(r['hit_at_2'] for r in answerable if not r['current_view_sufficient']))
    report=dict(status='complete',run_dir=str(run),protocol_sha256=sha(contract_path),
        script_sha256=sha(Path(__file__)),encoder_manifest=manifest,
        cpu_threads=2,device='cpu',dtype='float32',model_load_seconds=load_seconds,
        query_encode_seconds=text_seconds,query_count=len(queries),episodes=episodes,
        checks=checks,records=records,summary=summary,
        task_success_measured=False,action_conditioned_method_evaluated=False,
        causal_action_pairs=0,formal_training_eligible=False,
        annotation_limit='Single assistant visual annotation, no independent adjudication. Training-demo screening only.',
        efficiency_limit='Precomputed embeddings are used for semantic screening; full-episode index encode time reported, no online or GPU latency claim.')
    raw=json.dumps(report,indent=2)+'\n'
    (run/'report.json').write_text(raw)
    (ROOT/'reports/latest_activearena_retrieval_screen.json').write_text(raw)
    print(json.dumps(dict(run_dir=str(run),summary=summary)),flush=True)


if __name__=='__main__':main()
