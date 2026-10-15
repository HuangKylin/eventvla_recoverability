"""Train the first actual scorer; report held-out-family pilot limits explicitly."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from recoverability.scorer import DeletionScorer,public_features
    from recoverability.branch_protocol import validate_split_records
    collection=json.loads(a.collection.read_text())
    if collection['status']!='complete' or not collection['all_pairs_matched'] or not collection['all_repeated_R_matched']:
        raise RuntimeError('Only complete matched K/R collections can supply labels')
    rows=collection['labels']
    seeds=collection['seeds']
    if len(seeds)!=4: raise ValueError('This predeclared pilot uses two train, one validation, one test family')
    split_map={seeds[0]:'train',seeds[1]:'train',seeds[2]:'validation',seeds[3]:'test'}
    x=[]
    for row in rows:
        row['split']=split_map[row['seed']]
        with np.load(row['public_action'],allow_pickle=False) as action_data:
            x.append(public_features(np.asarray(Image.open(row['evidence_image']).convert('RGB')),
                row['evidence_slot'],row['ordinal'],action_data['sampled_drives']))
    validate_split_records(rows)
    x=np.stack(x);y=np.asarray([r['delta_cost'] for r in rows],np.float32)
    ids={s:np.array([i for i,r in enumerate(rows) if r['split']==s]) for s in ('train','validation','test')}
    mean=x[ids['train']].mean(0);scale=x[ids['train']].std(0)
    scale=np.maximum(scale,.01).astype(np.float32)
    z=torch.tensor((x-mean)/scale,dtype=torch.float32);target=torch.tensor(y)
    run=ROOT/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_pilot_scorer')
    run.mkdir(parents=True)
    weights=ROOT/'checkpoints/scorers'/run.name;weights.mkdir(parents=True)
    report=dict(status='running',run_dir=str(run),collection=str(a.collection),
        collection_sha256=hashlib.sha256(a.collection.read_bytes()).hexdigest(),
        scope='tiny_fixed_geometry_mechanism_pilot_not_formal_EventVLA_training',
        families_by_split={s:[f'action_latch_{k}' for k,v in split_map.items() if v==s] for s in ids},
        examples_by_split={s:len(v) for s,v in ids.items()},
        inputs='RGB ROI statistics, public evidence slot and age, 16 sampled motor target vectors',
        no_action_slot_label_input=True,backbone_updated=False,models={})
    predictions={}
    torch.set_num_threads(2)
    for ablated,name in ((False,'learned_action'),(True,'learned_no_action')):
        torch.manual_seed(42)
        model=DeletionScorer(x.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.001)
        inputs=z.clone()
        if ablated: inputs[:,11:]=0
        best=float('inf');best_state=None;best_epoch=0
        for epoch in range(1500):
            model.train();opt.zero_grad()
            loss=(model(inputs[ids['train']])-target[ids['train']]).square().mean()
            loss.backward();opt.step()
            model.eval()
            with torch.no_grad():
                val=float((model(inputs[ids['validation']])-target[ids['validation']]).square().mean())
            if val<best:
                best=val;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};best_epoch=epoch+1
        model.load_state_dict(best_state)
        with torch.inference_mode(): pred=model(inputs).numpy()
        predictions[name]=pred
        path=weights/(name+'.pt')
        torch.save(dict(protocol='physical_latch_pilot_v1',width=x.shape[1],state_dict=best_state,
            mean=torch.tensor(mean),scale=torch.tensor(scale),action_ablated=ablated),path)
        report['models'][name]=dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            parameters=sum(p.numel() for p in model.parameters()),selected_epoch=best_epoch,
            mse={s:float(np.mean((pred[ix]-y[ix])**2)) for s,ix in ids.items()})
    # Offline policy comparison uses independently measured K/R outcomes. It is
    # not a rollout of a shared sequential multi-query controller.
    comparisons=[]
    for seed in seeds:
        if split_map[seed]!='test': continue
        for action in collection['actions']:
            group=[i for i,r in enumerate(rows) if r['seed']==seed and r['action_slot']==action]
            for method in ('fifo','oldest','action_rule','archive','structured','learned_action','learned_no_action','learned_shuffled_action'):
                if method in ('archive','structured'): kept=group
                elif method=='fifo': kept=[max(group,key=lambda i:rows[i]['ordinal'])]
                elif method=='oldest': kept=[min(group,key=lambda i:rows[i]['ordinal'])]
                elif method=='action_rule': kept=[next(i for i in group if rows[i]['evidence_slot']==action)]
                elif method=='learned_shuffled_action':
                    other=next(v for v in collection['actions'] if v!=action)
                    paired={rows[i]['evidence_slot']:i for i in range(len(rows)) if rows[i]['seed']==seed and rows[i]['action_slot']==other}
                    kept=[max(group,key=lambda i:float(predictions['learned_action'][paired[rows[i]['evidence_slot']]]))]
                else: kept=[max(group,key=lambda i:float(predictions[method][i]))]
                outcomes=[rows[i]['keep' if i in kept else 'recover'] for i in group]
                comparisons.append(dict(seed=seed,action_slot=action,method=method,
                    kept_slots=[rows[i]['evidence_slot'] for i in kept],
                    correct=sum(q['success'] for q in outcomes),queries=len(outcomes),
                    recovery_steps=sum(q['recovery_steps'] for q in outcomes),
                    comparison_scope='offline_composition_of_independent_query_branches'))
    report.update(status='complete',comparisons=comparisons,
        limitations=['Only four color/order families at one fixed geometry and two motor programs.',
                     'No held-out action or geometry generalization claim.',
                     'Strong action rule solves the designed mechanism; archives retain every color.',
                     'Formal EventVLA K/R labels and scorer integration remain separate work.'])
    (run/'split_labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    text=json.dumps(report,indent=2)
    (run/'report.json').write_text(text);(ROOT/'reports/latest_pilot_scorer.json').write_text(text)
    print(text)


if __name__=='__main__': main()
