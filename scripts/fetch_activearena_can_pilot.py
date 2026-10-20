"""Download three predetermined public training demonstrations, pinned by revision.

Not an evaluation split. No execution of downloaded Python or remote model code.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DATASET='leeibo/ActiveArena-Data'
REVISION='4fc698d7b3c1871c342181a5fa38f837da945a80'
TASK='place_cans_plasticbox_rotate_view'
EPISODES=(0,1,2)
OUT=ROOT/'data/public_pilots/activearena_can_transfer_v1'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    names=['meta/info.json','meta/tasks.jsonl','meta/episodes.jsonl','meta/astribot_subtask_metadata.json']
    names += [f'data/chunk-000/episode_{episode:06d}.parquet' for episode in EPISODES]
    for name in names:
        path=OUT/TASK/name
        path.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{TASK}/{name}'
        if not path.exists():
            limit=64*1024*1024
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'EventVLA-research-pilot'}),timeout=45) as response:
                length=response.headers.get('Content-Length')
                if length and int(length)>limit:raise ValueError('pilot per-file size cap exceeded')
                raw=response.read(limit+1)
            if len(raw)>limit:raise ValueError('pilot per-file size cap exceeded')
            path.write_bytes(raw)
        raw=path.read_bytes()
        records.append(dict(url=url,path=str(path.relative_to(OUT)).replace('\\','/'),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        print(name,len(raw),flush=True)
    report=dict(dataset=DATASET,revision=REVISION,task=TASK,episodes=list(EPISODES),
        selection='first_three_training_episode_indices_chosen_before_inspection',
        downloaded_utc=datetime.now(timezone.utc).isoformat(),files=records)
    (OUT/'pilot_download_manifest.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
