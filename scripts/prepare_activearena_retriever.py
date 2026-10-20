"""Fetch a single frozen public image/text encoder, without changing environments."""
import os
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='45'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
repo='google/siglip2-base-patch16-224'
revision='75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
api=HfApi();info=api.model_info(repo,revision=revision,files_metadata=True)
assert info.sha==revision
out=ROOT/'checkpoints/siglip2-base-patch16-224'
out.mkdir(parents=True,exist_ok=True)
allowed={'config.json','preprocessor_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors','README.md'}
files=[f for f in info.siblings if f.rfilename in allowed]
assert sum(f.size or 0 for f in files)<2_500_000_000
print(json.dumps(dict(model=repo,revision=info.sha,bytes=sum(f.size or 0 for f in files))),flush=True)
records=[]
for file in files:
    path=Path(hf_hub_download(repo,file.rfilename,revision=info.sha,local_dir=out))
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    expected=(file.lfs or {}).get('sha256') if isinstance(file.lfs,dict) else getattr(file.lfs,'sha256',None)
    if expected:assert h.hexdigest()==expected
    records.append(dict(name=file.rfilename,bytes=path.stat().st_size,sha256=h.hexdigest()))
    print(file.rfilename,path.stat().st_size,flush=True)
(out/'local_download_manifest.json').write_text(json.dumps(dict(repo=repo,revision=info.sha,files=records),indent=2)+'\n')
