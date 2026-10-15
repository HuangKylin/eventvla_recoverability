"""Project-local deterministic, strict-load EventVLA server for paired probes."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/EventVLA/EventVLA'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    import torch
    from eventvla.model.framework.base_framework import baseframework
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    checkpoint=ROOT/'checkpoints/runtime/EventVLA-RoboTwin-MeM/final_model/pytorch_model.pt'
    model=baseframework.from_pretrained(str(checkpoint),strict_checkpoint=True)
    model=model.to(device='cuda',dtype=torch.bfloat16).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    class SeededPolicy:
        def reset_memory(self): model.reset_memory()
        def predict_action(self,examples=None,paired_seed=None,**kwargs):
            if paired_seed is None: raise ValueError('Explicit paired inference seed required')
            random.seed(paired_seed)
            np.random.seed(paired_seed)
            torch.manual_seed(paired_seed)
            torch.cuda.manual_seed_all(paired_seed)
            model.reset_memory()
            with torch.inference_mode():
                result=model.predict_action(examples=examples,**kwargs)
            result['paired_seed']=paired_seed
            result['server_memory_reset']=True
            return result
    a.report.write_text(json.dumps(dict(status='ready',checkpoint=str(checkpoint),
        strict_load=model.checkpoint_load_report,all_parameters_frozen=True,
        server_reset_per_request=True,host='127.0.0.1',port=a.port),indent=2))
    WebsocketPolicyServer(SeededPolicy(),host='127.0.0.1',port=a.port,idle_timeout=1800).serve_forever()


if __name__=='__main__': main()
