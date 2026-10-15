"""Check that reset depends on actual physics, never requested action identity."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def latch_methods():
    # Extract the environment methods without importing SAPIEN in the unit-test
    # process. Real simulator behavior is separately checked by K/R rollouts.
    path=Path(__file__).parents[1]/'src/recoverability/action_latch_task.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    funcs=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('arm_latches','tick_latches')]
    scope={'np':np}
    exec(compile(ast.Module(body=funcs,type_ignores=[]),str(path),'exec'),scope)
    return scope


def test_only_moved_cover_resets_and_reset_is_one_shot():
    methods=latch_methods()
    positions=[np.array([x,0.,.741]) for x in range(4)]
    covers=[SimpleNamespace(get_pose=lambda p=p:SimpleNamespace(p=p)) for p in positions]
    changes=[]
    env=SimpleNamespace(covers=covers,latch_events=[],latch_armed=False,
                        _set_indicator_colour=lambda s,c:changes.append(s))
    methods['arm_latches'](env)
    methods['tick_latches'](env)
    assert changes==[]
    # A different cover than a hypothetical requested left action moves.
    positions[3][2]+=.05
    methods['tick_latches'](env);methods['tick_latches'](env)
    assert changes==[3] and len(env.latch_events)==1
    env.latch_armed=False
    positions[0][2]+=.05
    methods['tick_latches'](env)
    assert changes==[3]
