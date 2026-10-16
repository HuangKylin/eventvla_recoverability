from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from recoverability.native_recovery import RecoveryProtocol, recovery_task_mode, joint_bridge, ARM


def test_recovery_open_does_not_submit_but_commit_still_scores():
    calls=[]; steps=[]
    env=SimpleNamespace(update_state_transition=lambda:calls.append('native'), scene=SimpleNamespace(step=lambda:steps.append(1)))
    original=env.scene; protocol=RecoveryProtocol(); protocol.begin()
    with recovery_task_mode(env, protocol):
        env.scene.step(); env.update_state_transition()
        assert not calls and len(steps)==1 and protocol.physics_steps==1
        protocol.commit(colours_hidden=True, grippers_open=True)
        env.update_state_transition()
        assert calls==['native']
    assert env.scene is original
    env.update_state_transition(); assert len(calls)==2


def test_cannot_recover_after_failure_or_resubmit():
    p=RecoveryProtocol()
    with pytest.raises(RuntimeError):p.begin(already_failed=True)
    p.commit(colours_hidden=True,grippers_open=True)
    with pytest.raises(RuntimeError):p.begin()
    with pytest.raises(RuntimeError):p.commit(colours_hidden=True,grippers_open=True)


@pytest.mark.parametrize('hidden,open_',[(False,True),(True,False),(False,False)])
def test_public_commit_gate(hidden,open_):
    p=RecoveryProtocol();p.begin()
    with pytest.raises(RuntimeError):p.commit(colours_hidden=hidden,grippers_open=open_)
    assert p.mode=='recover'


def test_budget_and_exception_restore():
    env=SimpleNamespace(update_state_transition=lambda:None, scene=SimpleNamespace(step=lambda:None))
    update,scene=env.update_state_transition,env.scene
    p=RecoveryProtocol(max_physics_steps=1);p.begin()
    with pytest.raises(RuntimeError):
        with recovery_task_mode(env,p):env.scene.step();env.scene.step()
    assert env.scene is scene and env.update_state_transition is update and p.physics_steps==1


def test_bridge_velocity_endpoints_and_finite_inputs():
    start=np.zeros(14);end=np.linspace(-1,1,14)
    rows=list(joint_bridge(start,end));q=np.array([r[0] for r in rows]);v=np.array([r[1] for r in rows])
    assert np.max(np.abs(v)) <= .4+1e-10
    assert np.max(np.abs(np.diff(q,axis=0)))/.004 <= .4+1e-10
    assert np.array_equal(q[-1],end[ARM]) and not v[-1].any()
    with pytest.raises(ValueError):list(joint_bridge(np.full(14,np.nan),end))


@pytest.mark.parametrize('always_miss',[False,True])
def test_rgb_retry_is_bounded_and_stops_after_evidence(monkeypatch,always_miss):
    import recoverability.native_recovery as recovery
    import recoverability.fixed_skills as skills
    import recoverability.snapshot as snapshot
    from recoverability.cue_reader import ReadResult
    state=np.zeros(14);state[[6,13]]=1
    monkeypatch.setattr(skills,'measured_proprioception',lambda robot:state.copy())
    monkeypatch.setattr(recovery,'bridge_to',lambda *a,**k:{'physics_steps':0})
    monkeypatch.setattr(snapshot,'apply_drive_vector',lambda *a:None)
    ticks=[];seen=[]
    skill=SimpleNamespace(start_proprioception=state,drives=[None],observe_after_steps=1)
    controller=SimpleNamespace(skills={i:skill for i in range(4)},select=lambda *a:(skill,0.),
        read=lambda slot,rgb:ReadResult('red' if rgb else None,20 if rgb else 0,1. if rgb else 0.))
    def observe(kind,slot):
        attempt=1+seen.count((kind,slot));seen.append((kind,slot))
        visible=kind=='visible' and not always_miss and (slot!=0 or attempt>1)
        return {'images':{'head_camera':visible}}
    env=SimpleNamespace(robot=None,scene=SimpleNamespace(step=lambda:ticks.append(1)))
    if always_miss:
        result=recovery.sweep_recover(env,controller,observe=observe)
        assert len(ticks)==12 and result['recovered_slots']==[] and result['unrecovered_slots']==list(range(4))
        assert all(len(row['attempts'])==3 for row in result['records'])
    else:
        result=recovery.sweep_recover(env,controller,observe=observe)
        assert len(ticks)==5 and len(result['records'][0]['attempts'])==2
        assert all(len(row['attempts'])==1 for row in result['records'][1:])
