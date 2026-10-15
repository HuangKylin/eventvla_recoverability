from types import SimpleNamespace
import numpy as np
import pytest
from recoverability.fixed_skills import SharedRecoveryController, Skill, deny_oracle_actions


def test_start_guard_uses_proprioception_and_rejects_out_of_calibration_state():
    controller=SharedRecoveryController.__new__(SharedRecoveryController)
    controller.start_tolerance=0.15
    controller.skills={0:Skill(0,np.zeros((3,8)),1,np.zeros(14))}
    state=np.zeros(14)
    assert controller.select(0,state)[1]==0
    state[0]=0.3
    with pytest.raises(RuntimeError,match='start envelope'):
        controller.select(0,state)


def test_runtime_expert_entry_points_raise_and_are_restored_after_audit():
    answer=lambda *a,**kw:'privileged'
    names=('inspect_cover','grasp_actor','place_actor','get_grasp_pose',
           'choose_grasp_pose','get_place_pose','_target_pose','get_keyframe_oracle_info')
    env=SimpleNamespace(**{name:answer for name in names},
                        robot=SimpleNamespace(left_plan_path=answer,right_plan_path=answer))
    with deny_oracle_actions(env) as audit:
        with pytest.raises(RuntimeError,match='forbidden'):
            env.grasp_actor()
        with pytest.raises(RuntimeError,match='forbidden'):
            env.robot.left_plan_path()
        assert audit['forbidden_calls']==2
    assert env.grasp_actor()=='privileged'
    assert env.robot.left_plan_path()=='privileged'
