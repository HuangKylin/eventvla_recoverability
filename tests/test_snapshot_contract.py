import numpy as np
import pytest

from recoverability.snapshot import capture,max_error


def test_vla_or_planner_snapshots_are_rejected_before_touching_the_environment():
    with pytest.raises(ValueError,match='Only fixed_drive_replay'):
        capture(object(),{},mode='full_vla')


@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_nonfinite_physics_cannot_pass_restore_gate(value):
    assert max_error(np.array([value]),np.zeros(1))==float('inf')


def test_changed_state_shape_cannot_pass_restore_gate():
    assert max_error(np.zeros(3),np.zeros(4))==float('inf')
