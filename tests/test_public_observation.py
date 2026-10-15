import numpy as np

from recoverability.sim_runtime import public_observation


def test_oracle_pose_and_metadata_never_enter_public_observation():
    image=np.ones((4,5,3),dtype=np.uint8)
    obs=dict(observation={view:dict(rgb=image,actor_segmentation=np.ones((4,5)),
                                    oracle_colour='secret')
                           for view in ('head_camera','left_camera','right_camera')},
             joint_action=dict(vector=np.ones(14),target_pose='secret'),
             oracle=dict(answer='secret'),actor_poses='secret')
    clean=public_observation(obs)
    assert set(clean)=={'images','proprioception'}
    assert set(clean['images'])=={'head_camera','left_camera','right_camera'}
    assert all(isinstance(v,np.ndarray) for v in clean['images'].values())
    image[:]=0
    obs['joint_action']['vector'][:]=0
    assert clean['images']['head_camera'].min()==1
    assert clean['proprioception'].min()==1
