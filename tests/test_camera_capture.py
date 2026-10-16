from types import SimpleNamespace
from recoverability.camera_capture import serialize_camera_capture


def test_capture_completes_before_next_camera():
    calls = []
    class Camera:
        def __init__(self, name): self.name = name
        def take_picture(self): calls.append((self.name, 'capture'))
        def get_picture(self, field): calls.append((self.name, field))
    native = lambda: None
    env = SimpleNamespace(cameras=SimpleNamespace(
        collect_wrist_camera=True, left_camera=Camera('left'), right_camera=Camera('right'),
        static_camera_list=[Camera('head')], update_picture=native))
    assert serialize_camera_capture(env) is native
    env.cameras.update_picture()
    assert calls == [(v, op) for v in ('left', 'right', 'head') for op in ('capture', 'Color')]
