"""Opt-in serialized GPU readback; retain each camera's capture count/order.

This is an operational workaround, not a proven renderer/driver root cause.
Native parent image and physics hashes must still be checked by each worker.
"""


def serialize_camera_capture(env):
    cameras = env.cameras
    original = cameras.update_picture

    def update_picture():
        ordered = []
        if cameras.collect_wrist_camera:
            ordered.extend((cameras.left_camera, cameras.right_camera))
        ordered.extend(cameras.static_camera_list)
        for camera in ordered:
            camera.take_picture()
            # Finish this readback before submitting another camera.
            camera.get_picture('Color')

    cameras.update_picture = update_picture
    return original
