import mujoco
from copy import copy
from utils.config import ConfigModelBase
import numpy as np
from utils.visualize.vis_mujoco import MujocoViewer

import mujoco
from copy import copy
from utils.config import ConfigModelBase
import numpy as np


def make_camera():
    cam = mujoco.MjvCamera()
    cam.lookat[0] = -1.2
    cam.lookat[1] = 1.2
    cam.lookat[2] = 1.2
    cam.distance = 0  # 2.5
    # cam.elevation = -25
    cam.elevation = 0
    # cam.azimuth = 135
    # cam.azimuth = 5
    cam.azimuth = 0
    return cam


def make_viewer(model, data):
    viewer = MujocoViewer(model, data, hide_menu=False)
    # viewer.cam = make_camera()
    return viewer


def loop_mj_vis(
    model: ConfigModelBase,
    controls: list = None,
    q: np.array = None,
    q_dot: np.array = None,
    num_motors: int = 4,
):

    mj_model = copy(model.mj_model)
    mj_data = copy(model.mj_data)
    pin_model = copy(model.pin_model)

    if controls is not None:
        n = len(controls)
    else:
        assert (
            np.shape(q)[0] == np.shape(q_dot)[0]
        ), "Lengths of q and q_dot must be equal."
        n = np.shape(q)[0]

    def run_step(controls, q, q_dot):
        if controls is not None:

            def one_step(controls, q, q_dot, step):
                mj_data.ctrl[:num_motors] = controls[step][:num_motors]

        elif q is not None and q_dot is not None:

            def one_step(controls, q, q_dot, step):
                mj_data.qpos[:] = q[step, :]
                mj_data.qvel[:] = q_dot[step, :]

        else:
            raise NotImplementedError("Give either list of controls or (q AND q_dot)")

        return one_step

    viewer = make_viewer(mj_model, mj_data)
    step = 0
    # reset the model and execute the computed torques
    mujoco.mj_resetData(mj_model, mj_data)

    mj_data.qpos[:] = model.q_config
    mj_data.qvel[:] = np.zeros(pin_model.nv)

    one_step = run_step(controls, q, q_dot)

    mujoco.mj_step(mj_model, mj_data)
    while step < n:
        one_step(controls, q, q_dot, step)
        mujoco.mj_step(mj_model, mj_data)
        viewer.render()
        step += 1
        if step % n == 0:
            mujoco.mj_resetData(mj_model, mj_data)
            mj_data.qpos[:] = model.q_config
            mj_data.qvel[:] = np.zeros(pin_model.nv)
            mujoco.mj_step(mj_model, mj_data)
            step = 0
