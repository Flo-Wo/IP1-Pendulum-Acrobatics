from utils.experiments.trajectory import (
    PenaltyTrajectory,
    mpc_state_space,
)
from utils.experiments.trajectory import ExperimentTrajectory
from utils.trajectories import circle, spiral
from utils.visualize import plot_trajectory, plot_single_traj
from utils.config import ConfigWAM
import numpy as np
import matplotlib.pyplot as plt
from utils.visualize import loop_mj_vis, visualize
from utils.pin_helper import (
    task_space_traj,
    angles_traj,
)
from utils.data_handling import DataSaver
from utils.costs import Penalty
import crocoddyl
import mujoco


model = ConfigWAM("rotated")
mode = "spiral"
smoothing = True
if smoothing:
    n = 3500
else:
    n = 3000

if mode == "circle":
    traj_x = circle(0.1, model.pend_end_world(), 4, n, "bottom", plane="y")
elif mode == "spiral":
    traj_x = spiral(0.1, 0.12, model.pend_end_world(), 2, n, "bottom", plane="z")
else:
    raise NotImplementedError

if smoothing:
    end_smooth = 500
    smooth_factor = np.linspace(0, 1, end_smooth, endpoint=True)
    # smooth_factor = 1 / (np.exp(1) - 1) * np.exp(
    #     np.linspace(0, 1, end_smooth, endpoint=True)
    # ) - (1 / (np.exp(1) - 1))
    # traj_x[:500, :] *= np.vstack((smooth_factor, smooth_factor, smooth_factor)).T
    traj_x[:end_smooth, 0] *= smooth_factor
    pass

experiment = ExperimentTrajectory(24, traj_x, 24, traj_rot=n * [np.eye(3)])


q_pen_weights = np.concatenate((np.ones(6), np.zeros(6)))
v_pen_weights = np.concatenate((np.zeros(6), np.ones(6)))

penalty_ddp = PenaltyTrajectory(
    u_pen=Penalty(1e-2),
    x_pen=Penalty(1e3),
    # v_pen=Penalty(
    #     0.5,
    #     act_func=crocoddyl.ActivationModelWeightedQuad,
    #     act_func_params=v_pen_weights,
    # ),
)


def get_penalty_mpc(
    angles_radian: np.ndarray,
    beta: float,
):
    return PenaltyTrajectory(
        u_pen=Penalty(1e-2),  # for the circle
        x_pen=Penalty(8e4),  # for the circle
        v_pen=Penalty(
            # 3e-2,
            1e-1,
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=v_pen_weights,
        ),
        q_pen=Penalty(
            # 9e-1,
            1,
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=q_pen_weights,
        ),
        rot_pen=Penalty(
            1e5,  # for the circle
            # 4e5,
            act_func=crocoddyl.ActivationModelWeightedQuadraticBarrier,
            act_func_params=[
                (
                    crocoddyl.ActivationBounds(
                        np.array([-2 * angle, -2 * angle, 4 * np.pi]),
                        np.array([2 * angle, 2 * angle, 4 * np.pi]),
                        beta,
                    ),
                    np.array([1.0, 1.0, 0.0]),
                )
                for angle in angles_radian
            ],
        ),
        prefix="mpc_",
    )


# data_saver = DataSaver("./data/setpoint_meta.csv", "./data/setpoint/")
mpc_horizon = 20
factor_int_time = 4
beta = 0.5
mpc_res, ddp_res, mpc_time = mpc_state_space(
    model,
    experiment,
    penalty_ddp,
    get_penalty_mpc,
    max_iter_ddp=1000,
    max_iter_mpc=100,
    mpc_horizon=mpc_horizon,
    factor_integration_time=factor_int_time,
    # data_saver=data_saver,
    beta=beta,
    t_crop=None,
    t_total=n,
    continue_end="repeat",  # "repeat" if mode == "circle" else "stationary",
)
print("worked")


q_list = np.array([x[:6] for x in mpc_res.states[1:]])
q_list_pin = np.array([x[:6] for x in ddp_res.state[1:]])
# q_dot_list = np.array([x[6:] for x in ddp_res.state[1:]])
# q_dot_list = np.array([x[6:] for x in res.x[1:]])  # shape (600, 6)
# controls = np.array([u for u in res.u])

x_task_pin = task_space_traj(model.pin_model, model.pin_data, q_list_pin, 24)
angles_mujoco = angles_traj(
    model.pin_model, model.pin_data, q_list, 24, 22, radian=False
)
angles_pin = angles_traj(
    model.pin_model, model.pin_data, q_list_pin, 24, 22, radian=False
)
# print(np.shape(x_task))
# plot_trajectory(traj_x, {"crocc": x_task})
plt.figure()
plt.plot(angles_mujoco, label="mujoco")
plt.plot(angles_pin, label="preplanned")
plt.title("Pendulum's angle with z-axis [degree]")
plt.legend(loc="upper right")
#
print("MPC: {} = {}".format(mode, mpc_time))
plot_trajectory(
    experiment.traj_x,
    {"mpc": mpc_res.x[1:, :], "preplanned": x_task_pin},
    header="Pendulum's tip coordinates ($x_1$, $x_2$, $x_3$ in the world frame):\n",
)
# plot_trajectory(q_list_pin, {"mujoco (q_t)_t": q_list})
# plot_single_traj(mpc_res.u, label="torques u")
plt.show()
# plot_single_traj(q_list, label="q(t)")
# plt.show()
# loop_mj_vis(model, q=q_list, q_dot=q_dot_list, num_motors=4)
loop_mj_vis(model, mpc_res.u, num_motors=4)
# loop_mj_vis(model, q=res.states[:, :6], q_dot=res.states[:, 6:], num_motors=4)

# save the results as a video
def make_camera():
    cam = mujoco.MjvCamera()
    cam.lookat[0] = 0.2
    cam.lookat[1] = 0.2
    cam.lookat[2] = 1.2
    cam.distance = 3  # 2.5
    cam.elevation = -20
    cam.azimuth = -120
    return cam


# results:
# circle:
#   time =  6.888531923294067s
#   error (3000 nodes)
#   avg abs error MPC =  0.006326638156965481
#   avg abs error preplanning 0.0037724290887143877
# spiral
#   time =  6.877151966094971
#   avg abs error MPC = 0.006326638156965481
#   avg abs error preplanning = 0.0037724290887143877

# smoothed
# CIRCLE
#   MPC: circle = 2.4587976932525635
#   plot_traj.py: err_avg = 0.0035966315396707077
#   plot_traj.py: err_avg = 0.0038431777233895676
# SPIRAL

frame_path = "../report/imgs/last_frame/"
# visualize(
#     model,
#     mode + "{}".format("_smoothed" if smoothing else ""),
#     controls=mpc_res.u[:3000, :],
#     cam=make_camera(),
#     frame_path=frame_path,
# )
