from utils.experiments.trajectory import (
    PenaltyTrajectory,
    mpc_state_space,
)
from utils.experiments.setpoint import ExperimentSetpoint
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


model = ConfigWAM("rotated")
n = 3000
experiment = ExperimentSetpoint(
    24,
    model.pend_end_world(),
    {"direction": "x", "orientation": -1, "radius": 0.15},
    24,
    n,
    rest_pos=model.q_config,
)


q_pen_weights = np.concatenate((np.ones(6), np.zeros(6)))
v_pen_weights = np.concatenate((np.zeros(6), np.ones(6)))

penalty_ddp = PenaltyTrajectory(
    u_pen=Penalty(1e-2),
    x_pen=Penalty(1e3),
    v_pen=Penalty(
        0.5,
        act_func=crocoddyl.ActivationModelWeightedQuad,
        act_func_params=v_pen_weights,
    ),
)


def get_penalty_mpc(
    angles_radian: np.ndarray,
    beta: float,
):
    return PenaltyTrajectory(
        u_pen=Penalty(1e-2),
        x_pen=Penalty(7e4),
        v_pen=Penalty(
            3e-2,
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=v_pen_weights,
        ),
        q_pen=Penalty(
            5e-3,
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=q_pen_weights,
        ),
        rot_pen=Penalty(
            1e5,
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
beta = 0.7
mpc_res, ddp_res, time_total = mpc_state_space(
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
    t_crop=1500,
    t_total=n,
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
# plt.figure()
# plt.plot(angles_mujoco, label="mujoco")
# plt.plot(angles_pin, label="preplanned")
# plt.title("Pendulum's angle with z-axis [degree]")
# plt.legend(loc="upper right")

# plot_trajectory(
#     experiment.traj_x,
#     {"mpc": mpc_res.x[1:, :], "preplanned": x_task_pin},
#     header="Pendulum's tip coordinates ($x_1$, $x_2$, $x_3$ in the world frame):\n",
# )
# plot_trajectory(q_list_pin, {"mujoco (q_t)_t": q_list})
# plot_single_traj(mpc_res.u, label="torques u")
# plt.show()
# plot_single_traj(q_list, label="q(t)")
# plt.show()
# loop_mj_vis(model, q=q_list, q_dot=q_dot_list, num_motors=4)
loop_mj_vis(model, mpc_res.u, num_motors=4)
# visualize(model, "name_", "../videos/", mpc_res.u, num_motors=4)
# loop_mj_vis(model, q=res.states[:, :6], q_dot=res.states[:, 6:], num_motors=4)
