from utils.experiments.trajectory import (
    PenaltyTrajectory,
    mpc_task_space,
)
from utils.trajectories.derivatives import derivatives_joint_task
from utils.experiments.setpoint import ExperimentSetpoint
import numpy as np
import matplotlib.pyplot as plt
from utils.pin_helper import (
    task_space_traj,
    angles_traj,
)
from utils.costs import Penalty
from utils.config import ConfigWAM

import tikzplotlib
import crocoddyl

model = ConfigWAM("rotated")

len_traj = 800

experiment = ExperimentSetpoint(
    24,
    model.pend_end_world(),
    {"direction": "x", "orientation": -1, "radius": 0.1},
    24,
    len_traj,
)
# PLOTTING
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

# fig, [ax1, ax2, ax3, ax4, ax5] = plt.subplots(5, 1, sharex=True, figsize=(10, 8))
fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
# ax4_acc = ax4.twinx()
# ax5_acc = ax5.twinx()

# print the target
x_target = experiment.traj_x
ax1.plot(np.zeros(np.shape(x_target[:, 0])), color="gray", label="target")
ax2.plot(x_target[:, 0], color="gray", label="target")
ax3.plot(x_target[:, 0], color="gray", label="target")

linestyles = ["--", "-."]
colors = ["royalblue", "orange"]
add_colors = ["forestgreen", "firebrick"]
add_linestyles = ["-", ":"]


v_pen_weights = np.concatenate((np.zeros(6), np.ones(6)))

for idx, v_pen_float in enumerate([0.0, 0.5]):  # , 0.5]:

    u_pen_float = 1e-2

    penalty = PenaltyTrajectory(
        u_pen=Penalty(
            u_pen_float,
        ),
        x_pen=Penalty(1e3),  # 1e6 for extremely aggressive moves
        v_pen=Penalty(
            v_pen_float,
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=v_pen_weights,
        ),
    )

    res, time_needed = mpc_task_space(
        model,
        experiment,
        penalty,
        solver="ddp",
        solver_max_iter=100,
        show_logs=True,
    )

    # COMPUTATION
    q_list = np.array([x[:6] for x in res.state[1:]])  # shape (600, 6)
    q_dot_list = np.array([x[6:] for x in res.state[1:]])  # shape (600, 6)
    controls = np.array([u for u in res.u])

    x_task = task_space_traj(model.pin_model, model.pin_data, q_list, 24)
    x_task_endeff = task_space_traj(model.pin_model, model.pin_data, q_list, 22)
    angles = angles_traj(
        model.pin_model, model.pin_data, q_list, 24, 22, project_2D=True
    )

    joint_id = 24
    derivs = derivatives_joint_task(
        model.pin_model,
        model.pin_data,
        q_list,
        q_dot_list,
        controls,
        joint_id,
        normalize=False,
        combined_max=False,
    )

    ax1.plot(
        angles,
        linestyle=linestyles[idx],
        color=colors[idx],
        label="$\\theta_t$: {}".format(v_pen_float),
    )

    ax2.plot(
        x_task[:, 0],
        linestyle=linestyles[idx],
        color=colors[idx],
        label="$(x_t^\mathrm{{tip}})_1$: {}".format(v_pen_float),
    )

    ax3.plot(
        x_task_endeff[:, 0],
        linestyle=linestyles[idx],
        color=colors[idx],
        label="$(x_t^\mathrm{{endeff}})_1$: {}".format(v_pen_float),
    )

    # ax4.plot(x_task[:, 2])  # , label="pend: {}".format(v_pen_float))
    # ax4.plot(x_task_endeff[:, 2])  # , label="endeff: {}".format(v_pen_float))
    # ax4.set_ylabel("$(x_t^\mathrm{<name>})_3$")
    # ax4.legend()

    ax4.plot(
        derivs.v_joint,
        color=colors[idx],
        linestyle=linestyles[idx],
        label="$||v^\mathrm{{JS}}||_2$ : {}".format(v_pen_float),
    )
    # ax4_acc.plot(
    #     derivs.a_joint,
    #     color=add_colors[idx],
    #     linestyle=add_linestyles[idx],
    #     label="$||a^\mathrm{{JS}}||_2$: {}".format(v_pen_float),
    # )

    # ax5.plot(
    #     derivs.v_task,
    #     color=colors[idx],
    #     linestyle=linestyles[idx],
    #     label="$||v^\mathrm{{TS}}||_2$: {}".format(v_pen_float),
    # )
    # ax5_acc.plot(
    #     derivs.a_task,
    #     color=add_colors[idx],
    #     linestyle=add_linestyles[idx],
    #     label="$||a^\mathrm{{TS}}||_2$: {}".format(v_pen_float),
    # )
ax2.set_ylabel("$x_\mathrm{world}$")
ax2.legend(loc="upper right", ncol=3)
ax3.set_ylabel("$x_\mathrm{world}$")
ax3.legend(loc="upper right", ncol=3)
ax4.set_ylabel("$||v||_2$")
# ax4_acc.set_ylabel("$||a||_2$")
# ax5.set_ylabel(r"$||v||_2$")
# ax5_acc.set_ylabel(r"$||a||_2$")
# shared x-axis
# ax5.set_xlabel("time step $t$")
ax4.set_xlabel("time step $t$")

ax1.set_ylabel("angle [$^{\circ}$]")
ax1.legend(loc="upper right", ncol=3)

# lines_labels = [ax.get_legend_handles_labels() for ax in [ax4, ax4_acc]]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# ax4.legend(lines, labels, loc="upper right", ncol=4)
ax4.legend(loc="upper right", ncol=4)

# lines_labels = [ax.get_legend_handles_labels() for ax in [ax5, ax5_acc]]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# ax5.legend(lines, labels, loc="upper right", ncol=4)

fig.align_labels()

tikzplotlib.save("../report/imgs/velocity_penalty.tex")
plt.savefig("../report/imgs/velocity_penalty.pdf")
plt.show()
