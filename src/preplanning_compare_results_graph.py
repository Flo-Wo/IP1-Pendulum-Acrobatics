from utils.experiments.trajectory import (
    PenaltyTrajectory,
    mpc_task_space,
    mpc_state_space,
)
from utils.experiments.setpoint import ExperimentSetpoint
from utils.config import ConfigWAM
from utils.costs import Penalty
from utils.pin_helper import task_space_traj, angles_traj
from utils.visualize import visualize

import numpy as np
import matplotlib.pyplot as plt
import crocoddyl
import tikzplotlib
import mujoco


def mpc_raw(
    model,
    experiment,
    mpc_horizon: int,
    factor_int_time: int,
):
    q_pen_weights = np.concatenate((np.ones(6), np.zeros(6)))
    v_pen_weights = np.concatenate((np.zeros(6), np.ones(6)))
    penalty = PenaltyTrajectory(
        u_pen=Penalty(1e-1),
        x_pen=Penalty(3e4),
        rot_pen=Penalty(
            3e5,
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=np.array([1.0, 1.0, 0.0]),
        ),
        q_pen=Penalty(
            5,
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=q_pen_weights,
        ),
        v_pen=Penalty(
            3e-2,
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=v_pen_weights,
        ),
    )

    # task space only
    res, time_needed = mpc_task_space(
        model,
        experiment,
        penalty,
        solver="mpc",
        solver_max_iter=100,
        mpc_horizon=mpc_horizon,
        mpc_factor_integration_time=factor_int_time,
    )
    return res, time_needed


def mpc_with_preplanning(
    model,
    experiment,
    mpc_horizon: int,
    factor_int_time: int,
    beta: float,
    time_total: int = 3000,
):
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

    mpc_res, ddp_res, time_total = mpc_state_space(
        model,
        experiment,
        penalty_ddp,
        get_penalty_mpc,
        max_iter_ddp=1000,
        max_iter_mpc=100,
        mpc_horizon=mpc_horizon,
        factor_integration_time=factor_int_time,
        beta=beta,
        t_crop=1500,
        t_total=time_total,
    )
    return mpc_res, ddp_res, time_total


def combined_plot(
    angles: dict,
    task_space: dict,
    labels: dict,
):
    linestyles = {
        "target": "-",
        "raw": "-.",
        "planned": "--",
        "ddp": ":",
    }
    colors = {
        # "target": "firebrick",
        "target": "gray",
        "raw": "royalblue",
        "planned": "orange",
        # "ddp": "forestgreen",
        "ddp": "firebrick",
    }

    plt.rc("text", usetex=True)
    plt.rc("font", family="serif")

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 3.7))

    x_label = "$(x^\mathrm{{tip}}_t)_{}$"
    for i, ax in enumerate(fig.axes):
        if i == 0:
            for model in angles.keys():
                ax.plot(
                    angles[model],
                    color=colors[model],
                    linestyle=linestyles[model],
                    label=labels[model],
                )
            ax.set_ylabel("angle [$^{\circ}$]")
            ax.legend(loc="lower right", ncol=4)
        else:
            for model in task_space.keys():
                ax.plot(
                    task_space[model][:, i - 1],
                    color=colors[model],
                    linestyle=linestyles[model],
                    label=labels[model],
                )
                ax.set_ylabel(x_label.format(i))
            if i < 1:
                ax.legend(loc="lower right", ncol=4)
    fig.align_labels()
    plt.xlabel("time step $t$")
    plt.savefig("../report/imgs/compare_raw_vs_preplanning.pdf")
    tikzplotlib.save("../report/imgs/compare_raw_vs_preplanning.tex")
    plt.show()


if __name__ == "__main__":
    vis = False

    # define the camera once
    def make_camera():
        cam = mujoco.MjvCamera()
        cam.lookat[0] = 0.2
        cam.lookat[1] = 0.2
        cam.lookat[2] = 1.2
        cam.distance = 3  # 2.5
        cam.elevation = -20
        cam.azimuth = -120
        return cam

    # define constants
    model = ConfigWAM("rotated")
    n = 3000
    experiment = ExperimentSetpoint(
        24,
        model.pend_end_world(),
        {"direction": "x", "orientation": -1, "radius": 0.1},
        24,
        n,
        rest_pos=model.q_config,
    )
    mpc_horizon = 20
    factor_int_time = 4
    beta = 0.5

    print("MPC RAW")
    mpc_raw_res, mpc_raw_time = mpc_raw(
        model,
        experiment,
        mpc_horizon=mpc_horizon,
        factor_int_time=factor_int_time,
    )
    print("MPC WITH PREPLANNING")
    mpc_res, ddp_res, _ = mpc_with_preplanning(
        model,
        experiment,
        mpc_horizon=mpc_horizon,
        factor_int_time=factor_int_time,
        beta=beta,
    )
    # analyse the raw mpc
    raw_q_list = np.array([x[:6] for x in mpc_raw_res.states[1:]])
    raw_angles_mujoco = angles_traj(
        model.pin_model,
        model.pin_data,
        raw_q_list,
        24,
        22,
        radian=False,
        project_2D=True,
    )
    if vis:
        frame_path = "../report/imgs/last_frame/"
        visualize(
            model,
            "mpc_raw",
            controls=mpc_raw_res.u,
            cam=make_camera(),
            frame_path=frame_path,
        )
        visualize(
            model,
            "mpc_with_preplanning",
            controls=mpc_res.u,
            cam=make_camera(),
            frame_path=frame_path,
        )
        visualize(
            model,
            "ddp_preplanned",
            q=np.array([x[:6] for x in ddp_res.state[1:]]),
            q_dot=np.array([x[6:] for x in ddp_res.state[1:]]),
            cam=make_camera(),
            frame_path=frame_path,
        )

    # analyse preplanning
    preplan_q_list = np.array([x[:6] for x in mpc_res.states[1:]])
    preplan_q_list_pin = np.array([x[:6] for x in ddp_res.state[1:]])

    preplan_x_task_ddp = task_space_traj(
        model.pin_model, model.pin_data, preplan_q_list_pin, 24
    )

    preplan_angles_mujoco = angles_traj(
        model.pin_model,
        model.pin_data,
        preplan_q_list,
        24,
        22,
        radian=False,
        project_2D=True,
    )
    preplan_angles_ddp = angles_traj(
        model.pin_model,
        model.pin_data,
        preplan_q_list_pin,
        24,
        22,
        radian=False,
        project_2D=True,
    )

    angles = {
        "target": np.zeros(np.shape(raw_angles_mujoco)),
        "raw": raw_angles_mujoco,
        "planned": preplan_angles_mujoco,
        "ddp": preplan_angles_ddp,
    }
    task_space = {
        "target": experiment.traj_x,
        "raw": mpc_raw_res.x,
        "planned": mpc_res.x,
        "ddp": preplan_x_task_ddp,
    }
    labels = {
        "target": "target",
        "raw": "MPC",
        "planned": "PMPC",
        "ddp": "FDDP",
    }
    combined_plot(
        angles,
        task_space,
        labels,
    )
