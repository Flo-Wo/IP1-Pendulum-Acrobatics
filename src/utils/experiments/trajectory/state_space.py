from utils.experiments.trajectory import (
    ExperimentTrajectory,
    PenaltyTrajectory,
    integrated_cost_models,
)
from utils.config import ConfigModelBase
from utils.costs import ControlBounds
from copy import copy
import numpy as np
import crocoddyl
from utils.solver import ddp, mpc
from utils.data_handling import DataSaver, AddData
from utils.pin_helper import (
    task_space_traj,
    angles_traj,
)
from utils.costs import Penalty


def mpc_state_space(
    model: ConfigModelBase,
    experiment_task_space: ExperimentTrajectory,
    penalty_ddp: PenaltyTrajectory,
    get_penalty_mpc: callable,
    max_iter_ddp: int = 1000,
    max_iter_mpc: int = 100,
    mpc_horizon: int = 10,
    factor_integration_time: int = 1,
    data_saver: DataSaver = None,
    beta: float = 0.5,
    t_crop: int = 1500,
    t_total: int = 3000,
    show_logs: bool = False,
    continue_end: str = "stationary",
):
    # we need copy here, otherwise the results are incorrect due to side effects
    mj_model = copy(model.mj_model)
    mj_data = copy(model.mj_data)
    pin_model = copy(model.pin_model)
    pin_data = copy(model.pin_data)

    state = crocoddyl.StateMultibody(pin_model)
    actuation_model = crocoddyl.ActuationModelFull(state)
    action_model = crocoddyl.DifferentialActionModelFreeFwdDynamics

    dt_const = 0.002
    ctrl_bounds = ControlBounds()

    # PART 1: BUILD COST MODELS FOR THE TASK SPACE OC
    # build intermediate/running cost models
    cost_models_ddp = integrated_cost_models(
        state,
        action_model,
        actuation_model,
        penalty_ddp,
        experiment_task_space,
        dt_const,
        ctrl_bounds,
    )
    # build terminal cost models
    terminal_costs_ddp = crocoddyl.CostModelSum(state)
    terminal_node_ddp = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation_model, terminal_costs_ddp
        ),
        0,
    )
    # terminal_node_ddp = cost_models_ddp[-1]

    # PART 2: SOLVE THE TASK SPACE OC VIA CROCODDYL'S FDDP SOLVER
    ddp_res, ddp_time = ddp(
        cost_models_ddp,
        terminal_node_ddp,
        model.q_config,
        qdot=np.zeros(state.nv),
        solver_max_iter=max_iter_ddp,
        show_logs=show_logs,
    )
    q_list = np.array([x[:6] for x in ddp_res.state[1:]])

    # NOTE: CLIPPING
    x_task_raw = task_space_traj(pin_model, pin_data, q_list, 24)
    if t_crop is not None:
        x_task_cropped = np.concatenate(
            (
                x_task_raw[:t_crop, :],
                np.tile(x_task_raw[t_crop, :], (t_total - t_crop, 1)),
            )
        )
    else:
        x_task_cropped = x_task_raw

    angles_radian = angles_traj(
        pin_model,
        pin_data,
        q_list,
        24,
        22,
        radian=True,
    )
    if t_crop is not None:
        angles_radian_cropped = np.concatenate(
            (angles_radian[:t_crop], np.tile(angles_radian[t_crop], t_total - t_crop))
        )
    else:
        angles_radian_cropped = angles_radian
    # PART 3: BUILD COST MODELS FOR THE STATE SPACE OC
    experiment_mpc = ExperimentTrajectory(
        24,
        traj_x=x_task_cropped,
        traj_rot=t_total * [np.eye(3)],
        rest_pos=model.q_config,
    )

    penalty_mpc = get_penalty_mpc(angles_radian_cropped, beta=beta)

    # TODO: currently the ddp works with the finest time and the mpc uses a more
    # coarse resolution
    dt_const_mpc = factor_integration_time * dt_const
    cost_models_mpc = integrated_cost_models(
        state,
        action_model,
        actuation_model,
        penalty_mpc,
        experiment_mpc,
        dt_const_mpc,
        ctrl_bounds,
    )

    terminal_costs_mpc = crocoddyl.CostModelSum(state)
    terminal_node_mpc = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation_model, terminal_costs_mpc
        ),
        0,
    )
    # terminal_node_mpc = cost_models_mpc[-1]

    # PART 4: SOLVE THE PROBLEM WITH OUR ONLINE MPC
    mpc_res, mpc_time = mpc(
        mj_model,
        mj_data,
        cost_models_mpc[::factor_integration_time],
        terminal_node_mpc,
        mpc_horizon,
        max_iter_mpc,
        show_logs=show_logs,
        factor_integration_time=factor_integration_time,
        continue_end=continue_end,
    )
    # PART 5: SAVE THE DATA (INCLUDING THE INTERNAL SOLUTION OF THE CROCODDYL OC SOLVER)
    add_data = AddData(
        str_repr="beta_{}".format(float(2 * beta)),
        to_metadata={
            "ddp_comp_time": ddp_time,
            "mpc_comp_time": mpc_time,
            "beta": float(beta * 2),
            "crop_idx": t_crop,
        },
        to_files={"ddp_states": ddp_res.state},
        exclude=["comp_time_ddp", "comp_time_mpc"],
    )

    if data_saver is not None:
        print("Run trajectory: saving")
        data_saver.save_data(
            model, penalty_ddp, penalty_mpc, experiment_task_space, mpc_res, add_data
        )
    return mpc_res, ddp_res, mpc_time
