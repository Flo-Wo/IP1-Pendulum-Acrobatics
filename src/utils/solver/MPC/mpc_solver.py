import crocoddyl
import numpy as np
import mujoco
from utils.decorators import log_time, remove_mj_logs
from utils.solver.MPC.mpc_results import MPCResults
from utils.solver.MPC.mpc_warmstart import MPCWarmStart
import logging


# @remove_mj_logs
@log_time
def mpc(
    model,
    mujoco_data,
    int_cost_models: list[crocoddyl.IntegratedActionModelEuler],
    terminal_model: crocoddyl.IntegratedActionModelEuler,
    mpc_horizon: int = 6,
    solver_max_iter: int = 100,
    solver: crocoddyl.SolverAbstract = crocoddyl.SolverBoxFDDP,
    continue_end: str = "repeat",
    start_with_static_warm: bool = True,
    custom_warmstart: MPCWarmStart = None,
    num_motors: int = 4,
    show_logs: bool = False,
    factor_integration_time: int = 1,
) -> MPCResults:
    """Model Predictive Control Solver, based on crocoddyl's DDP solver.

    Parameters
    ----------
    model : mujoco model
        Mujoco model used for internal simulations to compute the next state
        after applying the MPC computed torque.
    mujoco_data : mujoco data
        Mujoco data used for internal simulations.
    int_cost_models : list[crocoddyl.IntegratedActionModelEuler]
        List with already integrated cost models, used in a circular append
        fashion by the DDP solver.
    terminal_node : crocoddyl.IntegratedActionModelEuler
        One single terminal cost model, which is used for
        EVERY ITERATION in the ddp solver (we only append new problems but
        the terminal node stays the same). Therefore the terminal model
        should be used with zero costs.
    time_horizon : int, optional
        Time horizon used by the MPC to plan ahead, by default 6.
    max_iter : int, optional
        Maximum number of iterations for the DDP solver, by default 100.
    cont : str, optional
        Default continuation type if the time_horizon exceeds the length of
        the desired trajectory, by default "repeat".
    warm_start_x : np.ndarray, optional
        Optional list of pairs (x_d, \dot{x}_d) used as a warmstart
        for the DDP solver, by default None.
    warm_start_u : np.ndarray, optional
        Optional list of controls (u) used as a warmstart
        for the DDP solver, by default None.
    static_warmstart : bool, optional
        Initialize the solver with a list of x0 and quasiStatic commands,
        by default False.
    num_motors : float, optional
        Number of motor commands, used e.g. for the wam with attached pendulum,
        where pinocchio internally uses 6 torque commands, but in reality the
        wam only has 4 actuators.
    show_logs : bool, optional
        Optional debugging parameter to show the logs,
        by default False.
    factor_integration_time : int, optional
        Factor to multiply the integration time with, default is 1. I.e. in
        the mujoco simulation we will execute each torque/command
        (factor_integration_time - 1) times additionally in order to match the
        integration time of crocoddyl.

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        MPC-internal simulation results: x, \dot{x}, u.

    Raises
    ------
    NotImplementedError
        Caused by missing/wrong continuation methods.
    """
    if custom_warmstart is None:
        custom_warmstart = MPCWarmStart([], [], False)
    # start the simulation
    mujoco.mj_step(model, mujoco_data)

    # use the initial state to start planning --> will be updated in the following
    state = np.concatenate((mujoco_data.qpos, mujoco_data.qvel))

    results_data = MPCResults(
        state[:],
        np.zeros(num_motors + 2),
        state[:],
        mujoco_data.sensordata[:],
        0,
        factor_integration_time=factor_integration_time,
        mpc_horizon=mpc_horizon,
    )

    # duplicate each model --> to maybe improve the performance
    active_models = int_cost_models[:mpc_horizon]

    # how should the mpc continue to plan ahead when reaching the end nodes
    # reapeat the current trajectory --> add simple copies to the already enqueued cost models
    if continue_end == "repeat":
        int_cost_models.extend(int_cost_models[:mpc_horizon])
    elif continue_end == "stationary":
        int_cost_models.extend([int_cost_models[-1]] * mpc_horizon)
    else:
        raise NotImplementedError

    problem = crocoddyl.ShootingProblem(state, active_models, terminal_model)

    first_call = True

    num_total = len(int_cost_models)
    # create solver for this OC problem --> only once to save memory
    ddp = solver(problem)

    for idx, next_active_model in enumerate(int_cost_models[mpc_horizon:]):

        # the problem stays the same, but the initial state changes
        problem.x0 = state

        # ==== LOGGING FOR DEBUGGING ====
        if idx % 100 == 0 and show_logs:
            print("\nINDEX: {}/{}".format(idx, num_total))
            # log = crocoddyl.CallbackLogger()
            # ddp.setCallbacks([log, crocoddyl.CallbackVerbose()])

        # check for the first time the MPC is called --> static warmstart and
        # we have to define the warmstart for the next calls
        if first_call:
            if start_with_static_warm and not custom_warmstart:
                logging.info("MPC: used a static warmstart for the first step.")
                x0 = np.concatenate(
                    (mujoco_data.qpos, np.zeros(np.shape(mujoco_data.qvel)))
                )
                custom_warmstart.set_warm(
                    [x0] * (problem.T + 1), problem.quasiStatic([x0] * problem.T), True
                )
            # only perform this step once
            first_call = False

        # solve the problem
        warm_x, warm_u, is_feasible = custom_warmstart.get_warm()
        solved = ddp.solve(warm_x, warm_u, solver_max_iter, is_feasible)
        # solved = ddp.solve()

        # set the parameters for the next warmstart
        custom_warmstart.set_warm(ddp.xs, ddp.us, solved)

        # ===== MUJOCO SIMULATION: CONTROL DELAY IMPLEMENTATION =====
        # apply the first torque to the robot --> plan again in an MPC manner
        torque = ddp.us[0]

        mujoco_data.ctrl[:num_motors] = torque[:num_motors]  # apply torque
        mujoco.mj_step(model, mujoco_data)
        # update the current state for the next solution
        mj_x = mujoco_data.qpos
        mj_dx = mujoco_data.qvel
        state = np.concatenate((mj_x, mj_dx))

        results_data.add_data(ddp.xs[0], torque, state, mujoco_data.sensordata, solved)
        # if the factor is 2, we want to execute 1 extra step in the mujoco simulation
        # if the factor is 1, we do not want to execute a duplicate torque
        for _ in range(factor_integration_time - 1):
            # is equivalent to just call the mj_step function
            # mujoco_data.ctrl[:num_motors] = torque[:num_motors]
            mujoco.mj_step(model, mujoco_data)
            # update the current state for the next solution
            mj_x = mujoco_data.qpos
            mj_dx = mujoco_data.qvel
            state = np.concatenate((mj_x, mj_dx))
            results_data.add_data(
                ddp.xs[0], torque, state, mujoco_data.sensordata, solved
            )
        # append the next problem node to our Problem-Instance
        problem.circularAppend(next_active_model)
    return results_data
