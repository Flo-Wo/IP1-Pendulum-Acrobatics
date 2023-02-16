from utils.solver.solver_results_base import SolverResultsBase
import numpy as np


class MPCResults(SolverResultsBase):
    """
    Class to cache the results of an mpc optimization problem:
    - xs: crocoddyl list with (q,q_dot) pairs, saved only the first state
    of each iteration
    - us: crocoddyl list with (u) torques, only the first torque which is
    applied is saved
    - mj_xs: list with (q,q_dot) pairs of the mujoco simulation after
    applying the torque us
    - mj_sensordata: list with sensordata for each iteration
    """

    def __init__(
        self,
        ddp_x: np.ndarray,
        u: np.ndarray,
        state: np.ndarray,
        mj_sensor: np.ndarray,
        solved: bool,
        factor_integration_time: int,
        mpc_horizon: int,
    ):
        self.ddp_x = ddp_x[None, :]
        self.u = u[None, :]
        self.states = state[None, :]
        self.x, self.rot, self.x_pend_beg = self._extract_mj_data(mj_sensor)
        self.solved = [solved]

        self.factor_integration_time = factor_integration_time
        self.mpc_horizon = mpc_horizon

    def add_data(self, ddp_x, u, state, mj_sensor, solved):
        self.ddp_x = np.vstack((self.ddp_x, ddp_x[None, :]))
        self.u = np.vstack((self.u, u[None, :]))

        self.states = np.vstack((self.states, state[None, :]))

        # extract the data
        x, rot, x_pend_beg = self._extract_mj_data(mj_sensor)

        self.x = np.vstack((self.x, x))
        self.rot = np.vstack((self.rot, rot))
        self.x_pend_beg = np.vstack((self.x_pend_beg, x_pend_beg))

        self.solved.append(solved)

    def _extract_mj_data(self, mj_sensor: np.ndarray):
        """Return: x, rot, x_pend_beg"""
        mj_copy = mj_sensor.copy()
        return mj_copy[0:3][None, :], mj_copy[3:7][None, :], mj_copy[7:10][None, :]

    def __str__(self) -> str:
        return "MPC_factor_int_time_{}_horizon_{}".format(
            self.factor_integration_time, self.mpc_horizon
        )

    def save_to_file(self) -> dict:
        return {
            "x": self.x,
            "u": self.u,
            "x_rot": self.rot,
            "x_pend_beg": self.x_pend_beg,
        }

    def save_to_metadata(self) -> dict:
        return {
            "solver": "MPC",
            "factor_integration_time": self.factor_integration_time,
            "mpc_horizon": self.mpc_horizon,
        }
