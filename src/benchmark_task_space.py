from utils.experiments.trajectory import (
    PenaltyTrajectory,
    mpc_task_space,
)
from utils.experiments.setpoint import ExperimentSetpoint
from utils.config import ConfigWAM
from utils.data_handling import DataSaver
from utils.costs import Penalty

import numpy as np
import crocoddyl

if __name__ == "__main__":

    normal_sing = ConfigWAM("standard")
    normal_ang = ConfigWAM("standard_angled")
    rotated = ConfigWAM("rotated")
    human = ConfigWAM("human")

    models = [normal_sing, normal_ang, rotated, human]

    penalties = {
        normal_sing: PenaltyTrajectory(
            u_pen=Penalty(1e-1),
            x_pen=Penalty(5e4),
            rot_pen=Penalty(
                1e5,
                act_func=crocoddyl.ActivationModelWeightedQuad,
                act_func_params=np.array([1.0, 1.0, 0.0]),
            ),
            q_pen=Penalty(1e-2),
        ),
        normal_ang: PenaltyTrajectory(
            u_pen=Penalty(1e-1),
            x_pen=Penalty(1e5),
            rot_pen=Penalty(
                6e5,
                act_func=crocoddyl.ActivationModelWeightedQuad,
                act_func_params=np.array([1.0, 1.0, 0.0]),
            ),
            q_pen=Penalty(1e-2),
        ),
        rotated: PenaltyTrajectory(
            u_pen=Penalty(1e-1),
            x_pen=Penalty(3e4),
            rot_pen=Penalty(
                3e5,
                act_func=crocoddyl.ActivationModelWeightedQuad,
                act_func_params=np.array([1.0, 1.0, 0.0]),
            ),
            q_pen=Penalty(1e-2),
        ),
        human: PenaltyTrajectory(
            u_pen=Penalty(1e-1),
            x_pen=Penalty(5e4),
            rot_pen=Penalty(
                1e5,
                act_func=crocoddyl.ActivationModelWeightedQuad,
                act_func_params=np.array([1.0, 1.0, 0.0]),
            ),
            q_pen=Penalty(1e-2),
        ),
    }

    # define the data saver once
    data_saver = DataSaver("../data/setpoint/mpc_raw.csv", "../data/setpoint/mpc_raw/")
    experiment_counter = 0
    for model in models:
        penalty = penalties[model]
        for dir in ["x", "y", "z"]:
            for orientation in [-1, 1]:
                for radius in [0.05, 0.1, 0.15]:
                    for mpc_horizon in [5, 10, 20]:
                        for factor_int_time in [1, 2, 4]:
                            print(
                                "\n{}: {}, {}, {}, {}, {}\n".format(
                                    model,
                                    dir,
                                    orientation,
                                    radius,
                                    mpc_horizon,
                                    factor_int_time,
                                )
                            )
                            # define the individual penalties
                            experiment = ExperimentSetpoint(
                                24,
                                model.pend_end_world(),
                                {
                                    "direction": dir,
                                    "orientation": orientation,
                                    "radius": radius,
                                },
                                24,
                                3000,
                                rest_pos=model.q_config,
                            )

                            res, time_needed = mpc_task_space(
                                model,
                                experiment,
                                penalty,
                                solver="mpc",
                                solver_max_iter=100,
                                mpc_horizon=mpc_horizon,
                                mpc_factor_integration_time=factor_int_time,
                                data_saver=data_saver,
                            )
                            experiment_counter += 1
    print("\n\nNumber of Experiments: {}\n\n".format(experiment_counter))
    # Number of Experiments: 648
