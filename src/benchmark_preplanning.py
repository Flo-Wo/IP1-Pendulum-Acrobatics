from utils.experiments.trajectory import (
    ExperimentTrajectory,
    PenaltyTrajectory,
    mpc_state_space,
)
from utils.experiments.setpoint import ExperimentSetpoint
from utils.config import ConfigWAM
import numpy as np
from utils.data_handling import DataSaver
from utils.costs import Penalty
import crocoddyl

if __name__ == "__main__":
    model = ConfigWAM("rotated")
    n = 3000

    # appropriate weightening for v and q
    q_pen_weights = np.concatenate((np.ones(6), np.zeros(6)))
    v_pen_weights = np.concatenate((np.zeros(6), np.ones(6)))

    # penalty terms for the DDP case
    penalty_ddp = PenaltyTrajectory(
        u_pen=Penalty(1e-2),
        x_pen=Penalty(1e3),
        v_pen=Penalty(
            0.5,
            act_func=crocoddyl.ActivationModelWeightedQuad,
            act_func_params=v_pen_weights,
        ),
        prefix="ddp_",
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

    data_saver = DataSaver(
        "../data/setpoint/mpc_preplanning.csv",
        "../data/setpoint/mpc_preplanning/",
    )

    factor_int_time = 4
    experiment_counter = 0

    for dir in ["x", "y", "z"]:
        for orientation in [-1, 1]:
            for radius in [0.05, 0.1, 0.15]:
                for mpc_horizon in [5, 10, 20]:
                    for beta in [0.5, 0.6, 0.7]:
                        print("Experiment: {}".format(experiment_counter))
                        print(
                            "\n{}: {}, {}, {}, {}, {}\n".format(
                                dir,
                                orientation,
                                radius,
                                mpc_horizon,
                                factor_int_time,
                                beta,
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
                            n,
                            rest_pos=model.q_config,
                        )

                        _ = mpc_state_space(
                            model,
                            experiment,
                            penalty_ddp,
                            get_penalty_mpc,
                            max_iter_ddp=1000,
                            max_iter_mpc=100,
                            mpc_horizon=mpc_horizon,
                            factor_integration_time=factor_int_time,
                            data_saver=data_saver,
                            beta=beta,
                            t_crop=1500,
                            t_total=n,
                            show_logs=False,
                        )
                        experiment_counter += 1

    print("\n\nNumber of Experiments: {}\n\n".format(experiment_counter))
