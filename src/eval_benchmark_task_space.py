from utils.evaluation import eval_setpoint
import pandas as pd
from utils.evaluation import results_to_latex


def eval_integration_factor(
    metadata: pd.DataFrame,
    time_steps_ends_error: int = 300,
    time_horizon: int = 3000,
    factor_integration_time: int = 1,
    target_folder: str = "../report/setpoint_results/mpc_raw",
):

    # evaluate the results
    x_err, rot_err, comp_time, conf_overview = eval_setpoint(
        metadata,
        path_to_folder="../data/setpoint/mpc_raw/",
        end_error=time_steps_ends_error,
        radian=False,
        factor_integration_time=factor_integration_time,
        time_horizon=time_horizon,
    )
    # save to tex
    results_to_latex(
        x_err,
        rot_err,
        comp_time,
        conf_overview,
        target_folder=target_folder
        + "/factor_integration_time_{}".format(factor_integration_time),
        captions={
            "x": "Positional error: ",
            "rot": "Rotational error: ",
            "comp_time": "Computional time: ",
            "conf": "Solver configuration: ",
        },
        caption_meta={
            "factor_integration_time": factor_integration_time,
            "end_error": time_step_end_error,
        },
        filename_meta={
            "end_error": time_step_end_error,
            "factor_int": factor_integration_time,
        },
    )


if __name__ == "__main__":
    # define relevant constants
    goal_meta_data = pd.read_csv("../data/setpoint/mpc_raw.csv")

    time_step_end_error = 300
    time_horizon = 3000

    # TODO: check column format for all of the functions
    for factor_integration_time in [1, 2, 4]:
        eval_integration_factor(
            goal_meta_data,
            time_step_end_error,
            time_horizon,
            factor_integration_time,
        )
