from utils.evaluation import eval_setpoint, eval_ddp_precomputed
import pandas as pd
from utils.evaluation import results_to_latex


def eval_single_experiment(
    metadata: pd.DataFrame,
    time_step_end_error: int = 300,
    time_horizon: int = 3000,
    factor_integration_time: int = 4,
    beta: float = 1.0,
    target_folder: str = "../report/setpoint_results/mpc_preplanning/",
    prefix: int = "mpc_",
):

    # evaluate the results
    x_err, rot_err, comp_time, conf_overview = eval_setpoint(
        metadata,
        path_to_folder="../data/setpoint/mpc_preplanning/",
        end_error=time_step_end_error,
        radian=False,
        factor_integration_time=factor_integration_time,
        time_horizon=time_horizon,
        eval_models=["rotated"],
        prefix=prefix,
    )

    results_to_latex(
        x_err,
        rot_err,
        comp_time,
        conf_overview,
        target_folder=target_folder + "/beta_{}".format(int(10 * beta)),
        captions={
            "x": "Positional error: ",
            "rot": "Rotational error: ",
            "comp_time": "Computional time: ",
            "conf": "Solver configuration: ",
        },
        caption_meta={
            "factor_integration_time": factor_integration_time,
            "end_error": time_step_end_error,
            "\\beta": beta,
        },
        filename_meta={
            "end_error": time_step_end_error,
            "factor_int": factor_integration_time,
        },
    )


def eval_preplanning(
    metadata: pd.DataFrame,
    time_steps_end_error: int = 300,
    time_horizon: int = 3000,
    factor_integration_time: int = 4,
    target_folder: str = "../report/setpoint_results/mpc_preplanning/",
    prefix: int = "ddp_",
):
    x_err, rot_err, comp_time, conf_overview = eval_ddp_precomputed(
        metadata[(metadata["beta"] == 1.0)],
        path_to_folder="../data/setpoint/mpc_preplanning/",
        end_error=time_steps_end_error,
        radian=False,
        factor_integration_time=4,
        time_horizon=time_horizon,
        eval_models=["rotated"],
        prefix=prefix,
    )
    results_to_latex(
        x_err,
        rot_err,
        comp_time,
        conf_overview,
        target_folder=target_folder + "/ddp_preplanning",
        captions={
            "x": "Positional error (DDP): ",
            "rot": "Rotational error (DDP): ",
            "comp_time": "Computional time (DDP): ",
            "conf": "Solver configuration (DDP): ",
        },
        column_format={"x": "cccc", "rot": "cccc", "comp_time": "cccc", "conf": "cc"},
        how_bold={"x": None, "rot": None, "comp_time": None, "conf": None},
        caption_meta={
            "factor_integration_time": factor_integration_time,
            "end_error": time_step_end_error,
            "\\beta": beta,
        },
        filename_meta={
            "end_error": time_step_end_error,
            "factor_int": factor_integration_time,
        },
    )


if __name__ == "__main__":
    # define relevant constants
    goal_meta_data = pd.read_csv("../data/setpoint/mpc_preplanning.csv")

    time_step_end_error = 300
    time_horizon = 3000

    for beta in [1.0, 1.2, 1.4]:
        meta_data = goal_meta_data[(goal_meta_data["beta"] == beta)]
        eval_single_experiment(
            meta_data,
            time_step_end_error,
            time_horizon,
            factor_integration_time=4,
            beta=beta,
        )

    print("Eval ddp precomputed")
    eval_preplanning(
        goal_meta_data,
        time_steps_end_error=time_step_end_error,
        time_horizon=time_horizon,
        factor_integration_time=4,
    )
