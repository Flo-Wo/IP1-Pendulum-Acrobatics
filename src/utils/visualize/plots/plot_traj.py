from typing import Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(
    goal: np.ndarray,
    experiments: Dict[str, np.ndarray],
    header: str = "",
):
    for exp_traj in experiments.values():
        assert np.shape(goal) == np.shape(
            exp_traj
        ), "Different shapes {0} vs {1}, plots don't work.".format(
            np.shape(goal), np.shape(exp_traj)
        )
    n_subplots = goal.shape[1]
    idx = np.arange(0, goal.shape[0])
    fix, axs = plt.subplots(n_subplots, 1, figsize=(15, 6), edgecolor="k")
    # axs = axs.ravel()
    for i in range(n_subplots):
        axs[i].plot(idx, goal[:, i], "r--", label="goal")
        for name, exp_traj in experiments.items():
            axs[i].plot(idx, exp_traj[:, i], label=name)

    header += "Cumulated l2-errors: "
    for name, exp_traj in experiments.items():
        err = np.sum(np.linalg.norm(goal - exp_traj, axis=1))
        err_avg = np.average(np.linalg.norm(goal - exp_traj, axis=1))
        print("plot_traj.py: err_avg = {}".format(err_avg))
        header += "{}: {:.2f}, ".format(name, err)
        header += "Average l2-error {}: {:.2f}, ".format(name, err_avg)

    plt.suptitle(header)
    plt.legend()


def plot_single_traj(traj: np.ndarray, label: str = "traj"):
    try:
        idx = np.arange(0, traj.shape[0])
        n_subplots = traj.shape[1]
    except AttributeError:
        idx = np.arange(0, len(traj))
        n_subplots = traj[0].shape[0]
    except IndexError:
        _plot_scalar_traj(traj, label)
        return

    fig, axs = plt.subplots(n_subplots, 1, figsize=(15, 6), edgecolor="k")
    for i in range(n_subplots):
        try:
            axs[i].plot(idx, traj[:, i], "r--", label=label)
        except:
            axs[i].plot(idx, [x[i] for x in traj], "r--", label=label)

    plt.legend()


def _plot_scalar_traj(traj: list, label: str = "traj"):
    fig = plt.figure()
    idx = np.arange(0, len(traj))
    plt.plot(idx, traj, label=label)
    plt.legend()
