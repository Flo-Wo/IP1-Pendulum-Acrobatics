import numpy as np
from utils.config.config_base import ConfigModelBase


class ConfigWAM(ConfigModelBase):
    def __init__(self, model_name: str, len_pendulum: float = 0.3):
        """Configuration class for our WAM models used in all experiments.

        Parameters
        ----------
        model_name : str
            A unique model name which is mapped to a specific configuration.
        len_pendulum : float, optional
            Length of the pendulum, by default 0.3.

        Raises
        ------
        NotImplementedError
            If you choose a wrong model_name.
        """
        if model_name == "standard":
            q_config = np.array([0, -0.3, 0, 0.3, 0, 0])
            model_type = "normal"
        elif model_name == "standard_angled":
            q_config = np.array([0, -1.7, 0, 1.7, 0, 0])
            model_type = "normal"
        elif model_name == "rotated":
            q_config = np.array([0, -0.78, 0, 2.37, 0, 0])
            model_type = "rot"
        elif model_name == "human":
            q_config = np.array([0, -1.6, 1.55, 1.6, 0, 1.55])
            model_type = "rot"
        else:
            raise NotImplementedError("Wrong argument for the WAM model.")

        super(ConfigWAM, self).__init__(
            q_config,
            model_name,
            model_type=model_type,
            len_pendulum=len_pendulum,
        )
