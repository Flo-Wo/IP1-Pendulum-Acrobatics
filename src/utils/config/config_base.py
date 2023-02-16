import mujoco
import numpy as np
from utils import DataBase
from utils.config.config_utils import read_model


class ConfigModelBase(DataBase):
    def __init__(
        self,
        q_config: np.ndarray,
        model_name: str,
        model_type: str = "normal",
        len_pendulum: float = 0.3,
    ):

        self.model_type = model_type
        # set the pendulum length
        self.len_pendulum = len_pendulum

        # read the models
        self.mj_model = read_model(model_type, ".xml", len_pendulum)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.pin_robot = read_model(model_type, ".urdf", len_pendulum)
        self.pin_model = self.pin_robot.model
        self.pin_data = self.pin_robot.data
        # self.pin_data = self.pin_model.createData()

        # set the robot to it's start configuration
        self.mj_data.qpos[:] = q_config
        mujoco.mj_step(self.mj_model, self.mj_data)
        self.q_config = q_config
        # define a string representation for automatic filename generation
        self.model_name = model_name

    # DataBase method's implementations, a config is uniquely defined by the model type,
    # the config string and the length of the pendulum
    def __str__(self):
        return "{}_{}_{}".format(self.model_type, self.model_name, self.len_pendulum)

    def save_to_file(self) -> dict:
        return {}

    def save_to_metadata(self) -> dict:
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "len_pendulum": self.len_pendulum,
        }

    def pend_end_world(self):
        return self.mj_data.sensordata[0:3]

    def pend_beg_world(self):
        return self.mj_data.sensordata[7:10]

    def pend_end_id(self):
        return self.pin_model.getFrameId("links/endeffector")

    def pend_beg_id(self):
        return self.pin_model.getFrameId("links/pendulum_pole")
