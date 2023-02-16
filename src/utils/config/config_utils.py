import logging
import mujoco
from pinocchio import RobotWrapper


def add_folder(subfolder: str = ""):
    """Decorate to add the metafile subfolder to each of the reader files.

    Parameters
    ----------
    subfolder : str, optional
        Metafolder, by default "". Other functions use it in the form
        of ``subfolder="metafiles/``.
    """

    def decorate_folder(func: callable):
        def wrapped(filename, *args, **kwargs):
            try:
                return func("../wam/{}{}".format(subfolder, filename), *args, **kwargs)
            except:
                return func("./wam/{}{}".format(subfolder, filename), *args, **kwargs)

        return wrapped

    return decorate_folder


def read_model(model_type: str, file_end: str = ".urdf", len_pendulum: float = 0.3):
    """Read the model given the model type, the file ending and the pendulum length.
    This function will read the metafile, substitute the length of the pendulum and
    then read a temporary file with the set pendulum length.

    Parameters
    ----------
    model_type : str
        Type of the model, currently either "rot" or "normal".
    file_end : str, optional
        File ending, currently either ".urdf" (for pinocchio)
        or ".xml" (for mujoco), by default ".urdf".
    len_pendulum : float, optional
        Length of the pendulum, by default 0.3.

    Returns
    -------
    Mujoco/Pinocchio Model
        Robot model, either for mujoco or pinocchio.

    Raises
    ------
    NotImplementedError
        If you enter the wrong model type.
    NotImplementedError
        If you enter an incorrect file ending.
    """
    if model_type == "rot":
        logging.info("READ ROTATED")
        filename = "rot_wam_pend"
    elif model_type == "normal":
        logging.info("READ NORMAL")
        filename = "wam_pend"
    else:
        raise NotImplementedError

    # create a tmp-file with the correct pendulum length
    tmp_end = _set_len_pendulum(filename, file_end, len_pendulum)

    if file_end == ".urdf":
        return _read_raw(filename, tmp_end, file_end, RobotWrapper.BuildFromURDF)
    elif file_end == ".xml":
        return _read_raw(filename, tmp_end, file_end, mujoco.MjModel.from_xml_path)
    else:
        raise NotImplementedError


@add_folder()
def _read_raw(filename: str, tmp_end: str, file_end: str, reader: callable):
    temp_end = tmp_end + file_end
    return reader(filename + temp_end)


def _set_len_pendulum(filename: str, file_end: str, len_pendulum: float) -> str:
    filedata = _read_file(filename, file_end)
    replaced_data = _find_and_replace(filedata, len_pendulum)
    tmp_end = "_temp_{}".format(len_pendulum)
    return _save_file(filename, tmp_end, file_end, replaced_data)


@add_folder("metafiles/")
def _read_file(filename: str, file_end: str) -> str:
    with open(filename + file_end, "r") as file:
        filedata = file.read()
    return filedata


@add_folder()
def _save_file(filename: str, tmp_end: str, file_end: str, replaced_data: str):
    with open(filename + tmp_end + file_end, "w") as file:
        file.write(replaced_data)
    return tmp_end


def _find_and_replace(model_descr: str, len_pendulum: float = 0.3):
    logging.info("Pendulum length: {}".format(len_pendulum))
    logging.info("Inertia position: {}".format(len_pendulum / 2))
    # replace the length of the pendulum and the position of the sensors
    model_descr = model_descr.replace("pendulum_length_full", str(len_pendulum))
    # replace the position of the inertia mass
    model_descr = model_descr.replace(
        "pendulum_length_half", str(float(len_pendulum / 2))
    )
    return model_descr
