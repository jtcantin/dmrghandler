"""
This module contains functions to store and load data from HDF5 files.
Code mostly taken from qoptbench (https://github.com/zapatacomputing/bobqat-qb-opt-benchmark)"""

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import pandas as pd
import psutil

log = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = "./data_storage"
# DEFAULT_USER_DIR = "./user_storage"
# DEFAULT_QUBO_DIR = "./qubo_storage"
# DEFAULT_RAW_DATA_DIR = "./raw_data_files"
# DEFAULT_LOG_DIR = "./log_files"

AnyPath = None
if TYPE_CHECKING:
    from _typeshed import AnyPath


def has_handle(fpath):
    """Check if a file is open.
    From: https://stackoverflow.com/a/44615315/10548384
    """
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if fpath == item.path:
                    return True
        except Exception:
            pass

    return False


def show_dataset_hierarchy(hdf5_filename: AnyPath):
    hdf5_filename = Path(hdf5_filename)
    hdf5_file_object = h5py.File(hdf5_filename, "r")

    info_string = show_dataset_hierarchy_file_obj(hdf5_file_object)

    return info_string


def show_dataset_hierarchy_file_obj(hdf5_file_object: h5py.File, indent_level: int = 0):
    """
    Shows the hierarchy of datasets and groups inside an hdf5 file object.
    Acts recursively.

    Args:
        hdf5_file_object: An hdf5 file object.
        indent_level: The level of indentation to use when printing the hierarchy.

    Output:
        info_string: A string containing the hierarchy of datasets
                        and groups in the hdf5 file object.
    """

    # Get all top-level datasets and groups
    top_level = list(hdf5_file_object.keys())

    # Iterate through the top-level datasets and groups, going down the hierarchy
    # and collecting the names of all datasets and groups in a string, labeling the groups
    info_string = ""
    current_level = indent_level
    for iter, data_object_name in enumerate(top_level):
        # print(data_object_name)
        if isinstance(hdf5_file_object[data_object_name], h5py.Dataset):
            if indent_level > 0:
                info_string += (2 * indent_level) * " " + "--" + data_object_name + "\n"
            else:
                info_string += data_object_name + "\n"
        elif isinstance(hdf5_file_object[data_object_name], h5py.Group):
            group_label = " (Group) "

            # If the group is actually a Panadas dataframe, then label it as such
            panadas_dataframe_bool = "pandas_type" in list(
                hdf5_file_object[data_object_name].attrs.keys()
            )
            if panadas_dataframe_bool:
                group_label = " (Pandas dataframe) "

            if indent_level > 0:
                info_string += (
                    (2 * indent_level) * " "
                    + "--"
                    + data_object_name
                    + group_label
                    + "\n"
                )
            else:
                info_string += data_object_name + group_label + "\n"

            # Don't go down the hierarchy if the group is a Pandas dataframe
            if not panadas_dataframe_bool:
                info_string += show_dataset_hierarchy_file_obj(
                    hdf5_file_object[data_object_name], indent_level + 1
                )
        else:
            raise TypeError("Unknown data object type")
    return info_string


def generate_uuid() -> str:
    """Generate a unique ID, based on UUID Version 4."""
    return str(uuid.uuid4())


def save_many_variables_to_hdf5(
    hdf5_filepath: AnyPath,
    variables: dict,
    access_mode: str = "a",
    group: str = None,
    overwrite: bool = False,
) -> None:
    """Save multiple variables to the same HDF5 file.
    If the HDF5 file does not exist, it is created.

    Args:
        hdf5_filepath: The path to the HDF5 file.
        variables: A dictionary of variables to save to the HDF5 file.
                If the variable is  a string, number, or boolean, it is saved as a scalar dataset.
                                            Note that when reading a string, it is returned as a
                                                bytes object.
                                            Decode it using .decode("utf-8"):
                                                file["group_path"][()].decode("utf-8")
                                    a list, it is saved as a 1D dataset.
                                    a numpy ndarray, it is saved as an nD dataset.
                                            Up to 3D has been tested.
                                    a Pandas dataframe, it is saved using the Pandas HDF5 API.
        access_mode: The access mode for the HDF5 file. Defaults to "a".
        group: The group in the HDF5 file to save the variables to. None means top level group.
                Defaults to None.
        overwrite: If a variable in variables already exists in the HDF5 file, overwrite it.
                Defaults to False. If False, an error is raised if the variable already exists.
    """
    hdf5_filepath = Path(hdf5_filepath)
    # Check if the file is already open
    if has_handle(hdf5_filepath.resolve()):
        raise ValueError(
            f"File {hdf5_filepath.resolve()} is already open by another process."
        )

    if variables is None:
        Warning("No variables to save to HDF5 file.")
        return

    pandas_dict = {}
    with h5py.File(hdf5_filepath, access_mode) as hdf5_output_file:
        if group is None:
            output_object = hdf5_output_file
        else:
            output_object = hdf5_output_file.require_group(group)

        for variable_name, variable_value in variables.items():
            try:
                if variable_value is None:
                    continue
                # print(variable_name)
                if variable_name in output_object.keys() and not overwrite:
                    raise ValueError(
                        f"Variable {variable_name} already exists in HDF5 file {hdf5_filepath}."
                    )
                elif variable_name in output_object.keys() and overwrite:
                    del output_object[variable_name]
                if isinstance(variable_value, pd.DataFrame):
                    pandas_dict[variable_name] = variable_value

                else:
                    # print(variable_value)
                    output_object.create_dataset(variable_name, data=variable_value)
            except TypeError:
                log.error(
                    f"Error saving variables to HDF5 file. Couldn't save {variable_name} in group {group}."
                )
                raise

    if len(pandas_dict.keys()) > 0:
        for variable_name, variable_value in pandas_dict.items():
            variable_value.to_hdf(
                hdf5_filepath, key=group + "/" + variable_name, mode="a"
            )


def get_new_filename() -> str:
    """Generate a new filename based on UUID Version 4."""
    return f"{generate_uuid()}.hdf5"


def get_generic_filename() -> Path:
    """Get the filename of the data storage file.
    Currently, this is a UUID Version 4 generated name.
    """

    filename = get_new_filename()
    return Path(filename)


def get_output_hdf5_filename(
    output_hdf5_filename: AnyPath = None,
) -> Path:
    """Get the filename of the data storage file.
    If no filename is provided, a UUID Version 4 generated name is used.
    Otherwise, the provided filename is used and returned as a Path object.
    """

    if output_hdf5_filename is None:
        output_hdf5_filename = get_generic_filename()

    else:
        output_hdf5_filename = Path(output_hdf5_filename)

    return output_hdf5_filename


def get_output_hdf5_filepath(
    output_hdf5_filename: AnyPath = None,
    output_dir: AnyPath = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Get the full filepath of the data storage file.
    If no filename is provided, a UUID Version 4 generated name is used.
    Otherwise, the provided filename is used.
    The full filepath is returned as a Path object.
    """

    output_hdf5_filename = get_output_hdf5_filename(
        output_hdf5_filename=output_hdf5_filename,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / output_hdf5_filename
