import logging
import shutil
from pathlib import Path

import numpy as np
import yaml

import dmrghandler.hdf5_io as hdf5_io

log = logging.getLogger(__name__)


def ensure_required_in_dict(dictionary: dict, required_keys: list[str]):
    """
    Ensures that a dictionary has the required keys.

    Args:
        dictionary: The dictionary to check.
        required_keys: A list of the required keys.

    Raises:
        KeyError: If a required key is not in the dictionary.
    Code  taken from qoptbench (https://github.com/zapatacomputing/bobqat-qb-opt-benchmark)
    """
    for key in required_keys:
        if key not in dictionary.keys():
            raise KeyError(
                f"Required key {key} not in dictionary."
                + f" Required keys are: {required_keys}."
                + f"Current keys are: {dictionary.keys()}."
                + "Most keys allow values of None or 'N/A'."
            )


def load_configuration_data(config_file: str) -> dict:
    """
    Load configuration data from a YAML file.

    Args:
        congif_file: The path to the YAML file.

    Returns:
        A dictionary with the configuration data.

    """
    with open(config_file, "r") as file:
        data = yaml.safe_load(file)

    data["dmrg_basic_config"]["sweep_schedule_bond_dims"] = eval(
        data["dmrg_basic_config"]["sweep_schedule_bond_dims"]
    )
    data["dmrg_basic_config"]["sweep_schedule_noise"] = eval(
        data["dmrg_basic_config"]["sweep_schedule_noise"]
    )
    data["dmrg_basic_config"]["sweep_schedule_davidson_threshold"] = eval(
        data["dmrg_basic_config"]["sweep_schedule_davidson_threshold"]
    )
    return data


def save_dmrg_configuration_data(config_file: str, data: dict):
    """
    Save configuration data to a YAML file.

    Args:
        config_file: The path to the YAML file.
        data: The data to save.
    """
    config_file = Path(config_file)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    # Top level keys
    ensure_required_in_dict(
        dictionary=data,
        required_keys=[
            "data_config",
            "looping_config",
            "dmrg_basic_config",
            "dmrg_advanced_config",
        ],
    )
    # Data information
    ensure_required_in_dict(
        dictionary=data["data_config"],
        required_keys=[
            "data_file_path",
            "main_storage_folder_path",
            "plot_filename_prefix",
            "original_data_file_path",
            "submit_script_file",
        ],
    )
    # Looping parameters
    ensure_required_in_dict(
        dictionary=data["looping_config"],
        required_keys=[
            # Looping parameters
            "max_bond_dimension",
            "max_time_limit_sec",
            "min_energy_change_hartree",
            "extrapolation_type",
            "starting_bond_dimension",
            "track_mem",
        ],
    )

    # DMRG basic configuration
    ensure_required_in_dict(
        dictionary=data["dmrg_basic_config"],
        required_keys=[
            "max_num_sweeps",
            "energy_convergence_threshold",
            "sweep_schedule_bond_dims",
            "sweep_schedule_noise",
            "sweep_schedule_davidson_threshold",
            "temp_dir",
            "restart_dir",
            # "mps_storage_folder",
            "reordering_method",
            "init_state_bond_dimension",
            "init_state_seed",
            "initial_mps_method",
            "factor_half_convention",
            "symmetry_type",
            "num_threads",
            "n_mkl_threads",
            # "num_orbitals",
            # "num_spin_orbitals",
            # "num_electrons",
            # "two_S",
            # "two_Sz",
            # "orb_sym",
            # "core_energy",
        ],
    )
    # DMRG advanced configuration
    ensure_required_in_dict(
        dictionary=data["dmrg_advanced_config"],
        required_keys=[
            "occupancy_hint",
            "full_fci_space_bool",
            "init_state_direct_two_site_construction_bool",
            "davidson_type",
            "eigenvalue_cutoff",
            "davidson_max_iterations",
            "davidson_max_krylov_subspace_size",
            "lowmem_noise_bool",
            "sweep_start",
            "initial_sweep_direction",
            "stack_mem_ratio",
        ],
    )

    with open(config_file, "w") as file:
        yaml.dump(data, file)


def gen_config_files(
    data_file_list,
    config_dict,
    dmrg_advanced_config=None,
    config_storage_folder="config_store",
    config_file_prefix="config_",
):
    config_files_list = []
    if dmrg_advanced_config == None:
        dmrg_advanced_config = {
            "occupancy_hint": None,
            "full_fci_space_bool": True,
            "init_state_direct_two_site_construction_bool": False,
            "davidson_type": None,  # Default is None, for "Normal"
            "eigenvalue_cutoff": 1e-20,  # Cutoff of eigenvalues, default is 1e-20
            "davidson_max_iterations": 4000,  # Default is 4000
            "davidson_max_krylov_subspace_size": 50,  # Default is 50
            "lowmem_noise_bool": False,  # Whether to use a lower memory version of the noise, default is False
            "sweep_start": 0,  # Default is 0, where to start sweep
            "initial_sweep_direction": None,  # Default is None, True means forward sweep (left-to-right)
            "stack_mem": 1073741824,
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }
    config_dict_single_file_list = []
    for data_iter, data_file_path in enumerate(data_file_list):
        folder_uuid = hdf5_io.generate_uuid()
        plot_filename_prefix = common_or_list(
            config_dict["plot_filename_prefix_list"], data_iter
        )

        main_storage_folder_path = (
            Path(config_dict["main_storage_folder_path_prefix"]) / folder_uuid
        )
        config_storage_folder = Path(config_storage_folder)

        config_storage_folder.mkdir(parents=True, exist_ok=True)

        data_prep_path = config_storage_folder / folder_uuid
        # Copy datafile to data_prep_path folder
        data_file_path = Path(data_file_path)
        data_prep_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_file_path, data_prep_path / data_file_path.name)

        config_file = data_prep_path / Path(
            config_file_prefix + f"{str(data_iter)}_" + str(folder_uuid) + ".yaml"
        )

        log.info(f"Copied data file {data_file_path} to {data_prep_path}")

        data_config = {
            "original_data_file_path": str(data_file_path),
            "data_file_path": str(data_prep_path / data_file_path.name),
            "main_storage_folder_path": str(main_storage_folder_path),
            "plot_filename_prefix": str(plot_filename_prefix),
            "data_prep_path": str(data_prep_path),
            "python_run_file": str(data_prep_path / f"dmrg_loop_run_{folder_uuid}.py"),
            "submit_script_file": str(data_prep_path / f"submit_{folder_uuid}.sh"),
            "config_file": str(config_file),
            "folder_uuid": str(folder_uuid),
        }

        max_bond_dimension = common_or_list(
            config_dict["max_bond_dimension_list"], data_iter
        )
        max_time_limit_sec = common_or_list(
            config_dict["max_time_limit_sec_list"], data_iter
        )
        min_energy_change_hartree = common_or_list(
            config_dict["min_energy_change_hartree_list"], data_iter
        )
        extrapolation_type = common_or_list(
            config_dict["extrapolation_type_list"], data_iter
        )
        starting_bond_dimension = common_or_list(
            config_dict["starting_bond_dimension_list"], data_iter
        )
        track_mem = common_or_list(config_dict["track_mem"], data_iter)
        looping_config = {
            "max_bond_dimension": max_bond_dimension,
            "max_time_limit_sec": max_time_limit_sec,
            "min_energy_change_hartree": min_energy_change_hartree,
            "extrapolation_type": extrapolation_type,
            "starting_bond_dimension": starting_bond_dimension,
            "track_mem": track_mem,
        }

        max_num_sweeps = common_or_list(config_dict["max_num_sweeps_list"], data_iter)
        energy_convergence_threshold = common_or_list(
            config_dict["energy_convergence_threshold_list"], data_iter
        )
        if (
            "do_single_calc" in dmrg_advanced_config
            and dmrg_advanced_config["do_single_calc"]
        ):
            sweep_schedule_bond_dims = common_or_list(
                config_dict["sweep_schedule_bond_dims_parameters"], data_iter
            )
        else:
            sweep_schedule_bond_dims = generate_sweep_schedule(
                common_or_list(
                    config_dict["sweep_schedule_bond_dims_parameters"], data_iter
                ),
                starting_bond_dimension,
            )

        sweep_schedule_noise = common_or_list(
            config_dict["sweep_schedule_noise_list"], data_iter
        )
        sweep_schedule_davidson_threshold = common_or_list(
            config_dict["sweep_schedule_davidson_threshold_list"], data_iter
        )

        init_state_bond_dimension = starting_bond_dimension // common_or_list(
            config_dict["init_state_bond_dimension_division_factor_list"], data_iter
        )
        temp_dir = str(main_storage_folder_path / "temp")
        init_state_seed = common_or_list(config_dict["init_state_seed_list"], data_iter)
        initial_mps_method = common_or_list(
            config_dict["initial_mps_method_list"], data_iter
        )
        factor_half_convention = common_or_list(
            config_dict["factor_half_convention_list"], data_iter
        )
        symmetry_type = common_or_list(config_dict["symmetry_type_list"], data_iter)
        num_threads = common_or_list(config_dict["num_threads_list"], data_iter)
        n_mkl_threads = common_or_list(config_dict["n_mkl_threads_list"], data_iter)

        dmrg_basic_config = {
            "max_num_sweeps": max_num_sweeps,
            "energy_convergence_threshold": energy_convergence_threshold,
            "sweep_schedule_bond_dims": str(sweep_schedule_bond_dims),
            "sweep_schedule_noise": str(sweep_schedule_noise),
            "sweep_schedule_davidson_threshold": str(sweep_schedule_davidson_threshold),
            "temp_dir": temp_dir,
            "restart_dir": None,
            # "mps_storage_folder",
            "reordering_method": "none",
            "init_state_bond_dimension": init_state_bond_dimension,
            "init_state_seed": init_state_seed,
            "initial_mps_method": initial_mps_method,
            "factor_half_convention": factor_half_convention,
            "symmetry_type": symmetry_type,
            "num_threads": num_threads,
            "n_mkl_threads": n_mkl_threads,
            # "num_orbitals",
            # "num_spin_orbitals",
            # "num_electrons",
            # "two_S",
            # "two_Sz",
            # "orb_sym",
            # "core_energy",
        }

        config_dict_single_file = {
            "data_config": data_config,
            "looping_config": looping_config,
            "dmrg_basic_config": dmrg_basic_config,
            "dmrg_advanced_config": dmrg_advanced_config,
        }

        save_dmrg_configuration_data(
            config_file=config_file, data=config_dict_single_file
        )
        log.info(f"Saved configuration file {config_file}")
        config_files_list.append(config_file)
        config_dict_single_file_list.append(config_dict_single_file)

    return config_files_list, config_dict_single_file_list


def common_or_list(list, iiter):
    if len(list) == 1:
        return list[0]
    else:
        return list[iiter]


def generate_sweep_schedule(factor_count_tuples, starting_bond_dimension):
    sweep_schedule_bond_dims = []
    for factor, count in factor_count_tuples:
        new_sector = [np.max([1, starting_bond_dimension // factor])] * count
        sweep_schedule_bond_dims = sweep_schedule_bond_dims + new_sector

    return sweep_schedule_bond_dims


def fwd_reverse_schedule(start_bd, num_points):
    schedule = [start_bd // 2] * 4 + [start_bd] * 5
    for i in range(num_points):
        schedule += [int((0.9091 ** (i + 1)) * start_bd)] * 8
    return schedule


def fwd_reverse_schedule_noise(noise_1, noise_2, num_points):
    schedule = [noise_1] * 4 + [noise_2] * 4 + [0]
    for i in range(num_points):
        schedule += [0] * 8
    return schedule


def fwd_reverse_schedule_threshold(thresh, num_points):
    schedule = [thresh] * 9
    for i in range(num_points):
        schedule += [thresh] * 8
    return schedule
