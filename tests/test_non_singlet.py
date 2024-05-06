import logging
import os
import unittest
from pathlib import Path

import h5py
import numpy as np
import numpy.testing as npt
import scipy as sp
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

import dmrghandler.config_io as config_io
import dmrghandler.dmrg_calc_prepare as dmrg_calc_prepare
import dmrghandler.slurm_scripts as slurm_scripts

test_rtol = 1e-5
test_atol = 1e-8
large_test_rtol = 1e-4
large_test_atol = 1e-3
test_rsquared_rtol = 1e-1
test_rsquared_atol = 1e-1

# log = logging.getLogger("{Path(config_file_name).stem}")
log = logging.getLogger("dmrghandler")
log.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler("dmrghandler_test.log")
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - Line %(lineno)d - %(message)s"
)
fh.setFormatter(formatter)
# add the handlers to the log
log.addHandler(fh)


class TestNonSinglet(unittest.TestCase):
    def test_4a_48_qubits(self):
        # data_file = "fcidump.test"

        data_files_folder = Path("./tests/test_data")

        data_file_list_file = [
            "fcidump.test",
        ]

        data_file_list = []
        for data_file in data_file_list_file:
            data_file_path = Path(data_files_folder) / Path(data_file)
            data_file_list.append(str(data_file_path))
        print(
            f"data_file_list: {data_file_list}"
        )  # IF ERRORS HERE, WRONG PYTHON VERSION!!!!!

        config_dict = {
            "plot_filename_prefix_list": [
                "4a_48qubit_test",
            ],
            "main_storage_folder_path_prefix": "./data_storage",
            "max_bond_dimension_list": [30],
            "max_time_limit_sec_list": [240],  # 4min
            "min_energy_change_hartree_list": [1e-4],
            "extrapolation_type_list": ["discard_weights"],
            "starting_bond_dimension_list": [5],
            "max_num_sweeps_list": [20],
            "energy_convergence_threshold_list": [1e-8],
            "sweep_schedule_bond_dims_parameters": [
                [(2, 4), (1, 5)]
            ],  # (division_factor, count),
            # e.g. [(2, 4), (1, 5)] and init_bd of 3 ->[1, 1, 1, 1, 3, 3, 3, 3, 3]
            "sweep_schedule_noise_list": [[1e-4] * 4 + [1e-5] * 4 + [0]],
            "sweep_schedule_davidson_threshold_list": [[1e-10] * 9],
            "init_state_bond_dimension_division_factor_list": [2],
            "init_state_seed_list": [658724],
            "initial_mps_method_list": ["random"],
            "factor_half_convention_list": [True],
            # "symmetry_type_list": ["SZ", "SZ", "SU(2)", "SU(2)"],
            "symmetry_type_list": ["SU(2)"],
            "num_threads_list": [4],
            "n_mkl_threads_list": [1],
        }

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
            "stack_mem": 10 * 1024 * 1024 * 1024,  # =170*1024*1024*1024
            "stack_mem_ratio": 0.9,
        }

        config_files_list, config_dict_single_file_list = config_io.gen_config_files(
            data_file_list=data_file_list,
            config_dict=config_dict,
            dmrg_advanced_config=dmrg_advanced_config,
            config_file_prefix="4a_48qubit_test_",
        )
        print(f"config_files_list: {config_files_list}")
        # print(f"config_dict_single_file_list: {config_dict_single_file_list}")

        submit_dict = {
            "time_cap_string": "00-23:59:00",
            "job_name": "dmrg_thresholding",
            "email": "joshua.cantin@utoronto.ca",
            "account_name": "rrg-izmaylov",
            "tasks_per_node": "1",
            "cpus_per_task": "40",
            "partition": "debug",
            "python_environment_location": "env_dmrg_thresholding",
        }

        slurm_scripts.gen_run_files(submit_dict, config_dict_single_file_list)

        submit_commands = slurm_scripts.gen_submit_commands(
            config_dict_single_file_list
        )
        scratch_sim_path = Path("tests/scratch_sim")
        scratch_sim_path.mkdir(parents=True, exist_ok=True)
        scratch_sim_path_absolute = scratch_sim_path.resolve()
        for config_dict in config_dict_single_file_list:
            data_config = config_dict["data_config"]
            python_run_file_name = data_config["python_run_file"]
            os.environ["SCRATCH"] = str(scratch_sim_path_absolute)
            os.system(f"env_dmrghandler/bin/python {python_run_file_name}")
            log.debug("DMRG NOW EXITED")
            # Get results
            main_storage_folder_path = data_config["main_storage_folder_path"]
            hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            with h5py.File(hdf5_file_path, "r") as f:
                dmrg_energies = f["/final_dmrg_results/past_energies_dmrg"][:]
                dmrg_bond_dimensions = f["/final_dmrg_results/bond_dims_used"][:]
                discarded_weights = f["/final_dmrg_results/past_discarded_weights"][:]

            log.info(f"dmrg_energies: {dmrg_energies}")
            log.info(f"dmrg_bond_dimensions: {dmrg_bond_dimensions}")
            log.info(f"discarded_weights: {discarded_weights}")

            npt.assert_allclose(
                dmrg_energies[-1], -5103.5559419269, rtol=test_rtol, atol=1e-3
            )
