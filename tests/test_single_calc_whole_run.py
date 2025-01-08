import logging
import os
import unittest
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import numpy.testing as npt
import scipy as sp
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

import dmrghandler.config_io as config_io
import dmrghandler.dmrg_calc_prepare as dmrg_calc_prepare
import dmrghandler.slurm_scripts as slurm_scripts
import dmrghandler.pyscf_wrappers as pyscf_wrappers
import dmrghandler.qchem_dmrg_calc as qchem_dmrg_calc
import pyscf.tools.fcidump

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


class TestSingleCalcWholeRun(unittest.TestCase):
    def setUp(self):
        self.clean_up_files()

    def tearDown(self):
        # Wait 30s, doing nothing else
        time.sleep(30)

    def test_4a_48_qubits_reverse_schedule(self):
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

        num_reverse_points = 5
        config_dict = {
            "plot_filename_prefix_list": [
                "4a_48qubit_test",
            ],
            "main_storage_folder_path_prefix": "./data_storage",
            "max_bond_dimension_list": [35],
            "max_time_limit_sec_list": [240],  # 4min
            "min_energy_change_hartree_list": [1e-4],
            "extrapolation_type_list": ["discard_weights"],
            "starting_bond_dimension_list": [5],
            "max_num_sweeps_list": [num_reverse_points + 9 + 10],
            "energy_convergence_threshold_list": [1e-8],
            # "sweep_schedule_bond_dims_parameters": [
            #     [(2, 4), (1, 5)]
            # ],  # (division_factor, count),
            # # e.g. [(2, 4), (1, 5)] and init_bd of 3 ->[1, 1, 1, 1, 3, 3, 3, 3, 3]
            "sweep_schedule_bond_dims_parameters": [
                config_io.fwd_reverse_schedule(
                    start_bd=30, num_points=num_reverse_points
                ),
            ],
            "sweep_schedule_noise_list": [
                config_io.fwd_reverse_schedule_noise(
                    noise_1=1e-4, noise_2=1e-5, num_points=num_reverse_points
                )
            ],
            "sweep_schedule_davidson_threshold_list": [
                config_io.fwd_reverse_schedule_threshold(
                    thresh=1e-10, num_points=num_reverse_points
                )
            ],
            "init_state_bond_dimension_division_factor_list": [2],
            "init_state_seed_list": [658724],
            "initial_mps_method_list": ["random"],
            "factor_half_convention_list": [True],
            # "symmetry_type_list": ["SZ", "SZ", "SU(2)", "SU(2)"],
            "symmetry_type_list": ["SU(2)"],
            "num_threads_list": [4],
            "n_mkl_threads_list": [1],
            "track_mem": [False],
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
            "do_single_calc": True,
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
            return_value = os.system(
                f"env_dmrghandler/bin/python {python_run_file_name}"
            )
            if return_value != 0:
                raise Exception("DMRG failed to run properly")
            log.debug("DMRG NOW EXITED")
            # Get results
            main_storage_folder_path = data_config["main_storage_folder_path"]
            hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            with h5py.File(hdf5_file_path, "r") as f:
                dmrg_energies = f["/final_dmrg_results/past_energies_dmrg"][:]
                dmrg_bond_dimensions = f["/final_dmrg_results/bond_dims_used"][:]
                discarded_weights = f["/final_dmrg_results/past_discarded_weights"][:]

                self.check_csf_presence_and_threshold(config_dict, f)

            log.info(f"dmrg_energies: {dmrg_energies}")
            log.info(f"dmrg_bond_dimensions: {dmrg_bond_dimensions}")
            log.info(f"discarded_weights: {discarded_weights}")

            npt.assert_allclose(
                dmrg_energies[-1], -5103.5559419269, rtol=test_rtol, atol=1e-3
            )

    def test_4a_48_qubits_fielder_interaction_matrix(self):
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
            "max_time_limit_sec_list": [4 * 60],  # 10 min #[240],  # 4min
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
            "track_mem": [False],
            "reordering_method_list": ["fiedler, interaction matrix"],
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
            # "do_single_calc": False,
            "num_states": 1,
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
            return_value = os.system(
                f"env_dmrghandler/bin/python {python_run_file_name}"
            )
            if return_value != 0:
                raise Exception("DMRG failed to run properly")
            log.debug("DMRG NOW EXITED")
            # Get results
            main_storage_folder_path = data_config["main_storage_folder_path"]
            hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            with h5py.File(hdf5_file_path, "r") as f:
                dmrg_energies = f["/final_dmrg_results/past_energies_dmrg"][:]
                dmrg_bond_dimensions = f["/final_dmrg_results/bond_dims_used"][:]
                discarded_weights = f["/final_dmrg_results/past_discarded_weights"][:]

                self.check_csf_presence_and_threshold(config_dict, f)

            log.info(f"dmrg_energies: {dmrg_energies}")
            log.info(f"dmrg_bond_dimensions: {dmrg_bond_dimensions}")
            log.info(f"discarded_weights: {discarded_weights}")

            npt.assert_allclose(
                dmrg_energies[0], -5103.5559419269, rtol=test_rtol, atol=1e-3
            )

    def test_4a_48_qubits_fielder_interaction_matrix_SU2(self):
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
            "max_time_limit_sec_list": [4 * 60],  # 10 min #[240],  # 4min
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
            "track_mem": [False],
            "reordering_method_list": ["fiedler, interaction matrix, SU(2) calc"],
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
            # "do_single_calc": False,
            "num_states": 1,
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
            return_value = os.system(
                f"env_dmrghandler/bin/python {python_run_file_name}"
            )
            if return_value != 0:
                raise Exception("DMRG failed to run properly")
            log.debug("DMRG NOW EXITED")
            # Get results
            main_storage_folder_path = data_config["main_storage_folder_path"]
            hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            with h5py.File(hdf5_file_path, "r") as f:
                dmrg_energies = f["/final_dmrg_results/past_energies_dmrg"][:]
                dmrg_bond_dimensions = f["/final_dmrg_results/bond_dims_used"][:]
                discarded_weights = f["/final_dmrg_results/past_discarded_weights"][:]

                self.check_csf_presence_and_threshold(config_dict, f)

            log.info(f"dmrg_energies: {dmrg_energies}")
            log.info(f"dmrg_bond_dimensions: {dmrg_bond_dimensions}")
            log.info(f"discarded_weights: {discarded_weights}")

            npt.assert_allclose(
                dmrg_energies[0], -5103.5559419269, rtol=test_rtol, atol=1e-3
            )

    @unittest.expectedFailure  # Because of non-convergence, it seems
    def test_4a_48_qubits_gaopt_exchange_matrix(self):
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
            "max_bond_dimension_list": [45],
            "max_time_limit_sec_list": [15 * 60],  # 10 min #[240],  # 4min
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
            "track_mem": [False],
            "reordering_method_list": ["gaopt, exchange matrix"],
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
            # "do_single_calc": False,
            "num_states": 1,
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
            return_value = os.system(
                f"env_dmrghandler/bin/python {python_run_file_name}"
            )
            if return_value != 0:
                raise Exception("DMRG failed to run properly")
            log.debug("DMRG NOW EXITED")
            # Get results
            main_storage_folder_path = data_config["main_storage_folder_path"]
            hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            with h5py.File(hdf5_file_path, "r") as f:
                dmrg_energies = f["/final_dmrg_results/past_energies_dmrg"][:]
                dmrg_bond_dimensions = f["/final_dmrg_results/bond_dims_used"][:]
                discarded_weights = f["/final_dmrg_results/past_discarded_weights"][:]

                self.check_csf_presence_and_threshold(config_dict, f)

            log.info(f"dmrg_energies: {dmrg_energies}")
            log.info(f"dmrg_bond_dimensions: {dmrg_bond_dimensions}")
            log.info(f"discarded_weights: {discarded_weights}")

            npt.assert_allclose(
                dmrg_energies[0], -5103.5559419269, rtol=test_rtol, atol=1e-3
            )

    def test_4a_48_qubits_gaopt_interaction_matrix(self):
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
            "max_time_limit_sec_list": [4 * 60],  # 10 min #[240],  # 4min
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
            "track_mem": [False],
            "reordering_method_list": ["gaopt, interaction matrix"],
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
            # "do_single_calc": False,
            "num_states": 1,
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
            return_value = os.system(
                f"env_dmrghandler/bin/python {python_run_file_name}"
            )
            if return_value != 0:
                raise Exception("DMRG failed to run properly")
            log.debug("DMRG NOW EXITED")
            # Get results
            main_storage_folder_path = data_config["main_storage_folder_path"]
            hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            with h5py.File(hdf5_file_path, "r") as f:
                dmrg_energies = f["/final_dmrg_results/past_energies_dmrg"][:]
                dmrg_bond_dimensions = f["/final_dmrg_results/bond_dims_used"][:]
                discarded_weights = f["/final_dmrg_results/past_discarded_weights"][:]

                self.check_csf_presence_and_threshold(config_dict, f)

            log.info(f"dmrg_energies: {dmrg_energies}")
            log.info(f"dmrg_bond_dimensions: {dmrg_bond_dimensions}")
            log.info(f"discarded_weights: {discarded_weights}")

            npt.assert_allclose(
                dmrg_energies[0], -5103.5559419269, rtol=test_rtol, atol=1e-3
            )

    def test_4a_48_qubits_gaopt_interaction_matrix_SU2(self):
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
            "max_time_limit_sec_list": [7 * 60],  # 10 min #[240],  # 4min
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
            "track_mem": [False],
            "reordering_method_list": ["gaopt, interaction matrix, SU(2) calc"],
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
            # "do_single_calc": False,
            "num_states": 1,
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
            return_value = os.system(
                f"env_dmrghandler/bin/python {python_run_file_name}"
            )
            if return_value != 0:
                raise Exception("DMRG failed to run properly")
            log.debug("DMRG NOW EXITED")
            # Get results
            main_storage_folder_path = data_config["main_storage_folder_path"]
            hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            with h5py.File(hdf5_file_path, "r") as f:
                dmrg_energies = f["/final_dmrg_results/past_energies_dmrg"][:]
                dmrg_bond_dimensions = f["/final_dmrg_results/bond_dims_used"][:]
                discarded_weights = f["/final_dmrg_results/past_discarded_weights"][:]

                self.check_csf_presence_and_threshold(config_dict, f)

            log.info(f"dmrg_energies: {dmrg_energies}")
            log.info(f"dmrg_bond_dimensions: {dmrg_bond_dimensions}")
            log.info(f"discarded_weights: {discarded_weights}")

            npt.assert_allclose(
                dmrg_energies[0], -5103.5559419269, rtol=test_rtol, atol=1e-3
            )

    def test_4a_48_qubits_reverse_schedule_restart(self):
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

        num_reverse_points = 5
        config_dict = {
            "plot_filename_prefix_list": [
                "4a_48qubit_test",
            ],
            "main_storage_folder_path_prefix": "./data_storage",
            "max_bond_dimension_list": [35],
            "max_time_limit_sec_list": [240],  # 4min
            "min_energy_change_hartree_list": [1e-4],
            "extrapolation_type_list": ["discard_weights"],
            "starting_bond_dimension_list": [5],
            "max_num_sweeps_list": [13],
            "energy_convergence_threshold_list": [1e-8],
            # "sweep_schedule_bond_dims_parameters": [
            #     [(2, 4), (1, 5)]
            # ],  # (division_factor, count),
            # # e.g. [(2, 4), (1, 5)] and init_bd of 3 ->[1, 1, 1, 1, 3, 3, 3, 3, 3]
            # "sweep_schedule_bond_dims_parameters": [
            #     [
            #         15,
            #         15,
            #         15,
            #         15,
            #         30,
            #         30,
            #         30,
            #         30,
            #         30,
            #         27,
            #         27,
            #         27,
            #         27,
            #         27,
            #         27,
            #         27,
            #         27,
            #         24,
            #         24,
            #         24,
            #         24,
            #         24,
            #         24,
            #         24,
            #         24,
            #         22,
            #         22,
            #         22,
            #         22,
            #         22,
            #         22,
            #         22,
            #         22,
            #         20,
            #         20,
            #         20,
            #         20,
            #         20,
            #         20,
            #         20,
            #         20,
            #         18,
            #         18,
            #         18,
            #         18,
            #         18,
            #         18,
            #         18,
            #         18,
            #     ]
            # ],
            "sweep_schedule_bond_dims_parameters": [
                [
                    15,
                    15,
                    15,
                    15,
                    30,
                    30,
                    30,
                    30,
                    30,
                    27,
                    27,
                    27,
                    27,
                    # 27,
                    # 27,
                    # 27,
                    # 27,
                    # 24,
                    # 24,
                    # 24,
                    # 24,
                    # 24,
                    # 24,
                    # 24,
                    # 24,
                    # 22,
                    # 22,
                    # 22,
                    # 22,
                    # 22,
                    # 22,
                    # 22,
                    # 22,
                    # 20,
                    # 20,
                    # 20,
                    # 20,
                    # 20,
                    # 20,
                    # 20,
                    # 20,
                    # 18,
                    # 18,
                    # 18,
                    # 18,
                    # 18,
                    # 18,
                    # 18,
                    # 18,
                ]
            ],
            "sweep_schedule_bond_dims_parameters": [
                config_io.fwd_reverse_schedule(
                    start_bd=30, num_points=num_reverse_points
                ),
            ],
            "sweep_schedule_noise_list": [
                config_io.fwd_reverse_schedule_noise(
                    noise_1=1e-4, noise_2=1e-5, num_points=num_reverse_points
                )
            ],
            "sweep_schedule_davidson_threshold_list": [
                config_io.fwd_reverse_schedule_threshold(
                    thresh=1e-10, num_points=num_reverse_points
                )
            ],
            "init_state_bond_dimension_division_factor_list": [2],
            "init_state_seed_list": [658724],
            "initial_mps_method_list": ["random"],
            "factor_half_convention_list": [True],
            # "symmetry_type_list": ["SZ", "SZ", "SU(2)", "SU(2)"],
            "symmetry_type_list": ["SU(2)"],
            "num_threads_list": [4],
            "n_mkl_threads_list": [1],
            "track_mem": [False],
            "restart_dir_list": [
                "/home/jtcantin/utoronto/dmrghandler/tests/restart_temp_0",
            ],
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
            "do_single_calc": True,
        }

        config_files_list, config_dict_single_file_list = config_io.gen_config_files(
            data_file_list=data_file_list,
            config_dict=config_dict,
            dmrg_advanced_config=dmrg_advanced_config,
            config_file_prefix="4a_48qubit_test_restart_",
        )
        print(f"config_files_list: {config_files_list}")
        # print(f"config_dict_single_file_list: {config_dict_single_file_list}")

        submit_dict = {
            "time_cap_string": "00-23:59:00",
            "job_name": "4a_48qubit_test_restart_",
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
            return_value = os.system(
                f"env_dmrghandler/bin/python {python_run_file_name}"
            )
            if return_value != 0:
                raise Exception("DMRG failed to run properly")
            log.debug("DMRG NOW EXITED")
            # # Get results
            # main_storage_folder_path = data_config["main_storage_folder_path"]
            # hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            # with h5py.File(hdf5_file_path, "r") as f:
            #     dmrg_energies = f["/final_dmrg_results/past_energies_dmrg"][:]
            #     dmrg_bond_dimensions = f["/final_dmrg_results/bond_dims_used"][:]
            #     discarded_weights = f["/final_dmrg_results/past_discarded_weights"][:]

            # log.info(f"dmrg_energies: {dmrg_energies}")
            # log.info(f"dmrg_bond_dimensions: {dmrg_bond_dimensions}")
            # log.info(f"discarded_weights: {discarded_weights}")

            # npt.assert_allclose(
            #     dmrg_energies[-1], -5103.5559419269, rtol=test_rtol, atol=1e-3
            # )

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

        num_reverse_points = 5
        config_dict = {
            "plot_filename_prefix_list": [
                "4a_48qubit_test",
            ],
            "main_storage_folder_path_prefix": "./data_storage",
            "max_bond_dimension_list": [35],
            "max_time_limit_sec_list": [240],  # 4min
            "min_energy_change_hartree_list": [1e-4],
            "extrapolation_type_list": ["discard_weights"],
            "starting_bond_dimension_list": [5],
            "max_num_sweeps_list": [num_reverse_points + 9 + 10],
            "energy_convergence_threshold_list": [1e-8],
            # "sweep_schedule_bond_dims_parameters": [
            #     [(2, 4), (1, 5)]
            # ],  # (division_factor, count),
            # # e.g. [(2, 4), (1, 5)] and init_bd of 3 ->[1, 1, 1, 1, 3, 3, 3, 3, 3]
            "sweep_schedule_bond_dims_parameters": [
                [
                    15,
                    15,
                    15,
                    15,
                    30,
                    30,
                    30,
                    30,
                    30,
                    27,
                    27,
                    27,
                    27,
                    27,
                    27,
                    27,
                    27,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    24,
                    22,
                    22,
                    22,
                    22,
                    22,
                    22,
                    22,
                    22,
                    20,
                    20,
                    20,
                    20,
                    20,
                    20,
                    20,
                    20,
                    18,
                    18,
                    18,
                    18,
                    18,
                    18,
                    18,
                    18,
                ]
            ],
            "sweep_schedule_bond_dims_parameters": [
                config_io.fwd_reverse_schedule(
                    start_bd=30, num_points=num_reverse_points
                ),
            ],
            "sweep_schedule_noise_list": [
                config_io.fwd_reverse_schedule_noise(
                    noise_1=1e-4, noise_2=1e-5, num_points=num_reverse_points
                )
            ],
            "sweep_schedule_davidson_threshold_list": [
                config_io.fwd_reverse_schedule_threshold(
                    thresh=1e-10, num_points=num_reverse_points
                )
            ],
            "init_state_bond_dimension_division_factor_list": [2],
            "init_state_seed_list": [658724],
            "initial_mps_method_list": ["restart"],
            "factor_half_convention_list": [True],
            # "symmetry_type_list": ["SZ", "SZ", "SU(2)", "SU(2)"],
            "symmetry_type_list": ["SU(2)"],
            "num_threads_list": [4],
            "n_mkl_threads_list": [1],
            "track_mem": [False],
            "restart_dir_list": [
                "/home/jtcantin/utoronto/dmrghandler/tests/restart_temp_0",
            ],
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
            "sweep_start": 13,  # Default is 0, where to start sweep
            "initial_sweep_direction": None,  # Default is None, True means forward sweep (left-to-right)
            "stack_mem": 10 * 1024 * 1024 * 1024,  # =170*1024*1024*1024
            "stack_mem_ratio": 0.9,
            "do_single_calc": True,
        }

        config_files_list, config_dict_single_file_list = config_io.gen_config_files(
            data_file_list=data_file_list,
            config_dict=config_dict,
            dmrg_advanced_config=dmrg_advanced_config,
            config_file_prefix="4a_48qubit_test_restart_",
        )
        print(f"config_files_list: {config_files_list}")
        # print(f"config_dict_single_file_list: {config_dict_single_file_list}")

        submit_dict = {
            "time_cap_string": "00-23:59:00",
            "job_name": "4a_48qubit_test_restart_",
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
            return_value = os.system(
                f"env_dmrghandler/bin/python {python_run_file_name}"
            )
            if return_value != 0:
                raise Exception("DMRG failed to run properly")
            log.debug("DMRG NOW EXITED")
            # Get results
            main_storage_folder_path = data_config["main_storage_folder_path"]
            hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            with h5py.File(hdf5_file_path, "r") as f:
                dmrg_energies = f["/final_dmrg_results/past_energies_dmrg"][:]
                dmrg_bond_dimensions = f["/final_dmrg_results/bond_dims_used"][:]
                discarded_weights = f["/final_dmrg_results/past_discarded_weights"][:]
                self.check_csf_presence_and_threshold(config_dict, f)

            log.info(f"dmrg_energies: {dmrg_energies}")
            log.info(f"dmrg_bond_dimensions: {dmrg_bond_dimensions}")
            log.info(f"discarded_weights: {discarded_weights}")

            npt.assert_allclose(
                dmrg_energies[-1], -5103.5559419269, rtol=test_rtol, atol=1e-3
            )

    def test_4a_48_qubits_keep_only_optimized_ket(self):
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
            "max_time_limit_sec_list": [4 * 60],  # 10 min #[240],  # 4min
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
            "track_mem": [False],
            "reordering_method_list": ["fiedler, interaction matrix"],
            "keep_initial_ket_bool_list": [False],
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
            # "do_single_calc": False,
            "num_states": 1,
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
            return_value = os.system(
                f"env_dmrghandler/bin/python {python_run_file_name}"
            )
            if return_value != 0:
                raise Exception("DMRG failed to run properly")
            log.debug("DMRG NOW EXITED")
            # Get results
            main_storage_folder_path = data_config["main_storage_folder_path"]
            hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            with h5py.File(hdf5_file_path, "r") as f:
                dmrg_energies = f["/final_dmrg_results/past_energies_dmrg"][:]
                dmrg_bond_dimensions = f["/final_dmrg_results/bond_dims_used"][:]
                discarded_weights = f["/final_dmrg_results/past_discarded_weights"][:]
                self.check_csf_presence_and_threshold(config_dict, f)

            log.info(f"dmrg_energies: {dmrg_energies}")
            log.info(f"dmrg_bond_dimensions: {dmrg_bond_dimensions}")
            log.info(f"discarded_weights: {discarded_weights}")

            # Assert that the initial ket was not kept, while the optimized ket was
            # Do this by checking the folder names in the data storage folder that is part of scratch sim
            # The optimized ket folder should be there, while the initial ket folder should not
            # The optimized ket folders end with "ket_optimized", while the initial ket folders end with "initial_ket"
            calc_uuid = data_config["folder_uuid"]
            data_storage_folder = (
                Path(scratch_sim_path)
                / Path("data_storage")
                / Path(calc_uuid)
                / Path("mps_storage")
            )
            # Check if the folder exists
            assert (
                data_storage_folder.exists()
            ), f"Folder {data_storage_folder} does not exist"

            # Get folders inside, using Pathlib
            data_storage_folder_contents = [
                x.name for x in data_storage_folder.iterdir() if x.is_dir()
            ]

            log.info(f"data_storage_folder_contents: {data_storage_folder_contents}")
            # Loop through folder names and check if they contain "ket_optimized" or "initial_ket"
            num_optimized_ket_folders = 0
            num_initial_ket_folders = 0
            for folder_name in data_storage_folder_contents:
                if "ket_optimized" in folder_name:
                    num_optimized_ket_folders += 1
                if "initial_ket" in folder_name:
                    num_initial_ket_folders += 1

            self.assertTrue(
                num_optimized_ket_folders == len(data_storage_folder_contents),
                f"num_optimized_ket_folders: {num_optimized_ket_folders}, num_initial_ket_folders: {num_initial_ket_folders}, num_folders: {len(data_storage_folder_contents)}",
            )
            self.assertTrue(
                num_initial_ket_folders == 0,
                f"num_optimized_ket_folders: {num_optimized_ket_folders}, num_initial_ket_folders: {num_initial_ket_folders}, num_folders: {len(data_storage_folder_contents)}",
            )
            # Assert that the final energy is close to the expected value

            npt.assert_allclose(
                dmrg_energies[0], -5103.5559419269, rtol=test_rtol, atol=1e-3
            )

    def test_4a_48_qubits_v_score(self):
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
            "max_time_limit_sec_list": [6 * 60],  # 10 min #[240],  # 4min
            "min_energy_change_hartree_list": [1e-4],
            "extrapolation_type_list": ["discard_weights"],
            "starting_bond_dimension_list": [9],
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
            "track_mem": [False],
            "reordering_method_list": ["fiedler, interaction matrix"],
            "calc_v_score_bool_list": [True],
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
            # "do_single_calc": False,
            "num_states": 1,
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
            return_value = os.system(
                f"env_dmrghandler/bin/python {python_run_file_name}"
            )
            if return_value != 0:
                raise Exception("DMRG failed to run properly")
            log.debug("DMRG NOW EXITED")
            # Get results
            main_storage_folder_path = data_config["main_storage_folder_path"]
            hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            with h5py.File(hdf5_file_path, "r") as f:
                dmrg_energies = f["/final_dmrg_results/past_energies_dmrg"][:]
                dmrg_bond_dimensions = f["/final_dmrg_results/bond_dims_used"][:]
                discarded_weights = f["/final_dmrg_results/past_discarded_weights"][:]
                self.check_csf_presence_and_threshold(config_dict, f)

                # Check if the v_score is present
                self.assertTrue(
                    "/first_preloop_calc/dmrg_results/v_score_hartree_fock" in f,
                    "v_score_hartree_fock not in hdf5 file",
                )

                h_min_e_optket_norm = float(
                    f["/first_preloop_calc/dmrg_results/h_min_e_optket_norm"][()]
                )
                variance = float(
                    f["/first_preloop_calc/dmrg_results/optket_variance"][()]
                )
                v_score_numerator = float(
                    f["/first_preloop_calc/dmrg_results/v_score_numerator"][()]
                )
                deviation_init_ket = float(
                    f["/first_preloop_calc/dmrg_results/deviation_init_ket"][()]
                )
                v_score_init_ket = float(
                    f["/first_preloop_calc/dmrg_results/v_score_init_ket"][()]
                )
                hf_energy = float(f["/first_preloop_calc/dmrg_results/hf_energy"][()])
                deviation_hf = float(
                    f["/first_preloop_calc/dmrg_results/deviation_hf"][()]
                )
                v_score_hartree_fock = float(
                    f["/first_preloop_calc/dmrg_results/v_score_hartree_fock"][()]
                )
                initial_ket_energy = float(
                    f["/first_preloop_calc/dmrg_results/initial_ket_energy"][()]
                )

            log.info(f"dmrg_energies: {dmrg_energies}")
            log.info(f"dmrg_bond_dimensions: {dmrg_bond_dimensions}")
            log.info(f"discarded_weights: {discarded_weights}")
            log.info(f"h_min_e_optket_norm: {h_min_e_optket_norm}")
            log.info(f"variance: {variance}")
            log.info(f"v_score_numerator: {v_score_numerator}")
            log.info(f"deviation_init_ket: {deviation_init_ket}")
            log.info(f"v_score_init_ket: {v_score_init_ket}")
            log.info(f"hf_energy: {hf_energy}")
            log.info(f"deviation_hf: {deviation_hf}")
            log.info(f"v_score_hartree_fock: {v_score_hartree_fock}")
            log.info(f"initial_ket_energy: {initial_ket_energy}")

            energy_estimated = dmrg_energies[0]

            npt.assert_allclose(
                dmrg_energies[0], -5103.5559419269, rtol=test_rtol, atol=1e-3
            )

            # Check internal consistency of v_score
            v_score_internal_init_ket = v_score_numerator / (deviation_init_ket) ** 2
            v_score_internal_hf = v_score_numerator / (deviation_hf) ** 2

            npt.assert_allclose(
                v_score_internal_init_ket, v_score_init_ket, rtol=test_rtol, atol=1e-8
            )
            npt.assert_allclose(
                v_score_internal_hf, v_score_hartree_fock, rtol=test_rtol, atol=1e-8
            )

            # Check that the v_score is non-negative
            self.assertTrue(
                v_score_hartree_fock >= 0,
                f"v_score_hartree_fock: {v_score_hartree_fock}",
            )
            self.assertTrue(
                v_score_init_ket >= 0, f"v_score_init_ket: {v_score_init_ket}"
            )

            # Check that hf energy is lower than initial ket energy and larger than optimized ket energy

            self.assertTrue(
                hf_energy < initial_ket_energy,
                f"hf_energy: {hf_energy},{type(hf_energy)}; initial_ket_energy: {initial_ket_energy},{type(initial_ket_energy)}",
            )
            self.assertTrue(
                hf_energy > energy_estimated,
                f"hf_energy: {hf_energy}, energy_estimated: {energy_estimated}",
            )
            self.assertTrue(
                energy_estimated < initial_ket_energy,
                f"energy_estimated: {energy_estimated}, initial_ket_energy: {initial_ket_energy}",
            )

            # Check that both deviations are negative
            self.assertTrue(
                deviation_hf < 0,
                f"deviation_hf: {deviation_hf}",
            )
            self.assertTrue(
                deviation_init_ket < 0,
                f"deviation_init_ket: {deviation_init_ket}",
            )

    def test_co2_v_score(self):
        # data_file = "fcidump.test"

        data_files_folder = Path("./tests/test_data")

        data_file_list_file = [
            "fcidump.test_2",  # co2, old Hami
            "fcidump.test_3",  # fcidump.32_2ru_III_3pl_noncan_0.2_new
            "fcidump.be_cc-pVDZ.cc2b3628-dc13-4a95-8765-37211a995068",  # be_cc-pVDZ.cc2b3628-dc13-4a95-8765-37211a995068
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
                "co2_test",
                "32_2ru_III_3pl_noncan_0.2_new_test",
                "be_cc-pVDZ.cc2b3628-dc13-4a95-8765-37211a995068",
            ],
            "main_storage_folder_path_prefix": "./data_storage",
            "max_bond_dimension_list": [30],
            "max_time_limit_sec_list": [4 * 60],  # 10 min #[240],  # 4min
            "min_energy_change_hartree_list": [1e-4],
            "extrapolation_type_list": ["discard_weights"],
            "starting_bond_dimension_list": [5, 10, 10],
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
            "track_mem": [False],
            "reordering_method_list": ["fiedler, interaction matrix"],
            "calc_v_score_bool_list": [True],
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
            # "do_single_calc": False,
            "num_states": 1,
        }

        config_files_list, config_dict_single_file_list = config_io.gen_config_files(
            data_file_list=data_file_list,
            config_dict=config_dict,
            dmrg_advanced_config=dmrg_advanced_config,
            config_file_prefix="co2_test_",
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
            return_value = os.system(
                f"env_dmrghandler/bin/python {python_run_file_name}"
            )
            if return_value != 0:
                raise Exception("DMRG failed to run properly")
            log.debug("DMRG NOW EXITED")
            # Get results
            main_storage_folder_path = data_config["main_storage_folder_path"]
            hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            with h5py.File(hdf5_file_path, "r") as f:
                dmrg_energies = f["/final_dmrg_results/past_energies_dmrg"][:]
                dmrg_bond_dimensions = f["/final_dmrg_results/bond_dims_used"][:]
                discarded_weights = f["/final_dmrg_results/past_discarded_weights"][:]
                self.check_csf_presence_and_threshold(config_dict, f)

                # Check if the v_score is present
                self.assertTrue(
                    "/first_preloop_calc/dmrg_results/v_score_hartree_fock" in f,
                    "v_score_hartree_fock not in hdf5 file",
                )

                h_min_e_optket_norm = float(
                    f["/first_preloop_calc/dmrg_results/h_min_e_optket_norm"][()]
                )
                variance = float(
                    f["/first_preloop_calc/dmrg_results/optket_variance"][()]
                )
                v_score_numerator = float(
                    f["/first_preloop_calc/dmrg_results/v_score_numerator"][()]
                )
                deviation_init_ket = float(
                    f["/first_preloop_calc/dmrg_results/deviation_init_ket"][()]
                )
                v_score_init_ket = float(
                    f["/first_preloop_calc/dmrg_results/v_score_init_ket"][()]
                )
                hf_energy = float(f["/first_preloop_calc/dmrg_results/hf_energy"][()])
                deviation_hf = float(
                    f["/first_preloop_calc/dmrg_results/deviation_hf"][()]
                )
                v_score_hartree_fock = float(
                    f["/first_preloop_calc/dmrg_results/v_score_hartree_fock"][()]
                )
                initial_ket_energy = float(
                    f["/first_preloop_calc/dmrg_results/initial_ket_energy"][()]
                )

            log.info(f"dmrg_energies: {dmrg_energies}")
            log.info(f"dmrg_bond_dimensions: {dmrg_bond_dimensions}")
            log.info(f"discarded_weights: {discarded_weights}")
            log.info(f"h_min_e_optket_norm: {h_min_e_optket_norm}")
            log.info(f"variance: {variance}")
            log.info(f"v_score_numerator: {v_score_numerator}")
            log.info(f"deviation_init_ket: {deviation_init_ket}")
            log.info(f"v_score_init_ket: {v_score_init_ket}")
            log.info(f"hf_energy: {hf_energy}")
            log.info(f"deviation_hf: {deviation_hf}")
            log.info(f"v_score_hartree_fock: {v_score_hartree_fock}")
            log.info(f"initial_ket_energy: {initial_ket_energy}")

            data_file_path = Path(data_config["data_file_path"])
            fci_data = pyscf.tools.fcidump.read(data_file_path)
            kernel = pyscf_wrappers.pyscf_fcidump_fci(fci_data)
            energy_estimated = dmrg_energies[0]

            log.info("----------------------------------------------------------------")
            log.info("------------------------FCI----------------------------")
            log.info("----------------------------------------------------------------")
            # log.info(f"kernel: {kernel}")
            log.info(f"FCI energy: {kernel[0]}")
            log.info(f"DMRG energy: {energy_estimated}")
            npt.assert_allclose(
                kernel[0], energy_estimated, rtol=test_rtol, atol=test_atol
            )
            # Ensure within 1 mHa of FCI energy
            self.assertTrue(
                np.abs(kernel[0] - energy_estimated) < 1e-3,
                f"FCI energy: {kernel[0]}, DMRG energy: {energy_estimated}",
            )

            # Check internal consistency of v_score
            v_score_internal_init_ket = v_score_numerator / (deviation_init_ket) ** 2
            v_score_internal_hf = v_score_numerator / (deviation_hf) ** 2

            npt.assert_allclose(
                v_score_internal_init_ket, v_score_init_ket, rtol=test_rtol, atol=1e-8
            )
            npt.assert_allclose(
                v_score_internal_hf, v_score_hartree_fock, rtol=test_rtol, atol=1e-8
            )

            # Check that the v_score is non-negative
            self.assertTrue(
                v_score_hartree_fock >= 0,
                f"v_score_hartree_fock: {v_score_hartree_fock}",
            )
            self.assertTrue(
                v_score_init_ket >= 0, f"v_score_init_ket: {v_score_init_ket}"
            )

            # Check that hf energy is lower than initial ket energy and larger than optimized ket energy

            self.assertTrue(
                hf_energy < initial_ket_energy,
                f"hf_energy: {hf_energy},{type(hf_energy)}; initial_ket_energy: {initial_ket_energy},{type(initial_ket_energy)}",
            )
            self.assertTrue(
                hf_energy > energy_estimated,
                f"hf_energy: {hf_energy}, energy_estimated: {energy_estimated}",
            )
            self.assertTrue(
                energy_estimated < initial_ket_energy,
                f"energy_estimated: {energy_estimated}, initial_ket_energy: {initial_ket_energy}",
            )
            # Check that both deviations are negative
            self.assertTrue(
                deviation_hf < 0,
                f"deviation_hf: {deviation_hf}",
            )
            self.assertTrue(
                deviation_init_ket < 0,
                f"deviation_init_ket: {deviation_init_ket}",
            )

    def test_4a_48_qubits_csf_coeff_data(self):
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
            "max_time_limit_sec_list": [4 * 60],  # 10 min #[240],  # 4min
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
            "track_mem": [False],
            "reordering_method_list": ["fiedler, interaction matrix"],
            "keep_initial_ket_bool_list": [False],
            "csf_coeff_threshold_list": [0.1],
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
            # "do_single_calc": False,
            "num_states": 1,
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
            return_value = os.system(
                f"env_dmrghandler/bin/python {python_run_file_name}"
            )
            if return_value != 0:
                raise Exception("DMRG failed to run properly")
            log.debug("DMRG NOW EXITED")
            # Get results
            main_storage_folder_path = data_config["main_storage_folder_path"]
            hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            with h5py.File(hdf5_file_path, "r") as f:
                dmrg_energies = f["/final_dmrg_results/past_energies_dmrg"][:]
                dmrg_bond_dimensions = f["/final_dmrg_results/bond_dims_used"][:]
                discarded_weights = f["/final_dmrg_results/past_discarded_weights"][:]

                self.check_csf_presence_and_threshold(config_dict, f)

            log.info(f"dmrg_energies: {dmrg_energies}")
            log.info(f"dmrg_bond_dimensions: {dmrg_bond_dimensions}")
            log.info(f"discarded_weights: {discarded_weights}")

            # Assert that the initial ket was not kept, while the optimized ket was
            # Do this by checking the folder names in the data storage folder that is part of scratch sim
            # The optimized ket folder should be there, while the initial ket folder should not
            # The optimized ket folders end with "ket_optimized", while the initial ket folders end with "initial_ket"
            calc_uuid = data_config["folder_uuid"]
            data_storage_folder = (
                Path(scratch_sim_path)
                / Path("data_storage")
                / Path(calc_uuid)
                / Path("mps_storage")
            )
            # Check if the folder exists
            assert (
                data_storage_folder.exists()
            ), f"Folder {data_storage_folder} does not exist"

            # Get folders inside, using Pathlib
            data_storage_folder_contents = [
                x.name for x in data_storage_folder.iterdir() if x.is_dir()
            ]

            log.info(f"data_storage_folder_contents: {data_storage_folder_contents}")
            # Loop through folder names and check if they contain "ket_optimized" or "initial_ket"
            num_optimized_ket_folders = 0
            num_initial_ket_folders = 0
            for folder_name in data_storage_folder_contents:
                if "ket_optimized" in folder_name:
                    num_optimized_ket_folders += 1
                if "initial_ket" in folder_name:
                    num_initial_ket_folders += 1

            self.assertTrue(
                num_optimized_ket_folders == len(data_storage_folder_contents),
                f"num_optimized_ket_folders: {num_optimized_ket_folders}, num_initial_ket_folders: {num_initial_ket_folders}, num_folders: {len(data_storage_folder_contents)}",
            )
            self.assertTrue(
                num_initial_ket_folders == 0,
                f"num_optimized_ket_folders: {num_optimized_ket_folders}, num_initial_ket_folders: {num_initial_ket_folders}, num_folders: {len(data_storage_folder_contents)}",
            )
            # Assert that the final energy is close to the expected value

            npt.assert_allclose(
                dmrg_energies[0], -5103.5559419269, rtol=test_rtol, atol=1e-3
            )

    def check_csf_presence_and_threshold(self, config_dict, hdf5_file):
        to_assert_presence_list = [
            "csf_coefficients_real_part",
            "csf_coefficients_imag_part",
            "csf_definitions",
            "csf_coeff_threshold",
            "largest_csf_coefficient_real_part",
            "largest_csf_coefficient_imag_part",
            "largest_csf",
        ]
        loops_list = []
        subfolder = "dmrg_results"
        # Add all keys that start with "dmrg_loop_" to the list
        for key in hdf5_file.keys():
            if key.startswith("dmrg_loop_"):
                loops_list.append(key)
            elif key == "first_preloop_calc":
                loops_list.append(key)
            elif key == "second_preloop_calc":
                loops_list.append(key)

        # Assert that the list of keys is not empty
        self.assertTrue(len(loops_list) > 0, "No loops found in hdf5 file")
        # Assert that the csf data is present
        for loop in loops_list:
            for key in to_assert_presence_list:
                self.assertTrue(
                    f"/{loop}/{subfolder}/{key}" in hdf5_file,
                    f"/{loop}/{subfolder}/{key} not in hdf5 file",
                )

                # Assert that csf_coeff_threshold is the same as the one in the config
                csf_coeff_threshold = float(
                    hdf5_file[f"/{loop}/{subfolder}/csf_coeff_threshold"][()]
                )

                dmrg_basic_config = config_dict["dmrg_basic_config"]
                if "csf_coeff_threshold" in dmrg_basic_config:

                    self.assertTrue(
                        np.allclose(
                            csf_coeff_threshold,
                            dmrg_basic_config["csf_coeff_threshold"],
                        ),
                        f"csf_coeff_threshold: {csf_coeff_threshold}, config_dict['csf_coeff_threshold'][0]: {dmrg_basic_config['csf_coeff_threshold']}",
                    )
                else:
                    self.assertTrue(
                        np.allclose(
                            csf_coeff_threshold,
                            qchem_dmrg_calc.CSF_COEFF_THRESHOLD_DEFAULT,
                        ),
                        f"csf_coeff_threshold: {csf_coeff_threshold}, CSF_COEFF_THRESHOLD_DEFAULT: {qchem_dmrg_calc.CSF_COEFF_THRESHOLD_DEFAULT}",
                    )

    def clean_up_files(self):
        # Folders to clean up:
        folders = [
            "tests/restart",
            "tests/restart_temp_0",
            "tests/scratch_sim",
            "tests/temp",
            "tests/temp2",
            "tmp_dir",
            "data_storage",
            "config_store",
            "i_moved_myself",
        ]

        # Files to clean up:
        files = [
            "dmrghandler_test.log",
            "dmrghandler.log",
        ]

        # Get PWD
        pwd = os.getcwd()
        if "/tests" in pwd:
            # If in tests folder, go up one level
            pwd = os.path.dirname(pwd)
        # Remove files, if they exist
        for file in files:
            file_path = Path(pwd) / Path(file)
            if file_path.exists():
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            else:
                print(f"File does not exist: {file_path}")

        # Remove folders, if they exist
        for folder in folders:
            folder_path = Path(pwd) / Path(folder)
            if folder_path.exists():
                shutil.rmtree(folder_path, ignore_errors=True)
                print(f"Removed folder: {folder_path}")
            else:
                print(f"Folder does not exist: {folder_path}")
