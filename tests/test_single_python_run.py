import faulthandler
import sys
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pyscf

import dmrghandler.config_io as config_io
import dmrghandler.dmrg_calc_prepare as dmrg_calc_prepare
import dmrghandler.dmrg_looping as dmrg_looping
import dmrghandler.dmrghandler
import dmrghandler.energy_extrapolation as energy_extrapolation
import dmrghandler.hdf5_io as hdf5_io
import dmrghandler.pyscf_wrappers as pyscf_wrappers
import dmrghandler.qchem_dmrg_calc as qchem_dmrg_calc

test_rtol = 1e-5
test_atol = 1e-5

import logging

# create logger
logger = logging.getLogger("dmrghandler")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler("dmrghandler.log")
fh.setLevel(logging.DEBUG)
# # create console handler with a higher log level
# ch = logging.StreamHandler()
# ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
)
fh.setFormatter(formatter)
# ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
# logger.addHandler(ch)

# To access the original stdout/stderr, use sys.__stdout__/sys.__stderr__
# By xjcl From https://stackoverflow.com/a/66209331
sys.stdout = dmrghandler.dmrghandler.LoggerWriter(logger.info)
sys.stderr = dmrghandler.dmrghandler.LoggerWriter(logger.error)

log = logger


class TestDmrgSinglePythonRun(unittest.TestCase):
    def gen_files_prep(self):
        data_file_list = [
            ## "data_for_testing/1.0_BeH2_cc-pvtz_c8ad7a19-f5e7-44aa-bf38-8db419cb1031.hdf5",
            ## "data_for_testing/1.0_CH2_cc-pvdz_6_444895b0-6b73-4273-b7d7-579908f09c30.hdf5",
            "data_for_testing/fcidump.2_co2_6-311++G__",  # Passes, gets FCI answer
            "data_for_testing/fcidump.36_1ru_II_2pl_{'default' _ '6-31+G(d,p)', 'Ru' _ 'lanl2tz' }",  # Passes, gets FCI answer
            "data_for_testing/fcidump.34_3ruo_IV_2pl_{'Ru' _ 'lanl2tz', 'default' _ '6-31+G(d,p)'}",  # Passes, gets FCI answer
            "data_for_testing/fcidump.36_1ru_II_2pl_{'default' _ '6-31+G(d,p)', 'Ru' _ 'lanl2tz' }",  # Passes, gets FCI answer
            # "data_for_testing/fcidump.32_2ru_III_3pl_{'default' _ '6-31+G(d,p)', 'Ru' _ 'lanl2tz' }",  # SIGFPE, (a segfault)
            # SIGFPE stack trace below; error appears in pyblock2/driver/core.py, line 2836 in dmrg:
            # Either it is pyblock2 that has the problem, or the problem is in the input data
            # """Fatal Python error: Floating point exception
            #     Current thread 0x00007fba606d0740 (most recent call first):
            #     File "/home/jtcantin/utoronto/dmrg_calc_handler_dev/dmrghandler/env_dmrghandler/lib/python3.9/site-packages/pyblock2/driver/core.py", line 2836 in dmrg
            #     File "/home/jtcantin/utoronto/dmrg_calc_handler_dev/dmrghandler/src/dmrghandler/qchem_dmrg_calc.py", line 269 in single_qchem_dmrg_calc
            #     File "/home/jtcantin/utoronto/dmrg_calc_handler_dev/dmrghandler/src/dmrghandler/dmrg_looping.py", line 62 in dmrg_central_loop
            #     File "/home/jtcantin/utoronto/dmrg_calc_handler_dev/dmrghandler/tests/test_single_python_run.py", line 130 in test_single_python_runs
            #     File "/home/jtcantin/.pyenv/versions/3.9.17/lib/python3.9/unittest/case.py", line 550 in _callTestMethod
            #     File "/home/jtcantin/.pyenv/versions/3.9.17/lib/python3.9/unittest/case.py", line 592 in run
            #     File "/home/jtcantin/.pyenv/versions/3.9.17/lib/python3.9/unittest/case.py", line 651 in __call__
            #     File "/home/jtcantin/.pyenv/versions/3.9.17/lib/python3.9/unittest/suite.py", line 122 in run
            #     File "/home/jtcantin/.pyenv/versions/3.9.17/lib/python3.9/unittest/suite.py", line 84 in __call__
            #     File "/home/jtcantin/.pyenv/versions/3.9.17/lib/python3.9/unittest/suite.py", line 122 in run
            #     File "/home/jtcantin/.pyenv/versions/3.9.17/lib/python3.9/unittest/suite.py", line 84 in __call__
            #     File "/home/jtcantin/.pyenv/versions/3.9.17/lib/python3.9/unittest/runner.py", line 184 in run
            #     File "/home/jtcantin/.vscode-server/extensions/ms-python.python-2024.2.1/pythonFiles/unittestadapter/execution.py", line 211 in run_tests
            #     File "/home/jtcantin/.vscode-server/extensions/ms-python.python-2024.2.1/pythonFiles/unittestadapter/execution.py", line 341 in <module>
            #     """,
        ]
        config_dict = {
            "plot_filename_prefix_list": [
                "2_co2",
                "36_1ru_II",
                # "32_2ru_III",
                "34_3ruo_IV",
                "36_1ru_II",
            ],
            "main_storage_folder_path_prefix": "./data_storage",
            "max_bond_dimension_list": [100, 100, 100, 100],
            "max_time_limit_sec_list": [20],
            "min_energy_change_hartree_list": [1e-4],
            "extrapolation_type_list": ["discard_weights"],
            "starting_bond_dimension_list": [3, 3, 10, 3],
            "max_num_sweeps_list": [10],
            "energy_convergence_threshold_list": [1e-8],
            "sweep_schedule_bond_dims_parameters": [
                [(2, 4), (1, 5)]
            ],  # (division_factor, count),
            # e.g. [(2, 4), (1, 5)] and init_bd of 3 ->[1, 1, 1, 1, 3, 3, 3, 3, 3]
            "sweep_schedule_noise_list": [[1e-4] * 4 + [1e-5] * 4 + [0]],
            "sweep_schedule_davidson_threshold_list": [[1e-10] * 9],
            "init_state_bond_dimension_division_factor_list": [2],
            "init_state_seed_list": [64524, 21003, 98754, 88541, 5614],
            "initial_mps_method_list": ["random"],
            "factor_half_convention_list": [True],
            # "symmetry_type_list": ["SZ", "SZ", "SU(2)", "SU(2)"],
            "symmetry_type_list": ["SU(2)"],
            "num_threads_list": [1],
            "n_mkl_threads_list": [1],
        }

        config_files_list, config_dict_single_file_list = config_io.gen_config_files(
            data_file_list=data_file_list,
            config_dict=config_dict,
            dmrg_advanced_config=None,
            config_file_prefix="tester_",
        )
        return config_files_list

    def test_single_python_runs(self):
        # file_object_faulter = open("segfault.log", "w")
        # faulthandler.enable(file=file_object_faulter)
        config_files_list = self.gen_files_prep()

        log.debug(f"config_files_list: {config_files_list}")

        for config_file in config_files_list:
            (
                one_body_tensor,
                two_body_tensor,
                dmrg_parameters,
                looping_parameters,
                data_config,
            ) = dmrg_calc_prepare.prepare_calc(config_file)

            if len(one_body_tensor) == 2:
                log.debug(f"one_body_tensor: {one_body_tensor[0].shape}")
                log.debug(f"one_body_tensor: {one_body_tensor[1].shape}")
                log.debug(f"two_body_tensor: {two_body_tensor[0].shape}")
                log.debug(f"two_body_tensor: {two_body_tensor[1].shape}")
                log.debug(f"two_body_tensor: {two_body_tensor[2].shape}")
            else:
                log.debug(f"one_body_tensor: {one_body_tensor.shape}")
                log.debug(f"two_body_tensor: {two_body_tensor.shape}")
            log.debug(f"dmrg_parameters: {dmrg_parameters}")
            log.debug(f"looping_parameters: {looping_parameters}")
            log.debug(f"data_config: {data_config}")

            max_bond_dimension = looping_parameters["max_bond_dimension"]
            max_time_limit_sec = looping_parameters["max_time_limit_sec"]
            min_energy_change_hartree = looping_parameters["min_energy_change_hartree"]
            main_storage_folder_path = data_config["main_storage_folder_path"]
            loop_results = dmrg_looping.dmrg_central_loop(
                one_body_tensor=one_body_tensor,
                two_body_tensor=two_body_tensor,
                dmrg_parameters=dmrg_parameters,
                max_bond_dimension=max_bond_dimension,
                max_time_limit_sec=max_time_limit_sec,
                min_energy_change_hartree=min_energy_change_hartree,
                main_storage_folder_path=main_storage_folder_path,
                verbosity=2,
                move_mps_to_final_storage_path="i_moved_myself",
            )

            finish_reason = loop_results["finish_reason"]
            energy_change = loop_results["energy_change"]
            discard_weight_change = loop_results["discard_weight_change"]
            bond_dims_used = loop_results["bond_dims_used"]
            past_energies_dmrg = loop_results["past_energies_dmrg"]
            past_discarded_weights = loop_results["past_discarded_weights"]
            loop_entry_count = loop_results["loop_entry_count"]
            unmodified_fit_parameters_list = loop_results[
                "unmodified_fit_parameters_list"
            ]
            fit_parameters_list = loop_results["fit_parameters_list"]

            # final_dmrg_results = loop_results["final_dmrg_results"]
            log.info(f"finish_reason: {finish_reason}")
            log.info(f"energy_change: {energy_change}")
            log.info(f"discard_weight_change: {discard_weight_change}")
            log.info(f"bond_dims_used: {bond_dims_used}")
            log.info(f"past_energies_dmrg: {past_energies_dmrg}")
            log.info(f"past_discarded_weights: {past_discarded_weights}")
            log.info(f"loop_entry_count: {loop_entry_count}")
            log.info(
                f"unmodified_fit_parameters_list: {unmodified_fit_parameters_list}"
            )
            log.info(f"fit_parameters_list: {fit_parameters_list}")

            # Get final extrapolated energy
            result_obj, energy_estimated, fit_parameters, R_squared = (
                energy_extrapolation.dmrg_energy_extrapolation(
                    energies_dmrg=past_energies_dmrg,
                    independent_vars=past_discarded_weights,
                    extrapolation_type="discarded_weight",
                    past_parameters=fit_parameters_list[-1],
                    verbosity=2,
                )
            )
            log.info(f"energy_estimated: {energy_estimated}")
            log.info(f"fit_parameters: {fit_parameters}")
            log.info(f"R_squared: {R_squared}")
            log.info(f"result_obj.message: {result_obj.message}")
            log.info(f"result_obj.cost: {result_obj.cost}")
            log.info(f"result_obj.fun: {result_obj.fun}")
            plot_filename_prefix = data_config["plot_filename_prefix"]
            energy_extrapolation.plot_extrapolation(
                discarded_weights=past_discarded_weights,
                energies_dmrg=past_energies_dmrg,
                fit_parameters=fit_parameters,
                bond_dims=bond_dims_used,
                plot_filename=main_storage_folder_path
                / Path("plots")
                / Path(f"{plot_filename_prefix}_energy_extrapolation"),
                figNum=0,
            )
            data_file_path = Path(data_config["data_file_path"])
            fci_data = pyscf.tools.fcidump.read(data_file_path)
            kernel = pyscf_wrappers.pyscf_fcidump_fci(fci_data)
            log.info("----------------------------------------------------------------")
            log.info("------------------------FCI----------------------------")
            log.info("----------------------------------------------------------------")
            # log.info(f"kernel: {kernel}")
            log.info(f"FCI energy: {kernel[0]}")
            log.info(f"DMRG energy: {energy_estimated}")
            self.assertTrue(
                np.allclose(kernel[0], energy_estimated, rtol=test_rtol, atol=test_atol)
            )

        # file_object_faulter.close()
