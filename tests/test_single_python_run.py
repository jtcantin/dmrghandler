import sys
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

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
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
            "data_for_testing/1.0_BeH2_cc-pvtz_c8ad7a19-f5e7-44aa-bf38-8db419cb1031.hdf5",
            "data_for_testing/1.0_CH2_cc-pvdz_6_444895b0-6b73-4273-b7d7-579908f09c30.hdf5",
            "data_for_testing/fcidump.2_co2_6-311++G__",
            "data_for_testing/fcidump.36_1ru_II_2pl_{'default' _ '6-31+G(d,p)', 'Ru' _ 'lanl2tz' }",
        ]
        config_dict = {
            "plot_filename_prefix_list": ["BeH2_", "CH2_", "CO2_", "Ru2_"],
            "main_storage_folder_path_prefix": "./data_storage",
            "max_bond_dimension_list": [100],
            "max_time_limit_sec_list": [20],
            "min_energy_change_hartree_list": [1e-4],
            "extrapolation_type_list": ["discard_weights"],
            "starting_bond_dimension_list": [3],
            "max_num_sweeps_list": [10],
            "energy_convergence_threshold_list": [1e-8],
            "sweep_schedule_bond_dims_parameters": [
                [(2, 4), (1, 5)]
            ],  # (division_factor, count),
            # e.g. [(2, 4), (1, 5)] and init_bd of 3 ->[[1, 1, 1, 1], [3, 3, 3, 3, 3]]
            "sweep_schedule_noise_list": [[[1e-4] * 4 + [1e-5] * 4 + [0]]],
            "sweep_schedule_davidson_threshold_list": [[[1e-10] * 9]],
            "init_state_bond_dimension_division_factor_list": [2],
            "init_state_seed_list": [64524, 21003, 98754, 88541],
            "initial_mps_method_list": ["random"],
            "factor_half_convention_list": [True],
            "symmetry_type_list": ["SZ", "SZ", "SU(2)", "SU(2)"],
            "num_threads_list": [1],
            "n_mkl_threads_list": [1],
        }

        config_files_list = config_io.gen_config_files(
            data_file_list=data_file_list,
            config_dict=config_dict,
            dmrg_advanced_config=None,
            config_file_prefix="tester_",
        )
        return config_files_list

    def test_single_python_runs(self):
        config_files_list = self.gen_files_prep()
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
