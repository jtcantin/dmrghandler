import sys
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt

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


class TestDmrgLoopingSmallMolecule(unittest.TestCase):
    def dmrg_loop_func_su2(
        self,
        molecule_name,
        basis,
        geometry,
        num_unpaired_electrons,
        charge,
        multiplicity,
        init_state_bond_dimension,
        max_num_sweeps,
        energy_convergence_threshold,
        sweep_schedule_bond_dims,
        sweep_schedule_noise,
        sweep_schedule_davidson_threshold,
        max_bond_dimension,
        max_time_limit_sec,
        min_energy_change_hartree,
    ):
        mol = pyscf_wrappers.get_pyscf_mol(
            basis=basis,
            geometry=geometry,
            num_unpaired_electrons=num_unpaired_electrons,
            charge=charge,
            multiplicity=multiplicity,
        )

        one_body_tensor, two_body_tensor, nuc_rep_energy = (
            pyscf_wrappers.get_pyscf_tensors(mol)
        )

        E_FCI_HF, E_FCI_UHF, E_FCI_orb = pyscf_wrappers.get_pyscf_fci_energy(mol)

        num_orbitals = mol.nao_nr()
        num_spin_orbitals = 2 * num_orbitals

        dmrg_parameters = {
            "factor_half_convention": True,
            "symmetry_type": "SU(2)",
            "num_threads": 1,
            "n_mkl_threads": 1,
            "num_orbitals": num_orbitals,
            "num_spin_orbitals": num_spin_orbitals,
            "num_electrons": int(mol.nelectron),
            "two_S": num_unpaired_electrons,
            "two_Sz": int((multiplicity - 1) / 2),
            "orb_sym": None,
            "temp_dir": "./tests/temp",
            "stack_mem": 1073741824,
            "restart_dir": "./tests/restart",
            # "mps_storage_folder": "./tests/temp/mps_storage",
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 63857,  # 9844,  # 0 means random seed
            "initial_mps_method": "random",
            "init_state_bond_dimension": init_state_bond_dimension,
            "occupancy_hint": None,
            "full_fci_space_bool": True,
            "init_state_direct_two_site_construction_bool": False,
            "max_num_sweeps": max_num_sweeps,
            "energy_convergence_threshold": energy_convergence_threshold,
            "sweep_schedule_bond_dims": sweep_schedule_bond_dims,
            "sweep_schedule_noise": sweep_schedule_noise,
            "sweep_schedule_davidson_threshold": sweep_schedule_davidson_threshold,
            "davidson_type": None,  # Default is None, for "Normal"
            "eigenvalue_cutoff": 1e-20,  # Cutoff of eigenvalues, default is 1e-20
            "davidson_max_iterations": 4000,  # Default is 4000
            "davidson_max_krylov_subspace_size": 50,  # Default is 50
            "lowmem_noise_bool": False,  # Whether to use a lower memory version of the noise, default is False
            "sweep_start": 0,  # Default is 0, where to start sweep
            "initial_sweep_direction": None,  # Default is None, True means forward sweep (left-to-right)
            "stack_mem_ratio": 0.6,
        }

        uuid_main_storage_file_path = hdf5_io.get_generic_filename()
        uuid_main_storage_folder_path = uuid_main_storage_file_path.stem
        uuid_main_storage_folder_path = (
            Path("tests")
            / Path("temp2")
            / Path(f"{molecule_name}")
            / uuid_main_storage_folder_path
        )

        loop_results = dmrg_looping.dmrg_central_loop(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            dmrg_parameters=dmrg_parameters,
            max_bond_dimension=max_bond_dimension,
            max_time_limit_sec=max_time_limit_sec,
            min_energy_change_hartree=min_energy_change_hartree,
            main_storage_folder_path=uuid_main_storage_folder_path,
            verbosity=2,
        )

        finish_reason = loop_results["finish_reason"]
        energy_change = loop_results["energy_change"]
        discard_weight_change = loop_results["discard_weight_change"]
        bond_dims_used = loop_results["bond_dims_used"]
        past_energies_dmrg = loop_results["past_energies_dmrg"]
        past_discarded_weights = loop_results["past_discarded_weights"]
        loop_entry_count = loop_results["loop_entry_count"]
        unmodified_fit_parameters_list = loop_results["unmodified_fit_parameters_list"]
        fit_parameters_list = loop_results["fit_parameters_list"]

        # final_dmrg_results = loop_results["final_dmrg_results"]
        log.info(f"finish_reason: {finish_reason}")
        log.info(f"energy_change: {energy_change}")
        log.info(f"discard_weight_change: {discard_weight_change}")
        log.info(f"bond_dims_used: {bond_dims_used}")
        log.info(f"past_energies_dmrg: {past_energies_dmrg}")
        log.info(f"past_discarded_weights: {past_discarded_weights}")
        log.info(f"loop_entry_count: {loop_entry_count}")
        log.info(f"unmodified_fit_parameters_list: {unmodified_fit_parameters_list}")
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

        energy_extrapolation.plot_extrapolation(
            discarded_weights=past_discarded_weights,
            energies_dmrg=past_energies_dmrg,
            fit_parameters=fit_parameters,
            bond_dims=bond_dims_used,
            plot_filename=uuid_main_storage_folder_path
            / Path("plots")
            / Path(f"energy_extrapolation_{molecule_name}"),
            figNum=0,
        )

        npt.assert_allclose(
            loop_results["energy_estimated"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            loop_results["energy_estimated"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            loop_results["energy_estimated"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    def test_h2_sto6g_fci_su2(self):

        # Molecule Info
        molecule_name = "H2,sto6g"
        basis = "sto6g"  # 4 spin orbitals
        bond_length = 1.0
        geometry = f"H 0 0 0; H 0 0 {bond_length}"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1

        # DMRG info
        max_bond_dimension = 100
        max_time_limit_sec = 20
        min_energy_change_hartree = test_atol
        init_state_bond_dimension = 2
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = [init_state_bond_dimension // 2] * 4 + [
            init_state_bond_dimension
        ] * 4
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

        self.dmrg_loop_func_su2(
            molecule_name=molecule_name,
            basis=basis,
            geometry=geometry,
            num_unpaired_electrons=num_unpaired_electrons,
            charge=charge,
            multiplicity=multiplicity,
            init_state_bond_dimension=init_state_bond_dimension,
            max_num_sweeps=max_num_sweeps,
            energy_convergence_threshold=energy_convergence_threshold,
            sweep_schedule_bond_dims=sweep_schedule_bond_dims,
            sweep_schedule_noise=sweep_schedule_noise,
            sweep_schedule_davidson_threshold=sweep_schedule_davidson_threshold,
            max_bond_dimension=max_bond_dimension,
            max_time_limit_sec=max_time_limit_sec,
            min_energy_change_hartree=min_energy_change_hartree,
        )

    def test_beh2_sto3g_fci_su2(self):

        # Molecule Info
        molecule_name = "BeH2,sto3g"
        basis = "sto3g"  # 4 spin orbitals
        bond_length = 1.0
        geometry = f"Be 0 0 0; H 0 0 {bond_length}; H 0 0 -{bond_length}"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1

        # DMRG info
        max_bond_dimension = 100
        max_time_limit_sec = 20
        min_energy_change_hartree = test_atol
        init_state_bond_dimension = 2
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = [init_state_bond_dimension // 2] * 4 + [
            init_state_bond_dimension
        ] * 4
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

        self.dmrg_loop_func_su2(
            molecule_name=molecule_name,
            basis=basis,
            geometry=geometry,
            num_unpaired_electrons=num_unpaired_electrons,
            charge=charge,
            multiplicity=multiplicity,
            init_state_bond_dimension=init_state_bond_dimension,
            max_num_sweeps=max_num_sweeps,
            energy_convergence_threshold=energy_convergence_threshold,
            sweep_schedule_bond_dims=sweep_schedule_bond_dims,
            sweep_schedule_noise=sweep_schedule_noise,
            sweep_schedule_davidson_threshold=sweep_schedule_davidson_threshold,
            max_bond_dimension=max_bond_dimension,
            max_time_limit_sec=max_time_limit_sec,
            min_energy_change_hartree=min_energy_change_hartree,
        )

    def test_hneg_aug_cc_pVDZ_fci_su2(self):

        # Molecule Info
        molecule_name = "H-,aug-cc-pVDZ"
        basis = "aug-cc-pVDZ"  # 4 spin orbitals
        # bond_length = 1.0
        geometry = f"H 0 0 0"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = -1
        multiplicity = 1

        # DMRG info
        max_bond_dimension = 100
        max_time_limit_sec = 20
        min_energy_change_hartree = test_atol
        init_state_bond_dimension = 2
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = [init_state_bond_dimension // 2] * 4 + [
            init_state_bond_dimension
        ] * 4
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

        self.dmrg_loop_func_su2(
            molecule_name=molecule_name,
            basis=basis,
            geometry=geometry,
            num_unpaired_electrons=num_unpaired_electrons,
            charge=charge,
            multiplicity=multiplicity,
            init_state_bond_dimension=init_state_bond_dimension,
            max_num_sweeps=max_num_sweeps,
            energy_convergence_threshold=energy_convergence_threshold,
            sweep_schedule_bond_dims=sweep_schedule_bond_dims,
            sweep_schedule_noise=sweep_schedule_noise,
            sweep_schedule_davidson_threshold=sweep_schedule_davidson_threshold,
            max_bond_dimension=max_bond_dimension,
            max_time_limit_sec=max_time_limit_sec,
            min_energy_change_hartree=min_energy_change_hartree,
        )
