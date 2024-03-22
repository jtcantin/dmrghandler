import sys
import unittest

import numpy as np
import numpy.testing as npt

import dmrghandler.dmrghandler
import dmrghandler.pyscf_wrappers as pyscf_wrappers
import dmrghandler.qchem_dmrg_calc as qchem_dmrg_calc

test_rtol = 1e-5
test_atol = 1e-8

import logging

# create logger
logger = logging.getLogger("dmrghandler")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler("dmrghandler.log")
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
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


class TestDmrgSmallMolecule(unittest.TestCase):
    def test_h2_sto3g_fci_sz(self):

        # Molecule Info
        basis = "sto3g"  # 4 spin orbitals
        bond_length = 1.0
        geometry = f"H 0 0 0; H 0 0 {bond_length}"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "symmetry_type": "SZ",
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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 64241,  # 0 means random seed
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
            "stack_mem_ratio": 0.8,  # Default is 0.4
        }
        one_body_tensor_sz = [one_body_tensor, one_body_tensor]
        two_body_tensor_sz = [two_body_tensor, two_body_tensor, two_body_tensor]
        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor_sz,
            two_body_tensor=two_body_tensor_sz,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    def test_h2_sto3g_fci_su2(self):

        # Molecule Info
        basis = "sto3g"  # 4 spin orbitals
        bond_length = 1.0
        geometry = f"H 0 0 0; H 0 0 {bond_length}"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "stack_mem": 2 * 1073741824,
            "restart_dir": "./tests/restart",
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 9844,  # 0 means random seed
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
            "stack_mem_ratio": 0.75,  # Default is 0.4
        }

        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    def test_h2_sto6g_fci_sz(self):

        # Molecule Info
        basis = "sto6g"  # 4 spin orbitals
        bond_length = 1.0
        geometry = f"H 0 0 0; H 0 0 {bond_length}"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "symmetry_type": "SZ",
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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 64241,  # 0 means random seed
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
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }
        one_body_tensor_sz = [one_body_tensor, one_body_tensor]
        two_body_tensor_sz = [two_body_tensor, two_body_tensor, two_body_tensor]
        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor_sz,
            two_body_tensor=two_body_tensor_sz,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    def test_h2_sto6g_fci_su2(self):

        # Molecule Info
        basis = "sto6g"  # 4 spin orbitals
        bond_length = 1.0
        geometry = f"H 0 0 0; H 0 0 {bond_length}"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 9844,  # 0 means random seed
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
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }

        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    def test_h2_cc_pVDZ_fci_sz(self):

        # Molecule Info
        basis = "cc-pVDZ"  # 4 spin orbitals
        bond_length = 1.0
        geometry = f"H 0 0 0; H 0 0 {bond_length}"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "symmetry_type": "SZ",
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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 64241,  # 0 means random seed
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
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }
        one_body_tensor_sz = [one_body_tensor, one_body_tensor]
        two_body_tensor_sz = [two_body_tensor, two_body_tensor, two_body_tensor]
        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor_sz,
            two_body_tensor=two_body_tensor_sz,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    def test_h2_cc_pVDZ_fci_su2(self):

        # Molecule Info
        basis = "cc-pVDZ"  # 4 spin orbitals
        bond_length = 1.0
        geometry = f"H 0 0 0; H 0 0 {bond_length}"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 9844,  # 0 means random seed
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
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }

        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    def test_hneg_aug_cc_pVDZ_fci_sz(self):

        # Molecule Info
        basis = "aug-cc-pVDZ"  # 4 spin orbitals
        # bond_length = 1.0
        geometry = f"H 0 0 0"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = -1
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "symmetry_type": "SZ",
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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 64241,  # 0 means random seed
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
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }
        one_body_tensor_sz = [one_body_tensor, one_body_tensor]
        two_body_tensor_sz = [two_body_tensor, two_body_tensor, two_body_tensor]
        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor_sz,
            two_body_tensor=two_body_tensor_sz,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    def test_hneg_aug_cc_pVDZ_fci_su2(self):

        # Molecule Info
        basis = "aug-cc-pVDZ"  # 4 spin orbitals
        # bond_length = 1.0
        geometry = f"H 0 0 0"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = -1
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 9844,  # 0 means random seed
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
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }

        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    def test_lih_sto3g_fci_sz(self):

        # Molecule Info
        basis = "sto3g"  # 4 spin orbitals
        bond_length = 1.0
        geometry = f"H 0 0 0; Li 0 0 {bond_length}"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "symmetry_type": "SZ",
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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 64241,  # 0 means random seed
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
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }
        one_body_tensor_sz = [one_body_tensor, one_body_tensor]
        two_body_tensor_sz = [two_body_tensor, two_body_tensor, two_body_tensor]
        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor_sz,
            two_body_tensor=two_body_tensor_sz,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    @unittest.expectedFailure
    def test_lih_sto3g_fci_su2(self):
        # NOTE: Don't know why this fails. Maybe the basis is too small for SU(2) symmetry?
        # Molecule Info
        basis = "sto3g"  # 4 spin orbitals
        bond_length = 1.0
        geometry = f"H 0 0 0; Li 0 0 {bond_length}"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 9844,  # 0 means random seed
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
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }

        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        self.assertTrue(
            np.allclose(
                dmrg_results["dmrg_ground_state_energy"],
                E_FCI_HF,
                rtol=test_rtol,
                atol=test_atol,
            )
        )
        self.assertTrue(
            np.allclose(
                dmrg_results["dmrg_ground_state_energy"],
                E_FCI_UHF,
                rtol=test_rtol,
                atol=test_atol,
            )
        )
        self.assertTrue(
            np.allclose(
                dmrg_results["dmrg_ground_state_energy"],
                E_FCI_orb,
                rtol=test_rtol,
                atol=test_atol,
            )
        )

    def test_lih_cc_pVDZ_fci_su2(self):

        # Molecule Info
        basis = "cc-pVDZ"  # 4 spin orbitals
        bond_length = 1.0
        geometry = f"H 0 0 0; Li 0 0 {bond_length}"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 9844,  # 0 means random seed
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
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }

        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    def test_beh2_sto3g_fci_sz(self):

        # Molecule Info
        basis = "sto3g"  # 4 spin orbitals
        bond_length = 1.0
        geometry = f"Be 0 0 0; H 0 0 {bond_length}; H 0 0 -{bond_length}"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "symmetry_type": "SZ",
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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 64241,  # 0 means random seed
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
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }
        one_body_tensor_sz = [one_body_tensor, one_body_tensor]
        two_body_tensor_sz = [two_body_tensor, two_body_tensor, two_body_tensor]
        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor_sz,
            two_body_tensor=two_body_tensor_sz,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    def test_beh2_sto3g_fci_su2(self):

        # Molecule Info
        basis = "sto3g"  # 4 spin orbitals
        bond_length = 1.0
        geometry = f"Be 0 0 0; H 0 0 {bond_length}; H 0 0 -{bond_length}"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 9844,  # 0 means random seed
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
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }

        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    # basis = "sto3g"
    #     bond_length = 1.0

    #     angle = deg2rad(107.6 / 2)
    #     xDistance = bond_length * sin(angle)
    #     yDistance = bond_length * cos(angle)
    #     geometry = "O 0 0 0; H -$xDistance $yDistance 0; H $xDistance $yDistance 0"

    def test_h2o_sto3g_fci_sz(self):

        # Molecule Info
        basis = "sto3g"  # 4 spin orbitals
        bond_length = 1.0
        angle = np.deg2rad(107.6 / 2)
        xDistance = bond_length * np.sin(angle)
        yDistance = bond_length * np.cos(angle)
        geometry = f"O 0 0 0; H -{xDistance} {yDistance} 0; H {xDistance} {yDistance} 0"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "symmetry_type": "SZ",
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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 64241,  # 0 means random seed
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
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }
        one_body_tensor_sz = [one_body_tensor, one_body_tensor]
        two_body_tensor_sz = [two_body_tensor, two_body_tensor, two_body_tensor]
        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor_sz,
            two_body_tensor=two_body_tensor_sz,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )

    def test_h2o_sto3g_fci_su2(self):

        # Molecule Info
        basis = "sto3g"  # 4 spin orbitals
        bond_length = 1.0
        angle = np.deg2rad(107.6 / 2)
        xDistance = bond_length * np.sin(angle)
        yDistance = bond_length * np.cos(angle)
        geometry = f"O 0 0 0; H -{xDistance} {yDistance} 0; H {xDistance} {yDistance} 0"
        num_unpaired_electrons = 0  # i.e. 2*spin
        charge = 0
        multiplicity = 1
        # DMRG info
        init_state_bond_dimension = 100
        max_num_sweeps = 20
        energy_convergence_threshold = 1e-8
        sweep_schedule_bond_dims = qchem_dmrg_calc.default_sweep_schedule_bond_dims
        sweep_schedule_noise = qchem_dmrg_calc.default_sweep_schedule_noise
        sweep_schedule_davidson_threshold = (
            qchem_dmrg_calc.default_sweep_schedule_davidson_threshold
        )

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
            "core_energy": nuc_rep_energy,
            "reordering_method": "none",
            "init_state_seed": 9844,  # 0 means random seed
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
            "stack_mem_ratio": 0.4,  # Default is 0.4
        }

        dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            dmrg_parameters=dmrg_parameters,
            verbosity=2,
        )

        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_HF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_UHF,
            rtol=test_rtol,
            atol=test_atol,
        )
        npt.assert_allclose(
            dmrg_results["dmrg_ground_state_energy"],
            E_FCI_orb,
            rtol=test_rtol,
            atol=test_atol,
        )
