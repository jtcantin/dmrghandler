import logging
import os
import unittest
from pathlib import Path

import numpy as np
import numpy.testing as npt
import scipy as sp
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

import dmrghandler.config_io as config_io
import dmrghandler.dmrg_calc_prepare as dmrg_calc_prepare
import dmrghandler.fcidump_io as fcidump_io
import dmrghandler.hdf5_io as hdf5_io
import dmrghandler.model_hamiltonians as model_hamiltonians
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
fh = logging.FileHandler("model_hamis.log")
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - Line %(lineno)d - %(message)s"
)
fh.setFormatter(formatter)
# add the handlers to the log
log.addHandler(fh)


class TestTightBinding(unittest.TestCase):
    def prepare_dmrg_files(self, data_files_folder, data_file_list_file):

        data_file_list = []
        for data_file in data_file_list_file:
            data_file_path = Path(data_files_folder) / Path(data_file)
            data_file_list.append(str(data_file_path))
        log.info(f"data_file_list: {data_file_list}")

        config_dict = {
            "plot_filename_prefix_list": [
                "bands",
            ],
            "main_storage_folder_path_prefix": "./tests/data_storage",
            "max_bond_dimension_list": [3000],
            "max_time_limit_sec_list": [60],  # 23 hrs
            "min_energy_change_hartree_list": [1e-4],
            "extrapolation_type_list": ["discard_weights"],
            "starting_bond_dimension_list": [1],
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
            "num_threads_list": [40],
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
            "stack_mem": 10737418240,  # =10*1024*1024*1024
            "stack_mem_ratio": 0.9,
        }

        config_files_list, config_dict_single_file_list = config_io.gen_config_files(
            data_file_list=data_file_list,
            config_dict=config_dict,
            dmrg_advanced_config=dmrg_advanced_config,
            config_file_prefix="config_band_hamis_",
        )
        log.info(f"config_files_list: {config_files_list}")
        # print(f"config_dict_single_file_list: {config_dict_single_file_list}")

        submit_dict = {
            "time_cap_string": "00-23:59:00",
            "job_name": "NA",
            "email": "joshua.cantin@utoronto.ca",
            "account_name": "BLANK",
            "tasks_per_node": "1",
            "cpus_per_task": "40",
            "partition": "NA",
            "python_environment_location": "NA",
        }

        slurm_scripts.gen_run_files(submit_dict, config_dict_single_file_list)

        return config_files_list, config_dict_single_file_list

    def test_banded_tight_binding_model(self):
        dict_list = [
            {
                "num_orbitals": 3,
                "orbital_energies": [-1.0, 0.0, 1.0],
                "hopping_amplitude": 1.0,
                "bandwidth": 1,
                "num_electrons": 2,
                "two_S": 0,
                "orb_sym": np.ones(3, dtype=int),
                "answer": np.array(
                    [
                        [-1.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0],
                    ]
                ),
            },
            {
                "num_orbitals": 3,
                "orbital_energies": [-1.0, 0.0, 1.0],
                "hopping_amplitude": 1.0,
                "bandwidth": 2,
                "num_electrons": 2,
                "two_S": 0,
                "orb_sym": np.ones(3, dtype=int),
                "answer": np.array(
                    [
                        [-1.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ]
                ),
            },
            {
                "num_orbitals": 5,
                "orbital_energies": [-1.0, 0.0, 1.0, 534, 9.123],
                "hopping_amplitude": 1.0,
                "bandwidth": 3,
                "num_electrons": 3,
                "two_S": 1,
                "orb_sym": np.ones(5, dtype=int),
                "answer": np.array(
                    [
                        [-1.0, 1.0, 1.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 534, 1.0],
                        [0.0, 1.0, 1.0, 1.0, 9.123],
                    ]
                ),
            },
        ]

        for dict in dict_list:
            num_orbitals = dict["num_orbitals"]
            orbital_energies = dict["orbital_energies"]
            hopping_amplitude = dict["hopping_amplitude"]
            bandwidth = dict["bandwidth"]
            num_electrons = dict["num_electrons"]
            two_S = dict["two_S"]
            orb_sym = dict["orb_sym"]

            one_body_tensor = model_hamiltonians.banded_tight_binding_model(
                num_orbitals, orbital_energies, hopping_amplitude, bandwidth
            )

            npt.assert_allclose(
                one_body_tensor, dict["answer"], rtol=test_rtol, atol=test_atol
            )

            # Save the one_body_tensor to a file
            temp_fcidump_path = Path("tests/model_hami/fcidump.one_body_tensor.fcidump")
            temp_fcidump_path.parent.mkdir(parents=True, exist_ok=True)

            two_body_tensor_sparse = None
            one_body_tensor_sparse = sp.sparse.lil_matrix(one_body_tensor)

            output_dict = {
                # Variable: (Description, Default Value)
                "NORB": num_orbitals,
                "NELEC": num_electrons,
                "MS2": two_S,
                "ISYM": 1,
                "ORBSYM": orb_sym,
                # "IPRTIM": ("If 0, print additional CPU timing analysis", -1),
                # "INT": ("Fortran stream from which integrals will be read", 5),
                # "MEMORY": ("Size of workspace array in floating point words", 100000),
                # "CORE": nuc_rep_energy,
                # "MAXIT": ("Maximum number of iterations in Davidson diagonalisation", 25),
                # "THR": (
                #     "Convergence threshold for Davidson diagonalisation (floating point)",
                #     1e-5,
                # ),
                # "THRRES": ("Threshold for printing final CI coefficients (floating point)", 1e-1),
                # "NROOT": ("Number of eigenvalues of Hamiltonian to be found", 1),
            }

            fcidump_io.write_fcidump(
                output_dict=output_dict,
                one_electron_integrals=one_body_tensor_sparse,
                two_electron_integrals=two_body_tensor_sparse,
                core_energy=0.0,
                file_path=temp_fcidump_path,
                real_bool=False,
                verbose=True,
            )

            # Load the one_body_tensor from the file
            (
                one_body_tensor_2,
                two_body_tensor_2,
                nuc_rep_energy_2,
                num_orbitals_2,
                num_spin_orbitals_2,
                num_electrons_2,
                two_S_2,
                two_Sz_2,
                orb_sym_2,
                extra_attributes_2,
            ) = dmrg_calc_prepare.load_tensors_from_fcidump(temp_fcidump_path)

            npt.assert_allclose(
                one_body_tensor, one_body_tensor_2, rtol=test_rtol, atol=test_atol
            )
            npt.assert_allclose(
                np.zeros_like(two_body_tensor_2),
                two_body_tensor_2,
                rtol=test_rtol,
                atol=test_atol,
            )

            # Get the ground state energy directly from the one_body_tensor
            (
                ground_state_energy,
                eigenvalues,
                eigenvectors,
            ) = model_hamiltonians.get_one_body_term_ground_state(
                one_body_tensor, num_electrons
            )

            # Get the ground state energy from DMRG
            data_files_folder = temp_fcidump_path.parent

            data_file_list_file = [
                temp_fcidump_path.name,
            ]

            config_files_list, config_dict_single_file_list = self.prepare_dmrg_files(
                data_files_folder, data_file_list_file
            )

            for config_dict in config_dict_single_file_list:
                data_config = config_dict["data_config"]
                python_run_file_name = data_config["python_run_file"]
                main_storage_folder_path = data_config["main_storage_folder_path"]
                log.info(f"main_storage_folder_path: {main_storage_folder_path}")
                log.info(f"python_run_file_name: {python_run_file_name}")
                # log.info(open(python_run_file_name).readlines())
                # exec(open(python_run_file_name).read())
                # os.system("which python")
                # os.system("pwd")
                # os.system(f"env_dmrghandler/bin/python {python_run_file_name}")
                data_file_path = data_config["data_file_path"]
                driver = DMRGDriver(
                    scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=1
                )
                driver.read_fcidump(filename=data_file_path, pg="c1")
                driver.initialize_system(
                    n_sites=driver.n_sites,
                    n_elec=driver.n_elec,
                    spin=driver.spin,
                    orb_sym=driver.orb_sym,
                )
                mpo = driver.get_qc_mpo(
                    h1e=driver.h1e, g2e=driver.g2e, ecore=driver.ecore, iprint=1
                )

                n_sweeps = 30
                bond_dims = [25] * 4 + [50]
                noises = [1e-4] * 4 + [1e-5] * 4 + [0]
                thrds = [1e-8] * n_sweeps

                ket = driver.get_random_mps(tag="KET", bond_dim=50, nroots=1)

                DMRG_energy = driver.dmrg(
                    mpo,
                    ket,
                    n_sweeps=n_sweeps,
                    bond_dims=bond_dims,
                    noises=noises,
                    thrds=thrds,
                    iprint=1,
                    twosite_to_onesite=20,
                )

                # log.info(driver.get_dmrg_results())
                log.info(f"DMRG energy: {DMRG_energy}")
                log.info(f"Ground state energy: {ground_state_energy}")
                log.info(f"DMRG Model Spin: {driver.spin}")

                npt.assert_allclose(
                    DMRG_energy, ground_state_energy, rtol=test_rtol, atol=test_atol
                )

    def test_banded_tight_binding_model_large(self):
        num_orbitals = 20  # 60
        final_bond_dim = 7
        dict_list = [
            {
                "num_orbitals": num_orbitals,
                # "orbital_energies": -0.01 * np.array(range(num_orbitals)),
                "orbital_energies": np.zeros(num_orbitals),
                "hopping_amplitude": 1.0,
                "bandwidth": 1,
                "num_electrons": 1,
                "two_S": 1,
                "orb_sym": np.ones(num_orbitals, dtype=int),
            },
            # {
            #     "num_orbitals": num_orbitals,
            #     "orbital_energies": np.ones(num_orbitals),
            #     "hopping_amplitude": 1.0,
            #     "bandwidth": num_orbitals - 1,
            #     "num_electrons": num_orbitals,
            #     "two_S": 0,
            #     "orb_sym": np.ones(num_orbitals, dtype=int),
            # },
        ]

        for dict in dict_list:
            num_orbitals = dict["num_orbitals"]
            orbital_energies = dict["orbital_energies"]
            hopping_amplitude = dict["hopping_amplitude"]
            bandwidth = dict["bandwidth"]
            num_electrons = dict["num_electrons"]
            two_S = dict["two_S"]
            orb_sym = dict["orb_sym"]

            one_body_tensor = model_hamiltonians.banded_tight_binding_model(
                num_orbitals, orbital_energies, hopping_amplitude, bandwidth
            )

            # npt.assert_allclose(
            #     one_body_tensor, dict["answer"], rtol=test_rtol, atol=test_atol
            # )

            # Save the one_body_tensor to a file
            temp_fcidump_path = Path("tests/model_hami/fcidump.one_body_tensor.fcidump")
            temp_fcidump_path.parent.mkdir(parents=True, exist_ok=True)

            two_body_tensor_sparse = None
            one_body_tensor_sparse = sp.sparse.lil_matrix(one_body_tensor)

            output_dict = {
                # Variable: (Description, Default Value)
                "NORB": num_orbitals,
                "NELEC": num_electrons,
                "MS2": two_S,
                "ISYM": 1,
                "ORBSYM": orb_sym,
                # "IPRTIM": ("If 0, print additional CPU timing analysis", -1),
                # "INT": ("Fortran stream from which integrals will be read", 5),
                # "MEMORY": ("Size of workspace array in floating point words", 100000),
                # "CORE": nuc_rep_energy,
                # "MAXIT": ("Maximum number of iterations in Davidson diagonalisation", 25),
                # "THR": (
                #     "Convergence threshold for Davidson diagonalisation (floating point)",
                #     1e-5,
                # ),
                # "THRRES": ("Threshold for printing final CI coefficients (floating point)", 1e-1),
                # "NROOT": ("Number of eigenvalues of Hamiltonian to be found", 1),
            }

            fcidump_io.write_fcidump(
                output_dict=output_dict,
                one_electron_integrals=one_body_tensor_sparse,
                two_electron_integrals=two_body_tensor_sparse,
                core_energy=0.0,
                file_path=temp_fcidump_path,
                real_bool=False,
                verbose=True,
            )

            # Load the one_body_tensor from the file
            (
                one_body_tensor_2,
                two_body_tensor_2,
                nuc_rep_energy_2,
                num_orbitals_2,
                num_spin_orbitals_2,
                num_electrons_2,
                two_S_2,
                two_Sz_2,
                orb_sym_2,
                extra_attributes_2,
            ) = dmrg_calc_prepare.load_tensors_from_fcidump(temp_fcidump_path)

            npt.assert_allclose(
                one_body_tensor, one_body_tensor_2, rtol=test_rtol, atol=test_atol
            )
            npt.assert_allclose(
                np.zeros_like(two_body_tensor_2),
                two_body_tensor_2,
                rtol=test_rtol,
                atol=test_atol,
            )

            # Get the ground state energy directly from the one_body_tensor
            (
                ground_state_energy,
                eigenvalues,
                eigenvectors,
            ) = model_hamiltonians.get_one_body_term_ground_state(
                one_body_tensor, num_electrons
            )
            log.info(f"one_body_tensor: {one_body_tensor}")
            log.info(f"Ground state energy: {ground_state_energy}")
            log.info(f"Eigenvalues: {eigenvalues}")
            # Get the ground state energy from DMRG
            data_files_folder = temp_fcidump_path.parent

            data_file_list_file = [
                temp_fcidump_path.name,
            ]

            config_files_list, config_dict_single_file_list = self.prepare_dmrg_files(
                data_files_folder, data_file_list_file
            )

            for config_dict in config_dict_single_file_list:
                data_config = config_dict["data_config"]
                python_run_file_name = data_config["python_run_file"]
                main_storage_folder_path = data_config["main_storage_folder_path"]
                log.info(f"main_storage_folder_path: {main_storage_folder_path}")
                log.info(f"python_run_file_name: {python_run_file_name}")
                # log.info(open(python_run_file_name).readlines())
                # exec(open(python_run_file_name).read())
                # os.system("which python")
                # os.system("pwd")
                # os.system(f"env_dmrghandler/bin/python {python_run_file_name}")
                data_file_path = data_config["data_file_path"]
                driver = DMRGDriver(
                    scratch="./tmp", symm_type=SymmetryTypes.SU2, n_threads=1
                )
                driver.read_fcidump(filename=data_file_path, pg="c1")
                driver.initialize_system(
                    n_sites=driver.n_sites,
                    n_elec=driver.n_elec,
                    spin=driver.spin,
                    orb_sym=driver.orb_sym,
                )
                mpo = driver.get_qc_mpo(
                    h1e=driver.h1e, g2e=driver.g2e, ecore=driver.ecore, iprint=1
                )

                n_sweeps = 30
                bond_dims = [max(final_bond_dim // 2, 1)] * 4 + [final_bond_dim]
                noises = [1e-4] * 4 + [1e-5] * 4 + [0]
                thrds = [1e-8] * n_sweeps

                ket = driver.get_random_mps(
                    tag="KET", bond_dim=final_bond_dim, nroots=1
                )

                DMRG_energy = driver.dmrg(
                    mpo,
                    ket,
                    n_sweeps=n_sweeps,
                    bond_dims=bond_dims,
                    noises=noises,
                    thrds=thrds,
                    iprint=1,
                    twosite_to_onesite=20,
                )

                # log.info(driver.get_dmrg_results())
                log.info(f"DMRG energy: {DMRG_energy}")
                log.info(f"Ground state energy: {ground_state_energy}")
                log.info(f"DMRG Model Spin: {driver.spin}")

                npt.assert_allclose(
                    DMRG_energy,
                    ground_state_energy,
                    rtol=large_test_rtol,
                    atol=large_test_atol,
                )

                self.assertEqual(driver.spin, two_S)
                # self.assertEqual(final_bond_dim,)
                log.info(f"Final Bond Dimension: {final_bond_dim}")
                log.info(dir(ket))
                csfs, coeffs = driver.get_csf_coefficients(
                    ket, cutoff=0.00001, iprint=2
                )
                log.info(f"CSFs: {csfs}")
                log.info(f"Coeffs: {coeffs}")
                # bonds = ket.get_bond_dims(idxl=0, idxr=-1)
                # log.info("BOND DIMS HERE:")
                # log.info("|".join([str(sum(x.values())) for x in bonds]))

    def test_banded_tight_binding_model_large_auto_converge(self):
        num_orbitals_list = [
            # 2,
            3,
            5,
            10,
        ]  # 20, 30, 40, 50, 60, 70, 80, 90, 100]
        dict_list = [
            {
                "num_orbitals": num_orbitals,
                # "orbital_energies": -0.01 * np.array(range(num_orbitals)),
                "orbital_energies": np.zeros(num_orbitals),
                "hopping_amplitude": 1.0,
                "bandwidth": 1,
                "num_electrons": 1,
                "two_S": 1,
                # "orb_sym": np.zeros(num_orbitals, dtype=np.uint8),
                "orb_sym": np.ones(num_orbitals, dtype=np.int_),
                "calc_label": f"1e_tbt_{num_orbitals}_{hdf5_io.generate_uuid()}",
            }
            for num_orbitals in num_orbitals_list
            # {
            #     "num_orbitals": num_orbitals,
            #     "orbital_energies": np.ones(num_orbitals),
            #     "hopping_amplitude": 1.0,
            #     "bandwidth": num_orbitals - 1,
            #     "num_electrons": num_orbitals,
            #     "two_S": 0,
            #     "orb_sym": np.ones(num_orbitals, dtype=int),
            # },
        ]

        dict_list += [
            {
                "num_orbitals": num_orbitals,
                # "orbital_energies": -0.01 * np.array(range(num_orbitals)),
                "orbital_energies": np.zeros(num_orbitals),
                "hopping_amplitude": 1.0,
                "bandwidth": num_orbitals - 1,
                "num_electrons": 1,
                "two_S": 1,
                # "orb_sym": np.zeros(num_orbitals, dtype=np.uint8),
                "orb_sym": np.ones(num_orbitals, dtype=np.int_),
                "calc_label": f"1e_dense_{num_orbitals}_{hdf5_io.generate_uuid()}",
            }
            for num_orbitals in num_orbitals_list
            # {
            #     "num_orbitals": num_orbitals,
            #     "orbital_energies": np.ones(num_orbitals),
            #     "hopping_amplitude": 1.0,
            #     "bandwidth": num_orbitals - 1,
            #     "num_electrons": num_orbitals,
            #     "two_S": 0,
            #     "orb_sym": np.ones(num_orbitals, dtype=int),
            # },
        ]
        data_file_list = []
        gs_energy_list = []
        eigenvalues_list = []
        for dict in dict_list:
            num_orbitals = dict["num_orbitals"]
            orbital_energies = dict["orbital_energies"]
            hopping_amplitude = dict["hopping_amplitude"]
            bandwidth = dict["bandwidth"]
            num_electrons = dict["num_electrons"]
            two_S = dict["two_S"]
            orb_sym = dict["orb_sym"]
            calc_label = dict["calc_label"]

            one_body_tensor = model_hamiltonians.banded_tight_binding_model(
                num_orbitals, orbital_energies, hopping_amplitude, bandwidth
            )
            log.info(f"calc_label: {calc_label}")
            log.info(f"one_body_tensor: {one_body_tensor}")
            # npt.assert_allclose(
            #     one_body_tensor, dict["answer"], rtol=test_rtol, atol=test_atol
            # )

            # Save the one_body_tensor to a file
            temp_fcidump_path = Path(f"tests/model_hami/fcidump.{calc_label}")
            temp_fcidump_path.parent.mkdir(parents=True, exist_ok=True)

            two_body_tensor_sparse = None
            one_body_tensor_sparse = sp.sparse.lil_matrix(one_body_tensor)

            output_dict = {
                # Variable: (Description, Default Value)
                "NORB": num_orbitals,
                "NELEC": num_electrons,
                "MS2": two_S,
                "ISYM": 1,
                "ORBSYM": orb_sym,
                # "IPRTIM": ("If 0, print additional CPU timing analysis", -1),
                # "INT": ("Fortran stream from which integrals will be read", 5),
                # "MEMORY": ("Size of workspace array in floating point words", 100000),
                # "CORE": nuc_rep_energy,
                # "MAXIT": ("Maximum number of iterations in Davidson diagonalisation", 25),
                # "THR": (
                #     "Convergence threshold for Davidson diagonalisation (floating point)",
                #     1e-5,
                # ),
                # "THRRES": ("Threshold for printing final CI coefficients (floating point)", 1e-1),
                # "NROOT": ("Number of eigenvalues of Hamiltonian to be found", 1),
            }

            fcidump_io.write_fcidump(
                output_dict=output_dict,
                one_electron_integrals=one_body_tensor_sparse,
                two_electron_integrals=two_body_tensor_sparse,
                core_energy=0.0,
                file_path=temp_fcidump_path,
                real_bool=False,
                verbose=True,
            )
            data_file_list.append(str(temp_fcidump_path))

            # Get the ground state energy directly from the one_body_tensor
            (
                ground_state_energy,
                eigenvalues,
                eigenvectors,
            ) = model_hamiltonians.get_one_body_term_ground_state(
                one_body_tensor, num_electrons
            )
            # log.info(f"one_body_tensor: {one_body_tensor}")
            log.info(f"calc_label: {calc_label}")
            log.info(f"Ground state energy: {ground_state_energy}")
            log.info(f"Eigenvalues: {eigenvalues}")
            gs_energy_list.append(ground_state_energy)
            eigenvalues_list.append(eigenvalues)
        # data_file = "fcidump.test"

        log.info(
            f"data_file_list: {data_file_list}"
        )  # IF ERRORS HERE, WRONG PYTHON VERSION!!!!!

        config_dict = {
            "plot_filename_prefix_list": [
                "1e_bond_saturation_",
            ],
            "main_storage_folder_path_prefix": "./data_storage",
            "max_bond_dimension_list": [300],
            "max_time_limit_sec_list": [10 * 60],  # 10min
            "min_energy_change_hartree_list": [1e-4],
            "extrapolation_type_list": ["discard_weights"],
            "starting_bond_dimension_list": [2],
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
            "factor_half_convention_list": [False],
            # "symmetry_type_list": ["SZ", "SZ", "SU(2)", "SU(2)"],
            "symmetry_type_list": ["SZ"],
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
            config_file_prefix="1e_bond_saturation_",
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
        for dict_index, config_dict in enumerate(config_dict_single_file_list):
            data_config = config_dict["data_config"]
            python_run_file_name = data_config["python_run_file"]
            os.environ["SCRATCH"] = str(scratch_sim_path_absolute)
            os.system(f"env_dmrghandler/bin/python {python_run_file_name}")
            log.debug("DMRG NOW EXITED")
            # Get results
            main_storage_folder_path = data_config["main_storage_folder_path"]
            hdf5_file_path = Path(main_storage_folder_path) / Path("dmrg_results.hdf5")

            true_gs_dict = {
                "python_run_file_name": python_run_file_name,
                "gs_energy": gs_energy_list[dict_index],
                "eigenvalues": eigenvalues_list[dict_index],
                "data_file": data_file_list[dict_index],
            }
            hdf5_io.save_many_variables_to_hdf5(
                hdf5_filepath=hdf5_file_path,
                variables=true_gs_dict,
                access_mode="a",
                group="/true_gs_data",
                overwrite=False,
            )
