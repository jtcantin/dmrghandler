"""Note some of  the below code taken from examples in
QB Homogeneous catalyst benchmark:
https://zapco.sharepoint.com/:f:/s/ZapataExternalDocs/EnIIY-UDjjxHpbFju43rMAQBRxhgD1TNVUE3Yyhp4qDkSQ?e=Fac0sm
"""

import logging
from pathlib import Path

import h5py
import numpy as np
import pyscf
import pyscf.mcscf.avas
import pyscf.tools.fcidump

import dmrghandler.config_io as config_io
import dmrghandler.pyscf_wrappers as pyscf_wrappers

# import dmrghandler.data_loading as data_loading
log = logging.getLogger(__name__)


def prepare_calc(config_file_name):
    config_dict = config_io.load_configuration_data(config_file_name)

    data_config = config_dict["data_config"]
    data_file_path = data_config["data_file_path"]
    # plot_filename_prefix = data_config["plot_filename_prefix"]
    # main_storage_folder_path = data_config["main_storage_folder_path"]

    (
        one_body_tensor,
        two_body_tensor,
        nuc_rep_energy,
        num_orbitals,
        num_spin_orbitals,
        num_electrons,
        two_S,
        two_Sz,
        orb_sym,
        extra_attributes,
        spin_symm_broken,
    ) = load_tensors(data_file_path)

    dmrg_basic_config = config_dict["dmrg_basic_config"]
    dmrg_advanced_config = config_dict["dmrg_advanced_config"]
    dmrg_parameters = {**dmrg_basic_config, **dmrg_advanced_config}

    dmrg_parameters["num_orbitals"] = num_orbitals
    dmrg_parameters["num_spin_orbitals"] = num_spin_orbitals
    dmrg_parameters["num_electrons"] = num_electrons
    dmrg_parameters["two_S"] = two_S
    dmrg_parameters["two_Sz"] = two_Sz
    dmrg_parameters["orb_sym"] = orb_sym
    dmrg_parameters["core_energy"] = nuc_rep_energy

    config_io.ensure_required_in_dict(
        dictionary=dmrg_parameters,
        required_keys=[
            "factor_half_convention",
            "symmetry_type",
            "num_threads",
            "n_mkl_threads",
            "num_orbitals",
            "num_spin_orbitals",
            "num_electrons",
            "two_S",
            "two_Sz",
            "orb_sym",
            "temp_dir",
            "stack_mem",
            "restart_dir",
            "core_energy",
            "reordering_method",
            "init_state_seed",  # 0 means random seed
            "initial_mps_method",
            "init_state_bond_dimension",
            "occupancy_hint",
            "full_fci_space_bool",
            "init_state_direct_two_site_construction_bool",
            "max_num_sweeps",
            "energy_convergence_threshold",
            "sweep_schedule_bond_dims",
            "sweep_schedule_noise",
            "sweep_schedule_davidson_threshold",
            "davidson_type",  # Default is None, for "Normal"
            "eigenvalue_cutoff",  # Cutoff of eigenvalues, default is 1e-20
            "davidson_max_iterations",  # Default is 4000
            "davidson_max_krylov_subspace_size",  # Default is 50
            "lowmem_noise_bool",  # Whether to use a lower memory version of the noise, default is False
            "sweep_start",  # Default is 0, where to start sweep
            "initial_sweep_direction",  # Default is None, True means forward sweep (left-to-right)
        ],
    )

    looping_parameters = config_dict["looping_config"]

    if spin_symm_broken:
        assert dmrg_parameters["symmetry_type"] == "SZ", (
            "Spin-symmetry-broken systems must use 'SZ' symmetry type.\n"
            + f"Config file: {config_file_name}\n"
            + f"Data file: {data_file_path}"
        )

    return (
        one_body_tensor,
        two_body_tensor,
        dmrg_parameters,
        looping_parameters,
        data_config,
    )


def load_tensors(data_file_path):
    if data_file_path.endswith(".hdf5"):
        (
            one_body_tensor,
            two_body_tensor,
            nuc_rep_energy,
            num_orbitals,
            num_spin_orbitals,
            num_electrons,
            two_S,
            two_Sz,
            orb_sym,
            extra_attributes,
            spin_symm_broken,
        ) = load_tensors_from_hdf5(data_file_path)
    elif (Path(data_file_path).name).startswith("fcidump."):
        (
            one_body_tensor,
            two_body_tensor,
            nuc_rep_energy,
            num_orbitals,
            num_spin_orbitals,
            num_electrons,
            two_S,
            two_Sz,
            orb_sym,
            extra_attributes,
        ) = load_tensors_from_fcidump(data_file_path)
        spin_symm_broken = False  # FCI is always spin-symmetric
    else:
        raise ValueError(
            f"Data file format not recognized. File path: {data_file_path}"
        )

    return (
        one_body_tensor,
        two_body_tensor,
        nuc_rep_energy,
        num_orbitals,
        num_spin_orbitals,
        num_electrons,
        two_S,
        two_Sz,
        orb_sym,
        extra_attributes,
        spin_symm_broken,
    )


def load_tensors_from_hdf5(data_file_path, avas_threshold=0.2):
    with h5py.File(data_file_path, mode="r") as h5f:
        attributes = dict(h5f.attrs.items())
        one_body_tensor = np.array(h5f["one_body_tensor"])
        two_body_tensor = np.array(h5f["two_body_tensor"])
        log.debug(
            "------------------------------------------------------------------------------------------"
        )
        log.debug(
            "------------------------------------------------------------------------------------------"
        )
        log.debug(
            "------------------------------------------------------------------------------------------"
        )
        log.debug(
            "Spin-symmetry-broken system.\n"
            + f"two_body_tensor[0,0,0,0]: {two_body_tensor[0,0,0,0]}\n"
            + f"two_body_tensor[1,1,1,1]: {two_body_tensor[1,1,1,1]}\n"
            + f"two_body_tensor[0,0,1,1]: {two_body_tensor[0,0,1,1]}\n"
            + f"two_body_tensor[1,1,0,0]: {two_body_tensor[1,1,0,0]}\n"
            "Switching to Sz format."
        )
        log.debug(
            "------------------------------------------------------------------------------------------"
        )
        log.debug(
            "------------------------------------------------------------------------------------------"
        )
        log.debug(
            "------------------------------------------------------------------------------------------"
        )
        attributes["nqubits"] = one_body_tensor.shape[0]

    # if "avas_minao" in attributes.keys():
    #     raise ValueError(
    #         "The AVAS approach is not yet implemented for HDF5 file loading. \n"
    #         + "Please use the .FCIDUMP file format."
    #     )
    nuc_rep_energy_whole_molecule = float(attributes["constant"])
    num_orbitals_whole_molecule = int(attributes["nqubits"] // 2)
    num_spin_orbitals_whole_molecule = int(attributes["nqubits"])
    charge = int(attributes["charge"])
    multiplicity = float(attributes["multiplicity"])
    basis_whole_molecule = attributes["basis"]
    geometry_whole_molecule = attributes["geometry"]
    two_S = multiplicity - 1

    geometry_whole_molecule = standardize_geometry(geometry_whole_molecule)

    mol = pyscf_wrappers.get_pyscf_mol(
        basis=basis_whole_molecule,
        geometry=geometry_whole_molecule,
        num_unpaired_electrons=two_S,
        charge=charge,
        multiplicity=multiplicity,
    )

    if mol.spin == 0:
        mf = pyscf.scf.RHF(mol).run()
        log.debug(f"Using RHF.")
    else:
        log.warning(f"OPEN-SHELL system with {mol.spin} unpaired electrons.")
        log.warning(f"Using ROHF (NOT UHF).")
        # if (
        #     row["molecule"] == "I"
        #     or row["molecule"] == "pc-"
        #     or row["molecule"] == "4a"
        # ):
        #     mf = pyscf.scf.UHF(mol)
        # else:
        mf = pyscf.scf.ROHF(mol).run()

    if "avas_minao" in attributes.keys():
        avas_atomic_orbitals = attributes["avas_atomic_orbitals"]
        avas_minao = attributes["avas_minao"]
        log.debug(f"AVAS atomic orbitals: {avas_atomic_orbitals}")
        log.debug(f"AVAS minao: {avas_minao}")
        active_norb, active_ne, avas_orbs = pyscf.mcscf.avas.avas(
            mf,
            eval(avas_atomic_orbitals),
            minao=avas_minao,
            canonicalize=False,
            verbose=4,
            threshold=avas_threshold,
        )
        num_orbitals = active_norb
        num_spin_orbitals = active_norb * 2
        num_electrons = active_ne
        two_Sz = two_S
        nuc_rep_energy = 0.0
        log.warning("SETTING CORE ENERGY TO ZERO FOR AVAS.")
        log.warning("CAN WE GET CORE ENERGY WITHOUT RECALCULATING INTEGRALS?")
    else:
        num_orbitals = num_orbitals_whole_molecule
        num_spin_orbitals = num_spin_orbitals_whole_molecule
        num_electrons = mol.nelectron
        two_Sz = mol.spin
        nuc_rep_energy = nuc_rep_energy_whole_molecule
    orb_sym = None
    all_attributes = attributes
    # mean_field_object_from_fcidump = attributes["mean_field_object_from_fcidump"]

    one_body_tensor, two_body_tensor, spin_symm_broken = spinorbitals_to_orbitals(
        one_body_tensor, two_body_tensor
    )
    return (
        one_body_tensor,
        two_body_tensor,
        nuc_rep_energy,
        num_orbitals,
        num_spin_orbitals,
        num_electrons,
        two_S,
        two_Sz,
        orb_sym,
        all_attributes,
        spin_symm_broken,
    )


def load_tensors_from_fcidump(data_file_path):
    fci_data = pyscf.tools.fcidump.read(data_file_path)

    # dict_keys(['NORB', 'NELEC', 'MS2', 'ORBSYM', 'ISYM', 'ECORE', 'H1', 'H2'])
    num_orbitals = fci_data["NORB"]
    num_spin_orbitals = 2 * num_orbitals
    num_electrons = fci_data["NELEC"]
    two_S = fci_data["MS2"]
    two_Sz = fci_data["MS2"]
    orb_sym = fci_data["ORBSYM"]
    nuc_rep_energy = fci_data["ECORE"]
    one_body_tensor = fci_data["H1"]
    two_body_tensor_symmetrized = fci_data["H2"]

    two_body_tensor = pyscf.ao2mo.restore(
        "s1", two_body_tensor_symmetrized, num_orbitals
    )

    extra_attributes = {"ISYM": fci_data["ISYM"]}
    return (
        one_body_tensor,
        two_body_tensor,
        nuc_rep_energy,
        num_orbitals,
        num_spin_orbitals,
        num_electrons,
        two_S,
        two_Sz,
        orb_sym,
        extra_attributes,
    )


def standardize_geometry(geometry):
    """
    Convert [('Be', (0.0, 0.0, -0.0)), ('H', (-0.0, 0.0, 1.27457)), ('H', (0.0, -0.0, -1.27457))]
    to "Be 0.0 0.0 -0.0; H -0.0 0.0 1.27457; H 0.0 -0.0 -1.27457"
    """
    geo_string = ""
    for atom, coords in eval(geometry):
        geo_string += f"{atom} {coords[0]} {coords[1]} {coords[2]}; "

    return geo_string


def spinorbitals_to_orbitals(one_body_tensor, two_body_tensor):
    num_orbitals = one_body_tensor.shape[0] // 2
    one_body_tensor_orbitals = np.zeros((num_orbitals, num_orbitals))
    two_body_tensor_orbitals = np.zeros(
        (num_orbitals, num_orbitals, num_orbitals, num_orbitals)
    )
    spin_symm_broken = False
    for piter in range(num_orbitals):
        if spin_symm_broken:
            break
        for qiter in range(num_orbitals):
            if spin_symm_broken:
                break

            if (
                one_body_tensor[2 * piter, 2 * qiter]
                == one_body_tensor[2 * piter + 1, 2 * qiter + 1]
            ):
                one_body_tensor_orbitals[piter, qiter] = one_body_tensor[
                    2 * piter, 2 * qiter
                ]
            else:
                # raise NotImplementedError(
                #     "Spin-symmetry-broken systems not yet implemented.\n"
                #     + f"one_body_tensor[{2 * piter, 2 * qiter}]: {one_body_tensor[2 * piter, 2 * qiter]}\n"
                #     + f"one_body_tensor[{2 * piter + 1, 2 * qiter + 1}]: {one_body_tensor[2 * piter + 1, 2 * qiter + 1]}"
                # )
                log.warning(
                    "Spin-symmetry-broken systems not yet implemented.\n"
                    + f"one_body_tensor[{2 * piter, 2 * qiter}]: {one_body_tensor[2 * piter, 2 * qiter]}\n"
                    + f"one_body_tensor[{2 * piter + 1, 2 * qiter + 1}]: {one_body_tensor[2 * piter + 1, 2 * qiter + 1]}\n"
                    "Switching to Sz format."
                )
                spin_symm_broken = True
                break
            for riter in range(num_orbitals):
                if spin_symm_broken:
                    break
                for siter in range(num_orbitals):

                    if (
                        np.allclose(
                            two_body_tensor[2 * piter, 2 * qiter, 2 * riter, 2 * siter],
                            two_body_tensor[
                                2 * piter + 1, 2 * qiter + 1, 2 * riter, 2 * siter
                            ],
                        )
                        and np.allclose(
                            two_body_tensor[2 * piter, 2 * qiter, 2 * riter, 2 * siter],
                            two_body_tensor[
                                2 * piter, 2 * qiter, 2 * riter + 1, 2 * siter + 1
                            ],
                        )
                        and np.allclose(
                            two_body_tensor[2 * piter, 2 * qiter, 2 * riter, 2 * siter],
                            two_body_tensor[
                                2 * piter + 1,
                                2 * qiter + 1,
                                2 * riter + 1,
                                2 * siter + 1,
                            ],
                        )
                    ):
                        two_body_tensor_orbitals[piter, qiter, riter, siter] = (
                            two_body_tensor[2 * piter, 2 * qiter, 2 * riter, 2 * siter]
                        )
                    else:
                        # raise NotImplementedError(
                        #     "Spin-symmetry-broken systems not yet implemented.\n"
                        #     + f"two_body_tensor[{2 * piter, 2 * qiter, 2 * riter, 2 * siter}]: {two_body_tensor[2 * piter, 2 * qiter, 2 * riter, 2 * siter]}\n"
                        #     + f"two_body_tensor[{2 * piter + 1, 2 * qiter + 1, 2 * riter, 2 * siter}]: {two_body_tensor[2 * piter + 1, 2 * qiter + 1, 2 * riter, 2 * siter]}\n"
                        #     + f"two_body_tensor[{2 * piter, 2 * qiter, 2 * riter + 1, 2 * siter + 1}]: {two_body_tensor[2 * piter, 2 * qiter, 2 * riter + 1, 2 * siter + 1]}\n"
                        #     + f"two_body_tensor[{2 * piter + 1, 2 * qiter + 1, 2 * riter + 1, 2 * siter + 1}]: {two_body_tensor[2 * piter + 1, 2 * qiter + 1, 2 * riter + 1, 2 * siter + 1]}"
                        # )
                        log.warning(
                            "Spin-symmetry-broken system.\n"
                            + f"two_body_tensor[{2 * piter, 2 * qiter, 2 * riter, 2 * siter}]: {two_body_tensor[2 * piter, 2 * qiter, 2 * riter, 2 * siter]}\n"
                            + f"two_body_tensor[{2 * piter + 1, 2 * qiter + 1, 2 * riter, 2 * siter}]: {two_body_tensor[2 * piter + 1, 2 * qiter + 1, 2 * riter, 2 * siter]}\n"
                            + f"two_body_tensor[{2 * piter, 2 * qiter, 2 * riter + 1, 2 * siter + 1}]: {two_body_tensor[2 * piter, 2 * qiter, 2 * riter + 1, 2 * siter + 1]}\n"
                            + f"two_body_tensor[{2 * piter + 1, 2 * qiter + 1, 2 * riter + 1, 2 * siter + 1}]: {two_body_tensor[2 * piter + 1, 2 * qiter + 1, 2 * riter + 1, 2 * siter + 1]}\n"
                            "Switching to Sz format."
                        )
                        spin_symm_broken = True
                        break

    if spin_symm_broken:
        # Put into format for Block2 Sz
        # See https://block2.readthedocs.io/en/latest/api/pyblock2.html#pyblock2.driver.core.DMRGDriver.get_qc_mpo
        one_body_tensor_alpha = np.zeros((num_orbitals, num_orbitals))
        one_body_tensor_beta = np.zeros((num_orbitals, num_orbitals))
        two_body_tensor_orbitals_aaaa = np.zeros(
            (num_orbitals, num_orbitals, num_orbitals, num_orbitals)
        )
        two_body_tensor_orbitals_aabb = np.zeros(
            (num_orbitals, num_orbitals, num_orbitals, num_orbitals)
        )
        two_body_tensor_orbitals_bbbb = np.zeros(
            (num_orbitals, num_orbitals, num_orbitals, num_orbitals)
        )
        for piter in range(num_orbitals):
            for qiter in range(num_orbitals):
                one_body_tensor_alpha[piter, qiter] = one_body_tensor[
                    2 * piter, 2 * qiter
                ]
                one_body_tensor_beta[piter, qiter] = one_body_tensor[
                    2 * piter + 1, 2 * qiter + 1
                ]
                for riter in range(num_orbitals):
                    for siter in range(num_orbitals):
                        two_body_tensor_orbitals_aaaa[piter, qiter, riter, siter] = (
                            two_body_tensor[2 * piter, 2 * qiter, 2 * riter, 2 * siter]
                        )
                        two_body_tensor_orbitals_aabb[piter, qiter, riter, siter] = (
                            two_body_tensor[
                                2 * piter, 2 * qiter, 2 * riter + 1, 2 * siter + 1
                            ]
                        )
                        two_body_tensor_orbitals_bbbb[piter, qiter, riter, siter] = (
                            two_body_tensor[
                                2 * piter + 1,
                                2 * qiter + 1,
                                2 * riter + 1,
                                2 * siter + 1,
                            ]
                        )
        new_one_body_tensor = (one_body_tensor_alpha, one_body_tensor_beta)
        new_two_body_tensor = (
            two_body_tensor_orbitals_aaaa,
            two_body_tensor_orbitals_aabb,
            two_body_tensor_orbitals_bbbb,
        )
    else:
        new_one_body_tensor = one_body_tensor_orbitals
        new_two_body_tensor = two_body_tensor_orbitals

    return new_one_body_tensor, new_two_body_tensor, spin_symm_broken
