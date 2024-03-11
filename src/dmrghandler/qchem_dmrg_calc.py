"""
This module contains functions for running a single DMRG calculation.
"""

import inspect
import logging

log = logging.getLogger(__name__)
import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

default_final_bond_dim = 100
default_sweep_schedule_bond_dims = [default_final_bond_dim] * 4 + [
    default_final_bond_dim
] * 4
default_sweep_schedule_noise = [1e-4] * 4 + [1e-5] * 4 + [0]
default_sweep_schedule_davidson_threshold = [1e-10] * 8


def single_qchem_dmrg_calc(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    dmrg_parameters: dict,
    verbosity: int = 0,
):
    """
    This function runs a single DMRG calculation using the given one-body and two-body tensors and the DMRG parameters.

    Args:
        one_body_tensor (np.ndarray): The one-body tensor.
        two_body_tensor (np.ndarray): The two-body tensor.
        dmrg_parameters (dict): The DMRG parameters.

    Returns:
        dict: The DMRG calculation results.
    """
    # Extract the DMRG parameters
    factor_half_convention = dmrg_parameters["factor_half_convention"]
    symmetry_type_string = dmrg_parameters["symmetry_type"]
    num_threads = dmrg_parameters["num_threads"]
    n_mkl_threads = dmrg_parameters["n_mkl_threads"]
    num_orbitals = dmrg_parameters["num_orbitals"]
    num_spin_orbitals = dmrg_parameters["num_spin_orbitals"]
    num_electrons = dmrg_parameters["num_electrons"]
    two_S = dmrg_parameters["two_S"]
    two_Sz = dmrg_parameters["two_Sz"]
    orb_sym = dmrg_parameters["orb_sym"]  # Orbital symmetry, typically None
    temp_dir = dmrg_parameters["temp_dir"]
    stack_mem = dmrg_parameters["stack_mem"]  # Default value is 1073741824 bytes = 1GB
    restart_dir = dmrg_parameters[
        "restart_dir"
    ]  # MPS is copied to this directory after each DMRG sweep
    core_energy = dmrg_parameters["core_energy"]  # Default value is 0.0
    reordering_method = dmrg_parameters["reordering_method"]
    init_state_seed = dmrg_parameters["init_state_seed"]
    initial_mps_method = dmrg_parameters["initial_mps_method"]
    init_state_bond_dimension = dmrg_parameters["init_state_bond_dimension"]
    occupancy_hint = dmrg_parameters[
        "occupancy_hint"
    ]  # Hint of occupancy information, if None, uniform distribution is assumed
    full_fci_space_bool = dmrg_parameters[
        "full_fci_space_bool"
    ]  # If True, the full FCI space is used, default is True
    init_state_direct_two_site_construction_bool = dmrg_parameters[
        "init_state_direct_two_site_construction_bool"
    ]  # Default is False; if False, create MPS as one-site, then convert to two-site
    max_num_sweeps = dmrg_parameters["max_num_sweeps"]
    energy_convergence_threshold = dmrg_parameters[
        "energy_convergence_threshold"
    ]  # Default is 1E-8
    sweep_schedule_bond_dims = dmrg_parameters["sweep_schedule_bond_dims"]
    sweep_schedule_noise = dmrg_parameters["sweep_schedule_noise"]
    sweep_schedule_davidson_threshold = dmrg_parameters[
        "sweep_schedule_davidson_threshold"
    ]
    davidson_type = dmrg_parameters["davidson_type"]  # Default is None, for "Normal"
    eigenvalue_cutoff = dmrg_parameters[
        "eigenvalue_cutoff"
    ]  # Cutoff of eigenvalues, default is 1e-20
    davidson_max_iterations = dmrg_parameters[
        "davidson_max_iterations"
    ]  # Default is 4000
    davidson_max_krylov_subspace_size = dmrg_parameters[
        "davidson_max_krylov_subspace_size"
    ]  # Default is 50
    lowmem_noise_bool = dmrg_parameters[
        "lowmem_noise_bool"
    ]  # Whether to use a lower memory version of the noise, default is False
    sweep_start = dmrg_parameters["sweep_start"]  # Default is 0, where to start sweep
    initial_sweep_direction = dmrg_parameters[
        "initial_sweep_direction"
    ]  # Default is None, True means forward sweep (left-to-right)
    log.debug(f"LINE {inspect.getframeinfo(inspect.currentframe()).lineno}")
    assert (
        num_spin_orbitals == 2 * num_orbitals
    ), "The number of spin orbitals must be twice the number of orbitals."
    assert (
        num_electrons <= num_spin_orbitals
    ), "The number of electrons must be less than or equal to the number of spin orbitals."
    assert (
        np.abs(two_Sz) <= two_S
    ), "The z-component of the total spin must be less than or equal to the total spin."
    assert (
        np.abs(two_S) <= num_electrons
    ), "The twice total spin must be less than or equal to the number of electrons."

    if not factor_half_convention:
        # That is, H = h_pq*a^†_p a_q + 1/2 * g_pqrs * a^†_p a^†_r a_s a_q
        two_body_tensor_factor_half = two_body_tensor / 2
    else:
        two_body_tensor_factor_half = two_body_tensor

    if symmetry_type_string == "SU(2)":
        symmetry_type = SymmetryTypes.SU2
        spin = two_S

    elif symmetry_type_string == "SZ":
        symmetry_type = SymmetryTypes.SZ
        spin = two_Sz

    elif symmetry_type_string == "SGF":
        symmetry_type = SymmetryTypes.SGF
        spin = two_Sz
    else:
        raise ValueError(f"Invalid symmetry type: {symmetry_type_string}")
    log.debug(f"LINE {inspect.getframeinfo(inspect.currentframe()).lineno}")
    driver = DMRGDriver(
        stack_mem=stack_mem,
        scratch=temp_dir,
        clean_scratch=True,  # Default value
        restart_dir=restart_dir,
        n_threads=num_threads,
        # n_mkl_threads=n_mkl_threads,  # Default value is 1
        symm_type=symmetry_type,
        mpi=None,  # Default value
        stack_mem_ratio=0.4,  # Default value
        fp_codec_cutoff=1e-16,  # Default value
    )
    log.debug(f"LINE {inspect.getframeinfo(inspect.currentframe()).lineno}")
    driver.initialize_system(
        n_sites=num_orbitals,
        n_elec=num_electrons,
        spin=spin,
        orb_sym=orb_sym,
        heis_twos=-1,  # Default value
        heis_twosz=0,  # Default value
        singlet_embedding=True,  # Default value
        pauli_mode=False,  # Default value
        vacuum=None,  # Default value
        left_vacuum=None,  # Default value
        target=None,  # Default value
        hamil_init=True,  # Default value
    )
    log.debug(f"LINE {inspect.getframeinfo(inspect.currentframe()).lineno}")
    one_body_tensor_reordered, two_body_tensor_factor_half_reordered = (
        reorder_integrals(
            one_body_tensor=one_body_tensor,
            two_body_tensor_factor_half=two_body_tensor_factor_half,
            reordering_method=reordering_method,
        )
    )
    log.debug(f"LINE {inspect.getframeinfo(inspect.currentframe()).lineno}")
    qchem_hami_mpo = driver.get_qc_mpo(
        h1e=one_body_tensor_reordered,
        g2e=two_body_tensor_factor_half_reordered,
        ecore=core_energy,
        para_type=None,  # Default value
        reorder=None,  # Reordering not done here
        cutoff=1e-20,  # Default value
        integral_cutoff=1e-20,  # Default value
        post_integral_cutoff=1e-20,  # Default value
        fast_cutoff=1e-20,  # Default value
        unpack_g2e=True,  # Default value
        algo_type=None,  # Default value, MPOAlgorithmTypes.FastBipartite will be used
        normal_order_ref=None,  # Default value, normal ordering not done here
        normal_order_single_ref=None,  # Default value, only used if normal_order_ref is not None
        normal_order_wick=True,  # Default value, only used if normal_order_ref is not None
        symmetrize=False,  # Only impacts if orb_sym  in initialize_system is not None
        # Set to False to avoid unexpected behavior
        sum_mpo_mod=-1,  # Default value, no effect if algo_type=None
        compute_accurate_svd_error=True,  # Default value, no effect if algo_type=None
        csvd_sparsity=0.0,  # Default value, no effect if algo_type=None
        csvd_eps=1e-10,  # Default value, no effect if algo_type=None
        csvd_max_iter=1000,  # Default value, no effect if algo_type=None
        disjoint_levels=None,  # Default value, no effect if algo_type=None
        disjoint_all_blocks=False,  # Default value, no effect if algo_type=None
        disjoint_multiplier=1.0,  # Default value, no effect if algo_type=None
        block_max_length=False,  # Default value, no effect if algo_type=None
        add_ident=True,  # Default value, adds ecore*identity to the MPO for expectation values
        esptein_nesbet_partition=False,  # Default value, only used for perturbative DMRG
        ancilla=False,  # Default value, don't add ancilla sites
        # reorder_imat=None,  # Default value, will not reorder the integrals
        # gaopt_opts=None,  # Default value, options for gaopt reordering
        iprint=verbosity,
    )
    log.debug(f"LINE {inspect.getframeinfo(inspect.currentframe()).lineno}")
    # Generate the initial MPS
    ############################
    if initial_mps_method == "random":
        driver.bw.b.Random.rand_seed(init_state_seed)
    else:
        raise NotImplementedError(
            f"Not implemented initial MPS method: {initial_mps_method}"
        )
    log.debug(f"LINE {inspect.getframeinfo(inspect.currentframe()).lineno}")
    initial_ket = driver.get_random_mps(
        tag="init_ket",
        bond_dim=init_state_bond_dimension,
        center=0,  # Default value, canonical center of MPS
        dot=2,  # Default value, site type of MPS
        target=None,  # Default value, target quantum number
        nroots=1,  # Default value, number of roots, use 1 for ground state
        occs=occupancy_hint,  # Hint of occupancy information, if None, uniform distribution is assumed
        full_fci=full_fci_space_bool,  # If True, the full FCI space is used, default is True
        left_vacuum=None,  # Default value, only has effects for SU(2) and SE MPS with non-single target
        casci_ncore=0,  # For CASCI MPS, default is 0
        casci_nvirt=0,  # For CASCI MPS, default is 0
        mrci_order=0,  # For CASCI MPS, default is 0
        orig_dot=init_state_direct_two_site_construction_bool,  # Default is False; if False, create MPS as one-site, then convert to two-site
        # If True, create MPS as two-site or one-site directly
    )
    # log.debug(f"LINE {inspect.getframeinfo(inspect.currentframe()).lineno}")
    # b = driver.expr_builder()
    # b.add_term("(C+D)0", [0, 0], np.sqrt(2))
    # n_mpo = driver.get_mpo(b.finalize(), iprint=0)

    # n_0 = driver.expectation(initial_ket, n_mpo, initial_ket)
    # log.debug("N0 expectation = %20.15f" % (n_0))
    # initial_one_particle_density_matrix = driver.get_1pdm(initial_ket)
    # initial_one_particle_density_matrix = driver.get_npdm(
    #     ket=initial_ket,
    #     pdm_type=1,
    #     bra=None,
    #     soc=False,
    #     site_type=0,
    #     algo_type=None,
    #     npdm_expr=None,
    #     mask=None,
    #     simulated_parallel=0,
    #     fused_contraction_rotation=True,
    #     cutoff=1e-24,
    #     iprint=verbosity,
    #     # max_bond_dim=None,  # No restriction on the bond dimension
    # )
    # log.debug(f"LINE {inspect.getframeinfo(inspect.currentframe()).lineno}")
    # initial_two_particle_density_matrix = driver.get_npdm(
    #     ket=initial_ket,
    #     pdm_type=2,
    #     bra=None,
    #     soc=False,
    #     site_type=0,
    #     algo_type=None,
    #     npdm_expr=None,
    #     mask=None,
    #     simulated_parallel=0,
    #     fused_contraction_rotation=True,
    #     cutoff=1e-24,
    #     iprint=verbosity,
    #     # max_bond_dim=None,  # No restriction on the bond dimension
    # )
    log.debug(f"LINE {inspect.getframeinfo(inspect.currentframe()).lineno}")
    initial_bond_dims = initial_ket.info.bond_dim

    ket_optimized = driver.copy_mps(initial_ket, tag="ket_optimized")
    log.debug(f"LINE {inspect.getframeinfo(inspect.currentframe()).lineno}")
    dmrg_ground_state_energy = driver.dmrg(
        mpo=qchem_hami_mpo,
        ket=ket_optimized,
        n_sweeps=max_num_sweeps,
        tol=energy_convergence_threshold,
        bond_dims=sweep_schedule_bond_dims,
        noises=sweep_schedule_noise,
        thrds=sweep_schedule_davidson_threshold,
        iprint=verbosity,
        dav_type=davidson_type,
        cutoff=eigenvalue_cutoff,
        twosite_to_onesite=None,  # Don't switch the site type
        dav_max_iter=davidson_max_iterations,
        # dav_def_max_size=davidson_max_krylov_subspace_size,
        proj_mpss=None,  # For excited states, default is None
        proj_weights=None,  # For excited states, default is None
        store_wfn_spectra=True,  # Store MPS singular value spectra in self._sweep_wfn_spectra
        spectra_with_multiplicity=False,  # Don't multiply singular values with spin multiplicity (for SU2)
        # lowmem_noise=lowmem_noise_bool,  # Whether to use a lower memory version of the noise
        # sweep_start=sweep_start,
        # forward=initial_sweep_direction,
    )
    log.debug(f"LINE {inspect.getframeinfo(inspect.currentframe()).lineno}")
    dmrg_results = {
        "dmrg_ground_state_energy": dmrg_ground_state_energy,
        "initial_ket": initial_ket,
        # "initial_one_particle_density_matrix": initial_one_particle_density_matrix,
        # "initial_two_particle_density_matrix": initial_two_particle_density_matrix,
        "initial_bond_dims": initial_bond_dims,
        "ket_optimized": ket_optimized,
        # "singular_value_spectra": driver._sweep_wfn_spectra,
    }

    return dmrg_results


def reorder_integrals(
    one_body_tensor: np.ndarray,
    two_body_tensor_factor_half: np.ndarray,
    reordering_method: str,
):
    """
    This function reorders the one-body and two-body tensors.

    Args:
        one_body_tensor (np.ndarray): The one-body tensor.
        two_body_tensor_factor_half (np.ndarray): The two-body tensor.
        reordering_method (str): The reordering method.

    Returns:
        np.ndarray: The reordered one-body tensor.
        np.ndarray: The reordered two-body tensor.
    """
    if reordering_method == "none":
        one_body_tensor_reordered = one_body_tensor
        two_body_tensor_factor_half_reordered = two_body_tensor_factor_half
    else:
        raise ValueError(f"Invalid reordering method: {reordering_method}")

    return one_body_tensor_reordered, two_body_tensor_factor_half_reordered
