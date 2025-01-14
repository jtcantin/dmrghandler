"""
This module contains functions for running a single DMRG calculation.
"""

import inspect
import logging
import os
import time
import shutil

log = logging.getLogger(__name__)
import numpy as np
from memory_profiler import profile as mem_profile
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from dmrghandler.profiling import print_system_info

default_final_bond_dim = 100
default_sweep_schedule_bond_dims = [default_final_bond_dim] * 4 + [
    default_final_bond_dim
] * 4
default_sweep_schedule_noise = [1e-4] * 4 + [1e-5] * 4 + [0]
default_sweep_schedule_davidson_threshold = [1e-10] * 8
CSF_COEFF_THRESHOLD_DEFAULT = 1e-4


def single_qchem_dmrg_calc_mem_tracking(*args, track_mem=False, **kwargs):
    if track_mem:
        return mem_profile(single_qchem_dmrg_calc)(*args, **kwargs)
    else:
        return single_qchem_dmrg_calc(*args, **kwargs)


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
    wall_single_qchem_dmrg_calc_start_time_ns = time.perf_counter_ns()
    cpu_single_qchem_dmrg_calc_start_time_ns = time.process_time_ns()
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
    stack_mem_ratio = dmrg_parameters[
        "stack_mem_ratio"
    ]  # Default value is 0.4, ratio of stack memory to total memory

    if "keep_initial_ket_bool" in dmrg_parameters.keys():
        keep_initial_ket = dmrg_parameters["keep_initial_ket_bool"]
    else:
        keep_initial_ket = True

    if "calc_v_score_bool" in dmrg_parameters.keys():
        calc_v_score_bool = dmrg_parameters["calc_v_score_bool"]
    else:
        calc_v_score_bool = False

    if "csf_coeff_threshold" in dmrg_parameters.keys():
        csf_coeff_threshold = dmrg_parameters["csf_coeff_threshold"]
    else:
        csf_coeff_threshold = CSF_COEFF_THRESHOLD_DEFAULT

    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
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
    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    if "calc_1BD_energy_now" in dmrg_parameters.keys():
        if dmrg_parameters["calc_1BD_energy_now"]:
            hf_energy = get_bd_1_dmrg_energy(
                dmrg_parameters=dmrg_parameters,
                symmetry_type=symmetry_type,
                n_sites=num_orbitals,
                n_elec=num_electrons,
                spin=spin,
                orb_sym=orb_sym,
                one_body_tensor=one_body_tensor,
                two_body_tensor_factor_half=two_body_tensor_factor_half,
                ecore=core_energy,
                iprint=verbosity,
            )
            dmrg_parameters["hf_energy"] = hf_energy

        elif not dmrg_parameters["calc_1BD_energy_now"] and calc_v_score_bool:
            hf_energy = dmrg_parameters["hf_energy"]

    log.debug(f"DMRG parameters right before driver initialization: {dmrg_parameters}")
    # If start sweep not zero, then the initial MPS is loaded from the restart directory

    if sweep_start != 0:
        # If it already exists, clean out the scratch directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

        # Copy the MPS from the restart directory to the scratch directory
        shutil.copytree(restart_dir, temp_dir)

    wall_make_driver_start_time_ns = time.perf_counter_ns()
    cpu_make_driver_start_time_ns = time.process_time_ns()
    driver = DMRGDriver(
        stack_mem=stack_mem,
        scratch=temp_dir,
        clean_scratch=True,  # Default value
        restart_dir=restart_dir,
        n_threads=num_threads,
        # n_mkl_threads=n_mkl_threads,  # Default value is 1
        symm_type=symmetry_type,
        mpi=None,  # Default value
        stack_mem_ratio=stack_mem_ratio,  # Default value 0.4
        fp_codec_cutoff=1e-16,  # Default value 1e-16
    )
    log.debug(f"recorded stack_mem_ratio: {driver.stack_mem_ratio}")
    log.debug(f"recorded stack_mem: {driver.stack_mem}")
    wall_make_driver_end_time_ns = time.perf_counter_ns()
    cpu_make_driver_end_time_ns = time.process_time_ns()

    wall_make_driver_time_ns = (
        wall_make_driver_end_time_ns - wall_make_driver_start_time_ns
    )
    cpu_make_driver_time_ns = (
        cpu_make_driver_end_time_ns - cpu_make_driver_start_time_ns
    )

    log.info(f"wall_make_driver_time_s: {wall_make_driver_time_ns/1e9}")
    log.info(f"cpu_make_driver_time_s: {cpu_make_driver_time_ns/1e9}")
    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    wall_reorder_integrals_start_time_ns = time.perf_counter_ns()
    cpu_reorder_integrals_start_time_ns = time.process_time_ns()

    (
        one_body_tensor_reordered,
        two_body_tensor_factor_half_reordered,
        orb_sym_reordered,
        reordering_indices,
        reorder_output_dict,
    ) = reorder_integrals(
        one_body_tensor=one_body_tensor,
        two_body_tensor_factor_half=two_body_tensor_factor_half,
        orb_sym=orb_sym,
        reordering_method=reordering_method,
        driver=driver,
        n_sites=num_orbitals,
        n_elec=num_electrons,
        spin=spin,
        ecore=core_energy,
        iprint=verbosity,
        bond_dim=min(50, max(sweep_schedule_bond_dims)),
        dmrg_parameters=dmrg_parameters,
    )
    wall_reorder_integrals_end_time_ns = time.perf_counter_ns()
    cpu_reorder_integrals_end_time_ns = time.process_time_ns()

    wall_reorder_integrals_time_ns = (
        wall_reorder_integrals_end_time_ns - wall_reorder_integrals_start_time_ns
    )
    cpu_reorder_integrals_time_ns = (
        cpu_reorder_integrals_end_time_ns - cpu_reorder_integrals_start_time_ns
    )

    log.info(f"wall_reorder_integrals_time_s: {wall_reorder_integrals_time_ns/1e9}")
    log.info(f"cpu_reorder_integrals_time_s: {cpu_reorder_integrals_time_ns/1e9}")

    wall_driver_initialize_system_start_time_ns = time.perf_counter_ns()
    cpu_driver_initialize_system_start_time_ns = time.process_time_ns()

    driver.initialize_system(
        n_sites=num_orbitals,
        n_elec=num_electrons,
        spin=spin,
        orb_sym=orb_sym_reordered,
        heis_twos=-1,  # Default value
        heis_twosz=0,  # Default value
        singlet_embedding=True,  # Default value
        pauli_mode=False,  # Default value
        vacuum=None,  # Default value
        left_vacuum=None,  # Default value
        target=None,  # Default value
        hamil_init=True,  # Default value
    )
    log.debug("DRIVER INPUTS HERE!!!!!")
    log.debug(f"num_orbitals: {num_orbitals}")
    log.debug(f"num_electrons: {num_electrons}")
    log.debug(f"spin: {spin}")
    log.debug(f"orb_sym_reordered: {orb_sym_reordered}")
    log.debug(f"reordering_indices: {reordering_indices}")

    log.debug("DRIVER INFO HERE!!!!!")
    log.debug(f"driver.n_sites: {driver.n_sites}")
    # log.debug(f"driver.n_sites: {dir(driver)}")
    # log.debug(f"driver.n_elec: {driver.n_elec}")
    # log.debug(f"driver.spin: {driver.spin}")
    log.debug(f"driver.orb_sym: {driver.orb_sym}")

    wall_driver_initialize_system_end_time_ns = time.perf_counter_ns()
    cpu_driver_initialize_system_end_time_ns = time.process_time_ns()

    wall_driver_initialize_system_time_ns = (
        wall_driver_initialize_system_end_time_ns
        - wall_driver_initialize_system_start_time_ns
    )
    cpu_driver_initialize_system_time_ns = (
        cpu_driver_initialize_system_end_time_ns
        - cpu_driver_initialize_system_start_time_ns
    )

    log.info(
        f"wall_driver_initialize_system_time_s: {wall_driver_initialize_system_time_ns/1e9}"
    )
    log.info(
        f"cpu_driver_initialize_system_time_s: {cpu_driver_initialize_system_time_ns/1e9}"
    )

    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    wall_get_qchem_hami_mpo_start_time_ns = time.perf_counter_ns()
    cpu_get_qchem_hami_mpo_start_time_ns = time.process_time_ns()
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
    wall_get_qchem_hami_mpo_end_time_ns = time.perf_counter_ns()
    cpu_get_qchem_hami_mpo_end_time_ns = time.process_time_ns()

    wall_get_qchem_hami_mpo_time_ns = (
        wall_get_qchem_hami_mpo_end_time_ns - wall_get_qchem_hami_mpo_start_time_ns
    )
    cpu_get_qchem_hami_mpo_time_ns = (
        cpu_get_qchem_hami_mpo_end_time_ns - cpu_get_qchem_hami_mpo_start_time_ns
    )

    log.info(f"wall_get_qchem_hami_mpo_time_s: {wall_get_qchem_hami_mpo_time_ns/1e9}")
    log.info(f"cpu_get_qchem_hami_mpo_time_s: {cpu_get_qchem_hami_mpo_time_ns/1e9}")
    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    # Generate the initial MPS
    ############################
    wall_generate_initial_mps_start_time_ns = time.perf_counter_ns()
    cpu_generate_initial_mps_start_time_ns = time.process_time_ns()
    if initial_mps_method == "random":
        driver.bw.b.Random.rand_seed(init_state_seed)
        if keep_initial_ket:
            tag_initial = "ket_optimized"
        else:
            tag_initial = "init_ket"
        initial_ket = driver.get_random_mps(
            tag=tag_initial,
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

    elif initial_mps_method == "restart":
        initial_ket = driver.load_mps(tag="ket_optimized", nroots=1)

    else:
        raise NotImplementedError(
            f"Not implemented initial MPS method: {initial_mps_method}"
        )
    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
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
    wall_generate_initial_mps_end_time_ns = time.perf_counter_ns()
    cpu_generate_initial_mps_end_time_ns = time.process_time_ns()

    wall_generate_initial_mps_time_ns = (
        wall_generate_initial_mps_end_time_ns - wall_generate_initial_mps_start_time_ns
    )
    cpu_generate_initial_mps_time_ns = (
        cpu_generate_initial_mps_end_time_ns - cpu_generate_initial_mps_start_time_ns
    )

    log.info(
        f"wall_generate_initial_mps_time_s: {wall_generate_initial_mps_time_ns/1e9}"
    )
    log.info(f"cpu_generate_initial_mps_time_s: {cpu_generate_initial_mps_time_ns/1e9}")
    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    initial_bond_dims = initial_ket.info.bond_dim

    wall_copy_mps_start_time_ns = time.perf_counter_ns()
    cpu_copy_mps_start_time_ns = time.process_time_ns()
    initial_ket_energy = driver.expectation(initial_ket, qchem_hami_mpo, initial_ket)

    if keep_initial_ket:
        ket_optimized = driver.copy_mps(initial_ket, tag="ket_optimized")
    else:
        ket_optimized = initial_ket

    wall_copy_mps_end_time_ns = time.perf_counter_ns()
    cpu_copy_mps_end_time_ns = time.process_time_ns()

    wall_copy_mps_time_ns = wall_copy_mps_end_time_ns - wall_copy_mps_start_time_ns
    cpu_copy_mps_time_ns = cpu_copy_mps_end_time_ns - cpu_copy_mps_start_time_ns

    log.info(f"wall_copy_mps_time_s: {wall_copy_mps_time_ns/1e9}")
    log.info(f"cpu_copy_mps_time_s: {cpu_copy_mps_time_ns/1e9}")

    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    log.debug(f"sweep_schedule_bond_dims: {sweep_schedule_bond_dims}")
    log.debug(f"sweep_schedule_noise: {sweep_schedule_noise}")
    log.debug(f"sweep_schedule_davidson_threshold: {sweep_schedule_davidson_threshold}")
    wall_dmrg_optimization_start_time_ns = time.perf_counter_ns()
    cpu_dmrg_optimization_start_time_ns = time.process_time_ns()
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
        sweep_start=sweep_start,
        forward=initial_sweep_direction,
    )
    wall_dmrg_optimization_end_time_ns = time.perf_counter_ns()
    cpu_dmrg_optimization_end_time_ns = time.process_time_ns()

    wall_dmrg_optimization_time_ns = (
        wall_dmrg_optimization_end_time_ns - wall_dmrg_optimization_start_time_ns
    )
    cpu_dmrg_optimization_time_ns = (
        cpu_dmrg_optimization_end_time_ns - cpu_dmrg_optimization_start_time_ns
    )

    log.info(f"wall_dmrg_optimization_time_s: {wall_dmrg_optimization_time_ns/1e9}")
    log.info(f"cpu_dmrg_optimization_time_s: {cpu_dmrg_optimization_time_ns/1e9}")

    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    sweep_bond_dims, sweep_max_discarded_weight, sweep_energies = (
        driver.get_dmrg_results()
    )
    wall_single_qchem_dmrg_calc_end_time_ns = time.perf_counter_ns()
    cpu_single_qchem_dmrg_calc_end_time_ns = time.process_time_ns()

    wall_single_qchem_dmrg_calc_time_ns = (
        wall_single_qchem_dmrg_calc_end_time_ns
        - wall_single_qchem_dmrg_calc_start_time_ns
    )
    cpu_single_qchem_dmrg_calc_time_ns = (
        cpu_single_qchem_dmrg_calc_end_time_ns
        - cpu_single_qchem_dmrg_calc_start_time_ns
    )

    if calc_v_score_bool:
        print_system_info(
            f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
        )
        wall_v_score_start_time_ns = time.perf_counter_ns()
        cpu_v_score_start_time_ns = time.process_time_ns()
        # # Diagonalize the one-body tensor to get the orbital energies
        # orbital_energies, _ = np.linalg.eigh(one_body_tensor_reordered)
        (
            h_min_e_ket_norm,
            variance,
            v_score_numerator,
            deviation_init_ket,
            v_score_init_ket,
            hf_energy,
            deviation_hf,
            v_score_hartree_fock,
        ) = calc_v_score(
            ket=ket_optimized,
            hamiltonian_mpo=qchem_hami_mpo,
            num_electrons=num_electrons,
            driver=driver,
            init_ket=None,
            # dmrg_parameters=dmrg_parameters,
            # symmetry_type=symmetry_type,
            # n_sites=num_orbitals,
            # n_elec=num_electrons,
            # spin=spin,
            # orb_sym=orb_sym_reordered,
            # one_body_tensor=one_body_tensor_reordered,
            # two_body_tensor_factor_half=two_body_tensor_factor_half_reordered,
            # ecore=core_energy,
            # iprint=verbosity,
            # orbital_energies=np.diag(one_body_tensor_reordered),
            # orbital_energies=orbital_energies,
            core_energy=core_energy,
            ket_energy=sweep_energies[-1],
            init_ket_energy=initial_ket_energy,
            hf_energy=hf_energy,
        )

        wall_v_score_end_time_ns = time.perf_counter_ns()
        cpu_v_score_end_time_ns = time.process_time_ns()

        wall_v_score_time_ns = wall_v_score_end_time_ns - wall_v_score_start_time_ns
        cpu_v_score_time_ns = cpu_v_score_end_time_ns - cpu_v_score_start_time_ns

        log.info(f"wall_v_score_time_s: {wall_v_score_time_ns/1e9}")
        log.info(f"cpu_v_score_time_s: {cpu_v_score_time_ns/1e9}")

        v_score_result_dict = {
            "h_min_e_optket_norm": h_min_e_ket_norm,
            "optket_variance": variance,
            "v_score_numerator": v_score_numerator,
            "deviation_init_ket": deviation_init_ket,
            "v_score_init_ket": v_score_init_ket,
            "hf_energy": hf_energy,
            "deviation_hf": deviation_hf,
            "v_score_hartree_fock": v_score_hartree_fock,
            "initial_ket_energy": initial_ket_energy,
        }
        for key, value in v_score_result_dict.items():
            log.info(f"{key}: {value}")

    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )

    # Obtaining overlaps based on code by Alex Kunitsa: https://github.com/isi-usc-edu/qb-gsee-benchmark/blob/5449de07c12974b0ea6ee7409b9528d9edecbb33/src/qb_gsee_benchmark/dmrg_utils.py#L10
    csf_definitions, csf_coefficients = driver.get_csf_coefficients(
        ket_optimized,
        cutoff=csf_coeff_threshold,
        given_dets=None,
        max_print=20,
        fci_conv=False,
        # max_excite=None,
        # ref_det=None,
        iprint=verbosity,
    ) 
    # csf_definitions: 2D array  of shape (n_dets, n_sites); for value meanings, see https://block2.readthedocs.io/en/latest/api/pyblock2.html#pyblock2.driver.core.DMRGDriver.get_csf_coefficients
    # csf_coefficients: 1D array of length n_dets

    abs_coeffs = np.abs(csf_coefficients)
    largest_csf_coefficient = csf_coefficients[np.argmax(abs_coeffs)]
    largest_csf = csf_definitions[np.argmax(abs_coeffs),:]

    argsort_abs_coeffs = np.argsort(abs_coeffs) # Smallest to largest
    sorted_abs_coeffs = abs_coeffs[argsort_abs_coeffs]
    sorted_csf_definitions = csf_definitions[argsort_abs_coeffs,:]
    sorted_csf_coefficients = csf_coefficients[argsort_abs_coeffs]


    dmrg_discarded_weight = sweep_max_discarded_weight[-1]
    dmrg_results_dict = {
        "dmrg_driver": driver,
        "dmrg_ground_state_energy": dmrg_ground_state_energy,
        "initial_ket": initial_ket,
        # "initial_one_particle_density_matrix": initial_one_particle_density_matrix,
        # "initial_two_particle_density_matrix": initial_two_particle_density_matrix,
        "initial_bond_dims": initial_bond_dims,
        "ket_optimized": ket_optimized,
        # "singular_value_spectra": driver._sweep_wfn_spectra,
        "dmrg_discarded_weight": dmrg_discarded_weight,
        "sweep_bond_dims": sweep_bond_dims,
        "sweep_max_discarded_weight": sweep_max_discarded_weight,
        "sweep_energies": sweep_energies,
        "wall_make_driver_time_s": wall_make_driver_time_ns / 1e9,
        "cpu_make_driver_time_s": cpu_make_driver_time_ns / 1e9,
        "wall_driver_initialize_system_time_s": wall_driver_initialize_system_time_ns
        / 1e9,
        "cpu_driver_initialize_system_time_s": cpu_driver_initialize_system_time_ns
        / 1e9,
        "wall_reorder_integrals_time_s": wall_reorder_integrals_time_ns / 1e9,
        "cpu_reorder_integrals_time_s": cpu_reorder_integrals_time_ns / 1e9,
        "wall_get_qchem_hami_mpo_time_s": wall_get_qchem_hami_mpo_time_ns / 1e9,
        "cpu_get_qchem_hami_mpo_time_s": cpu_get_qchem_hami_mpo_time_ns / 1e9,
        "wall_generate_initial_mps_time_s": wall_generate_initial_mps_time_ns / 1e9,
        "cpu_generate_initial_mps_time_s": cpu_generate_initial_mps_time_ns / 1e9,
        "wall_copy_mps_time_s": wall_copy_mps_time_ns / 1e9,
        "cpu_copy_mps_time_s": cpu_copy_mps_time_ns / 1e9,
        "wall_dmrg_optimization_time_s": wall_dmrg_optimization_time_ns / 1e9,
        "cpu_dmrg_optimization_time_s": cpu_dmrg_optimization_time_ns / 1e9,
        "wall_single_qchem_dmrg_calc_time_s": wall_single_qchem_dmrg_calc_time_ns / 1e9,
        "cpu_single_qchem_dmrg_calc_time_s": cpu_single_qchem_dmrg_calc_time_ns / 1e9,
        "reordering_indices_used": reordering_indices,
        "reordering_method_used": reordering_method,
        "initial_ket_energy": initial_ket_energy,
        "csf_coefficients_real_part": np.real(sorted_csf_coefficients),
        "csf_coefficients_imag_part": np.imag(sorted_csf_coefficients),
        "csf_definitions_top20": sorted_csf_definitions[-20:,:],
        "num_csf": len(csf_coefficients),
        "csf_coeff_threshold": csf_coeff_threshold,
        "largest_csf_coefficient_real_part": np.real(largest_csf_coefficient),
        "largest_csf_coefficient_imag_part": np.imag(largest_csf_coefficient),
        "largest_csf": largest_csf,
    }

    if reorder_output_dict is not None:
        dmrg_results_dict.update(reorder_output_dict)

    if calc_v_score_bool:
        dmrg_results_dict.update(v_score_result_dict)

    return dmrg_results_dict


def reorder_integrals(
    one_body_tensor: np.ndarray,
    two_body_tensor_factor_half: np.ndarray,
    orb_sym: list,
    reordering_method: str,
    driver: DMRGDriver,
    n_sites,
    n_elec,
    spin,
    ecore,
    iprint=0,
    bond_dim=50,
    dmrg_parameters=None,
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
    log.debug(f"Reordering method: {reordering_method}")
    if reordering_method == "none":
        one_body_tensor_reordered = one_body_tensor
        two_body_tensor_factor_half_reordered = two_body_tensor_factor_half
        orb_sym_reordered = orb_sym
        reordering_indices = np.array(list(range(np.array(one_body_tensor).shape[0])))
        reorder_output_dict = None

    elif reordering_method == "gaopt, exchange matrix":
        log.debug("Orbital Reordering Method: gaopt, exchange matrix")
        idx = driver.orbital_reordering(
            one_body_tensor, two_body_tensor_factor_half, method="gaopt"
        )
        h1e = one_body_tensor[idx][:, idx]
        g2e = two_body_tensor_factor_half[idx][:, idx][:, :, idx][:, :, :, idx]
        orb_sym_reordered = np.array(orb_sym)[idx]
        one_body_tensor_reordered = h1e
        two_body_tensor_factor_half_reordered = g2e
        reordering_indices = idx
        reorder_output_dict = None

        # raise NotImplementedError(
        #     "The 'gaopt, exchange matrix' reordering method is not implemented."
        # )
        # log.debug("Orbital Reordering Method: gaopt, exchange matrix")
        # idx = driver.orbital_reordering(
        #     one_body_tensor, two_body_tensor_factor_half, method="gaopt"
        # )
        # h1e = one_body_tensor[idx][:, idx]
        # g2e = two_body_tensor_factor_half[idx][:, idx][:, :, idx][:, :, :, idx]
        # orb_sym_reordered = np.array(orb_sym)[idx]
        # one_body_tensor_reordered = h1e
        # two_body_tensor_factor_half_reordered = g2e
        # reordering_indices = idx
        # reorder_output_dict = None

    elif reordering_method == "gaopt, interaction matrix":
        # raise NotImplementedError(
        #     "The 'gaopt, interaction matrix' reordering method is not implemented."
        # )
        log.debug("Orbital Reordering Method: gaopt, interaction matrix")
        # approx DMRG to get orbital_interaction_matrix
        driver_local = DMRGDriver(
            stack_mem=dmrg_parameters["stack_mem"],
            scratch=dmrg_parameters["temp_dir"],
            clean_scratch=True,  # Default value
            restart_dir=dmrg_parameters["restart_dir"],
            n_threads=dmrg_parameters["num_threads"],
            # n_mkl_threads=n_mkl_threads,  # Default value is 1
            symm_type=SymmetryTypes.SZ,
            mpi=None,  # Default value
            stack_mem_ratio=dmrg_parameters["stack_mem_ratio"],  # Default value 0.4
            fp_codec_cutoff=1e-16,  # Default value 1e-16
        )
        driver_local.initialize_system(
            n_sites=n_sites, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )
        mpo = driver_local.get_qc_mpo(
            h1e=one_body_tensor,
            g2e=two_body_tensor_factor_half,
            ecore=ecore,
            iprint=iprint,
        )
        ket = driver_local.get_random_mps(
            tag="orbital_ordering", bond_dim=bond_dim, nroots=1
        )
        energy = driver_local.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=[bond_dim] * 9,
            noises=[1e-4] * 4 + [1e-5] * 4 + [0],
            thrds=[1e-10] * 9,
            iprint=1,
        )
        log.debug("Approx Orbital Reordering DMRG energy = %20.15f" % energy)
        minfo_orig = driver_local.get_orbital_interaction_matrix(ket)

        idx = driver_local.orbital_reordering_interaction_matrix(
            minfo_orig, method="gaopt"
        )
        h1e = one_body_tensor[idx][:, idx]
        g2e = two_body_tensor_factor_half[idx][:, idx][:, :, idx][:, :, :, idx]
        orb_sym_reordered = np.array(orb_sym)[idx]
        one_body_tensor_reordered = h1e
        two_body_tensor_factor_half_reordered = g2e
        reordering_indices = idx
        reorder_output_dict = {
            "minfo_orig": minfo_orig,
            "reordering_bond_dim": bond_dim,
        }

    elif reordering_method == "gaopt, interaction matrix, SU(2) calc":
        # raise NotImplementedError(
        #     "The 'gaopt, interaction matrix' reordering method is not implemented."
        # )
        log.debug("Orbital Reordering Method: gaopt, interaction matrix, SU(2) calc")
        # approx DMRG to get orbital_interaction_matrix
        driver_local = DMRGDriver(
            stack_mem=dmrg_parameters["stack_mem"],
            scratch=dmrg_parameters["temp_dir"],
            clean_scratch=True,  # Default value
            restart_dir=dmrg_parameters["restart_dir"],
            n_threads=dmrg_parameters["num_threads"],
            # n_mkl_threads=n_mkl_threads,  # Default value is 1
            symm_type=SymmetryTypes.SU2,
            mpi=None,  # Default value
            stack_mem_ratio=dmrg_parameters["stack_mem_ratio"],  # Default value 0.4
            fp_codec_cutoff=1e-16,  # Default value 1e-16
        )
        driver_local.initialize_system(
            n_sites=n_sites, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )
        mpo = driver_local.get_qc_mpo(
            h1e=one_body_tensor,
            g2e=two_body_tensor_factor_half,
            ecore=ecore,
            iprint=iprint,
        )
        ket = driver_local.get_random_mps(
            tag="orbital_ordering", bond_dim=bond_dim, nroots=1
        )
        energy = driver_local.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=[bond_dim] * 9,
            noises=[1e-4] * 4 + [1e-5] * 4 + [0],
            thrds=[1e-10] * 9,
            iprint=1,
        )
        log.debug("Approx Orbital Reordering DMRG energy = %20.15f" % energy)

        # Convert to Sz symmetry
        zket = driver_local.mps_change_to_sz(ket, "ZKET")

        driver_local.symm_type = SymmetryTypes.SZ
        driver_local.initialize_system(
            n_sites=n_sites, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )

        minfo_orig = driver_local.get_orbital_interaction_matrix(zket)

        idx = driver_local.orbital_reordering_interaction_matrix(
            minfo_orig, method="gaopt"
        )
        h1e = one_body_tensor[idx][:, idx]
        g2e = two_body_tensor_factor_half[idx][:, idx][:, :, idx][:, :, :, idx]
        orb_sym_reordered = np.array(orb_sym)[idx]
        one_body_tensor_reordered = h1e
        two_body_tensor_factor_half_reordered = g2e
        reordering_indices = idx
        reorder_output_dict = {
            "minfo_orig": minfo_orig,
            "reordering_bond_dim": bond_dim,
        }

    elif reordering_method == "fiedler, interaction matrix, SU(2) calc":
        # raise NotImplementedError(
        #     "The 'gaopt, interaction matrix' reordering method is not implemented."
        # )
        log.debug("Orbital Reordering Method: fiedler, interaction matrix, SU(2) calc")
        # approx DMRG to get orbital_interaction_matrix
        driver_local = DMRGDriver(
            stack_mem=dmrg_parameters["stack_mem"],
            scratch=dmrg_parameters["temp_dir"],
            clean_scratch=True,  # Default value
            restart_dir=dmrg_parameters["restart_dir"],
            n_threads=dmrg_parameters["num_threads"],
            # n_mkl_threads=n_mkl_threads,  # Default value is 1
            symm_type=SymmetryTypes.SU2,
            mpi=None,  # Default value
            stack_mem_ratio=dmrg_parameters["stack_mem_ratio"],  # Default value 0.4
            fp_codec_cutoff=1e-16,  # Default value 1e-16
        )
        driver_local.initialize_system(
            n_sites=n_sites, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )
        mpo = driver_local.get_qc_mpo(
            h1e=one_body_tensor,
            g2e=two_body_tensor_factor_half,
            ecore=ecore,
            iprint=iprint,
        )
        ket = driver_local.get_random_mps(
            tag="orbital_ordering", bond_dim=bond_dim, nroots=1
        )
        energy = driver_local.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=[bond_dim] * 9,
            noises=[1e-4] * 4 + [1e-5] * 4 + [0],
            thrds=[1e-10] * 9,
            iprint=1,
        )
        log.debug("Approx Orbital Reordering DMRG energy = %20.15f" % energy)

        # Convert to Sz symmetry
        zket = driver_local.mps_change_to_sz(ket, "ZKET")

        driver_local.symm_type = SymmetryTypes.SZ
        driver_local.initialize_system(
            n_sites=n_sites, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )

        minfo_orig = driver_local.get_orbital_interaction_matrix(zket)

        idx = driver_local.orbital_reordering_interaction_matrix(
            minfo_orig, method="fiedler"
        )
        h1e = one_body_tensor[idx][:, idx]
        g2e = two_body_tensor_factor_half[idx][:, idx][:, :, idx][:, :, :, idx]
        orb_sym_reordered = np.array(orb_sym)[idx]
        one_body_tensor_reordered = h1e
        two_body_tensor_factor_half_reordered = g2e
        reordering_indices = idx
        reorder_output_dict = {
            "minfo_orig": minfo_orig,
            "reordering_bond_dim": bond_dim,
        }

    elif reordering_method == "fiedler, exchange matrix":
        log.debug("Orbital Reordering Method: fiedler, exchange matrix")
        idx = driver.orbital_reordering(one_body_tensor, two_body_tensor_factor_half)
        h1e = one_body_tensor[idx][:, idx]
        g2e = two_body_tensor_factor_half[idx][:, idx][:, :, idx][:, :, :, idx]
        orb_sym_reordered = np.array(orb_sym)[idx]
        one_body_tensor_reordered = h1e
        two_body_tensor_factor_half_reordered = g2e
        reordering_indices = idx
        reorder_output_dict = None

    elif reordering_method == "fiedler, interaction matrix":
        # raise NotImplementedError(
        #     "The 'fiedler, interaction matrix' reordering method is not implemented."
        # )
        log.debug("Orbital Reordering Method: fiedler, interaction matrix")
        # approx DMRG to get orbital_interaction_matrix

        # print(dmrg_parameters["restart_dir"])
        # input()
        driver_local = DMRGDriver(
            stack_mem=dmrg_parameters["stack_mem"],
            scratch=dmrg_parameters["temp_dir"],
            clean_scratch=True,  # Default value
            restart_dir=dmrg_parameters["restart_dir"],
            n_threads=dmrg_parameters["num_threads"],
            # n_mkl_threads=n_mkl_threads,  # Default value is 1
            symm_type=SymmetryTypes.SZ,
            mpi=None,  # Default value
            stack_mem_ratio=dmrg_parameters["stack_mem_ratio"],  # Default value 0.4
            fp_codec_cutoff=1e-16,  # Default value 1e-16
        )
        driver_local.initialize_system(
            n_sites=n_sites, n_elec=n_elec, spin=spin, orb_sym=orb_sym
        )
        mpo = driver_local.get_qc_mpo(
            h1e=one_body_tensor,
            g2e=two_body_tensor_factor_half,
            ecore=ecore,
            iprint=iprint,
        )
        ket = driver_local.get_random_mps(
            tag="orbital_ordering", bond_dim=bond_dim, nroots=1
        )
        energy = driver_local.dmrg(
            mpo,
            ket,
            n_sweeps=20,
            bond_dims=[bond_dim] * 9,
            noises=[1e-4] * 4 + [1e-5] * 4 + [0],
            thrds=[1e-10] * 9,
            iprint=1,
        )
        log.debug("Approx Orbital Reordering DMRG energy = %20.15f" % energy)
        minfo_orig = driver_local.get_orbital_interaction_matrix(ket)

        idx = driver_local.orbital_reordering_interaction_matrix(
            minfo_orig, method="fiedler"
        )
        h1e = one_body_tensor[idx][:, idx]
        g2e = two_body_tensor_factor_half[idx][:, idx][:, :, idx][:, :, :, idx]
        orb_sym_reordered = np.array(orb_sym)[idx]
        one_body_tensor_reordered = h1e
        two_body_tensor_factor_half_reordered = g2e
        reordering_indices = idx
        reorder_output_dict = {
            "minfo_orig": minfo_orig,
            "reordering_bond_dim": bond_dim,
        }

    else:
        raise ValueError(f"Invalid reordering method: {reordering_method}")

    return (
        one_body_tensor_reordered,
        two_body_tensor_factor_half_reordered,
        orb_sym_reordered,
        reordering_indices,
        reorder_output_dict,
    )


def calc_v_score(
    ket,
    hamiltonian_mpo,
    num_electrons,
    driver,
    # orbital_energies,
    core_energy,
    hf_energy,
    # dmrg_parameters,
    # symmetry_type,
    # n_sites,
    # n_elec,
    # spin,
    # orb_sym,
    # one_body_tensor,
    # two_body_tensor_factor_half,
    # ecore,
    # iprint,
    ket_energy=None,
    init_ket=None,
    init_ket_energy=None,
):
    """Get the V-score of the optimized MPS.
    V-score = num_electrons*(Var E_opt) / (E_opt - E_oo)^2
    where E_opt is the optimized energy, E_oo is the reference energy,
    and Var E_opt is the variance of the optimized energy.
    E_oo will be either the Hartee-Fock energy or the energy of the initial MPS.

    Args:
        ket (MPS): The optimized MPS.
        hamiltonian_mpo (MPO): The Hamiltonian MPO.
        num_electrons (int): The number of electrons.
        driver (DMRGDriver): The DMRG driver.
        init_ket (MPS): The initial (i.e. non-optimized) MPS.
        orbital_energies (np.ndarray): The orbital energies.
        core_energy (float): The core energy.
        ket_energy (float, optional): The energy of the optimized MPS. Defaults to None.
        init_ket_energy (float, optional): The energy of the initial MPS. Defaults to None.
    """
    if ket_energy is None:
        ket_energy_local = driver.expectation(ket, hamiltonian_mpo, ket)

    if init_ket_energy is None and init_ket is not None:
        init_ket_energy = driver.expectation(init_ket, hamiltonian_mpo, init_ket)

    elif init_ket_energy is None and init_ket is None:
        raise ValueError("init_ket_energy and init_ket cannot both be None.")

    # Remove core energy to reduce numerical error
    init_ket_energy_local = init_ket_energy - core_energy
    ket_energy_local = ket_energy - core_energy

    #### CALCULATE VARIANCE ####
    # We get variance as <ket|(H-E_ket)^2|ket>
    # We do this instead of <ket|H^2|ket> - E_ket^2 to eliminate numerical error
    # from subtracting two large numbers

    # Set core energy of mpo to -ket_energy_local to get H-E_ket for the operator
    # Note that the original core energy cancels out, so is left out here
    log.info(f"core_energy: {core_energy}")
    log.info(f"hamiltonian_mpo core_energy: {hamiltonian_mpo.const_e}")
    hamiltonian_mpo.const_e = -1 * ket_energy_local
    log.info(f"New hamiltonian_mpo core_energy: {hamiltonian_mpo.const_e}")

    h_min_e_ket = driver.copy_mps(ket, tag="h_ket_optimized")
    h_min_e_ket_norm = driver.multiply(  # (H-E_ket)|ket>
        bra=h_min_e_ket,  # Take the orginal ket as the guess for the bra
        mpo=hamiltonian_mpo,
        ket=ket,
        n_sweeps=30,
        tol=1e-10,
        bond_dims=None,
        bra_bond_dims=None,  # Keep the bond dims the same as the ket
        # bra_bond_dims=[30*ket.info.bond_dim]*300,
        noises=None,
        noise_mpo=None,
        # thrds=[1e-15]*300,#None,
        thrds=None,
        left_mpo=None,
        cutoff=1e-24,
        linear_max_iter=4000,
        linear_rel_conv_thrd=0.0,
        proj_mpss=None,
        proj_weights=None,
        proj_bond_dim=-1,
        solver_type=None,
        right_weight=0.0,
        iprint=3,
        kernel=None,
    )

    variance = h_min_e_ket_norm**2

    v_score_numerator = num_electrons * variance
    deviation_init_ket = ket_energy_local - init_ket_energy_local

    v_score_init_ket = v_score_numerator / (deviation_init_ket) ** 2

    #     # hf_energy
    #     hf_energy = get_bd_1_dmrg_energy(
    #     dmrg_parameters,
    #     symmetry_type,
    #     n_sites,
    #     n_elec,
    #     spin,
    #     orb_sym,
    #     one_body_tensor,
    #     two_body_tensor_factor_half,
    #     ecore,
    #     iprint,
    # )
    # indices = np.argsort(orbital_energies)
    # log.info(f"indices: {indices}")
    # log.info(f"orbital_energies: {orbital_energies}")
    # # dets = "2" * (num_electrons // 2) + "+" * (num_electrons % 2) + "0" * (
    # #     len(orbital_energies) - (num_electrons // 2 + num_electrons % 2)
    # # )
    # dets = "0" * len(orbital_energies)
    # elec_count = num_electrons
    # for iiter, index_value in enumerate(indices):
    #     if elec_count > 1:
    #         # dets[index_value] = "2"
    #         dets = dets[:index_value] + "2" + dets[index_value + 1 :]
    #         elec_count -= 2
    #     elif elec_count == 1:
    #         # dets[index_value] = "+"
    #         dets = dets[:index_value] + "a" + dets[index_value + 1 :]
    #         elec_count -= 1
    #     else:
    #         log.info(f"elec_count: {elec_count}")
    #         break

    # log.info(f"num_electrons: {num_electrons}")
    # log.info(f"dets: {dets}")

    # hf_mps = driver.get_mps_from_csf_coefficients(
    #     dets=[dets],
    #     dvals=[1.0],
    #     tag="hf",
    #     # dot=2,
    #     # target=None,
    #     # full_fci=True,
    #     # left_vacuum=None,
    #     # casci_ncore=0,
    #     # casci_nvirt=0,
    #     # casci_mask=None,
    #     # mrci_order=0,
    #     # iprint=1,
    # )
    # log.info(f"hamiltonian_mpo core_energy: {hamiltonian_mpo.const_e}")
    # hamiltonian_mpo.const_e = 0.0
    # log.info(f"New hamiltonian_mpo core_energy: {hamiltonian_mpo.const_e}")
    # hf_energy_expectation = driver.expectation(hf_mps, hamiltonian_mpo, hf_mps)
    # log.info(f"hf_energy_expectation: {hf_energy_expectation}")

    # # # Double the mo_energy array to account for alpha and beta electrons
    # # spin_orbital_energies = np.array(orbital_energies)
    # # spin_orbital_energies = np.concatenate(
    # #     (spin_orbital_energies, spin_orbital_energies)
    # # )
    # # # Sort from lowest to highest energy
    # # spin_orbital_energies = np.sort(spin_orbital_energies)
    # # log.info(f"spin_orbital_energies: {spin_orbital_energies}")

    # # hf_energy = np.sum(spin_orbital_energies[:num_electrons])  # + core_energy

    # hf_energy = hf_energy_expectation

    hf_energy_local = hf_energy - core_energy
    deviation_hf = ket_energy_local - hf_energy_local
    v_score_hartree_fock = v_score_numerator / (deviation_hf) ** 2

    # Reset core energy of mpo
    hamiltonian_mpo.const_e = core_energy
    log.info(f"Reset hamiltonian_mpo core_energy: {hamiltonian_mpo.const_e}")

    log.info(f"ket_energy_local: {ket_energy_local}")
    log.info(f"ket_energy: {ket_energy}")
    log.info(f"init_ket_energy_local: {init_ket_energy_local}")
    log.info(f"init_ket_energy: {init_ket_energy}")
    log.info(f"hf_energy: {hf_energy}")
    log.info(f"hf_energy_local: {hf_energy_local}")
    # log.info(f"hf_energy_expectation: {hf_energy_expectation}")
    log.info(f"deviation_hf: {deviation_hf}")
    log.info(f"v_score_hartree_fock: {v_score_hartree_fock}")
    log.info(f"deviation_init_ket: {deviation_init_ket}")
    log.info(f"v_score_init_ket: {v_score_init_ket}")
    log.info(f"h_min_e_ket_norm: {h_min_e_ket_norm}")
    log.info(f"ket_energy_local^2: {ket_energy_local**2}")
    log.info(f"variance: {variance}")
    log.info(f"v_score_numerator: {v_score_numerator}")

    return (
        h_min_e_ket_norm,
        variance,
        v_score_numerator,
        deviation_init_ket,
        v_score_init_ket,
        hf_energy,
        deviation_hf,
        v_score_hartree_fock,
    )


def get_bd_1_dmrg_energy(
    dmrg_parameters,
    symmetry_type,
    n_sites,
    n_elec,
    spin,
    orb_sym,
    one_body_tensor,
    two_body_tensor_factor_half,
    ecore,
    iprint,
):
    bond_dim = 1
    # raise NotImplementedError(
    #     "The 'fiedler, interaction matrix' reordering method is not implemented."
    # )
    log.debug("Calc energy for BD = 1 MPS")
    # approx DMRG to get orbital_interaction_matrix

    # print(dmrg_parameters["restart_dir"])
    # input()
    driver_local = DMRGDriver(
        stack_mem=dmrg_parameters["stack_mem"],
        scratch=dmrg_parameters["temp_dir"],
        clean_scratch=True,  # Default value
        restart_dir="./tmp_dir",
        n_threads=dmrg_parameters["num_threads"],
        # n_mkl_threads=n_mkl_threads,  # Default value is 1
        symm_type=symmetry_type,
        mpi=None,  # Default value
        stack_mem_ratio=dmrg_parameters["stack_mem_ratio"],  # Default value 0.4
        fp_codec_cutoff=1e-16,  # Default value 1e-16
    )
    driver_local.initialize_system(
        n_sites=n_sites, n_elec=n_elec, spin=spin, orb_sym=orb_sym
    )
    mpo = driver_local.get_qc_mpo(
        h1e=one_body_tensor,
        g2e=two_body_tensor_factor_half,
        ecore=ecore,
        iprint=iprint,
    )
    ket = driver_local.get_random_mps(tag="bd_1", bond_dim=bond_dim, nroots=1)
    energy = driver_local.dmrg(
        mpo,
        ket,
        n_sweeps=50,
        bond_dims=[bond_dim] * 30,
        noises=[1e-4] * 4 + [1e-5] * 4 + [0] * (30 - 8),
        thrds=[1e-10] * 30,
        iprint=1,
    )
    log.info("BD = 1 DMRG energy = %20.15f" % energy)
    sweep_bond_dims, sweep_max_discarded_weight, sweep_energies = (
        driver_local.get_dmrg_results()
    )
    # Assert that the last change in sweep energy is less than 1e-6
    sweep_energy_change = np.abs(sweep_energies[-1] - sweep_energies[-2])
    log.info(f"sweep_energies: {sweep_energies}")
    log.info(f"last_sweep_energy_change: {sweep_energy_change}")
    assert (
        sweep_energy_change < 1e-6
    ), f"BD = 1 DMRG energy did not converge: {sweep_energies[-1]} - {sweep_energies[-2]} = {sweep_energies[-1] - sweep_energies[-2]}"
    return energy
