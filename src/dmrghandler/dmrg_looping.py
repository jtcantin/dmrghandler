import gc
import logging
import time
from pathlib import Path

import numpy as np

import dmrghandler.energy_extrapolation as energy_extrapolation
import dmrghandler.hdf5_io as hdf5_io
import dmrghandler.qchem_dmrg_calc as qchem_dmrg_calc

log = logging.getLogger(__name__)


def dmrg_central_loop(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    dmrg_parameters: dict,
    max_bond_dimension: int,
    max_time_limit_sec: int,
    min_energy_change_hartree: float,
    main_storage_file_path: str,
    verbosity: int = 0,
):
    wall_time_start_ns = time.perf_counter_ns()
    cpu_time_start_ns = time.process_time_ns()
    base_mps_storage_folder = Path(dmrg_parameters["restart_dir"])

    # Initial two calculations
    # Run DMRG
    dmrg_parameters["restart_dir"] = str(base_mps_storage_folder / "dmrg_first_calc")
    dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        dmrg_parameters=dmrg_parameters,
        verbosity=verbosity,
    )

    energies_dmrg = np.array(dmrg_results["dmrg_ground_state_energy"])
    discarded_weights = np.array(dmrg_results["dmrg_discarded_weight"])
    bond_dims_used = np.array(dmrg_parameters["sweep_schedule_bond_dims"][-1])

    # Get initial_ket and ket_optimized and convert them to a representation of tensors
    # See https://github.com/block-hczhai/block2-preview/blob/b4eb5bfe1020ffed6c5e90db444dd878052dd66d/pyblock2/algebra/core.py#L483
    # Perhaps use a dict that has a partial view of dmrg_results and then add representation
    dmrg_results_saveable = {}
    for key, value in dmrg_results.items():
        if key in ["ket_optimized", "initial_ket"]:
            dmrg_results_saveable[key] = repr(value.tensors)
        else:
            dmrg_results_saveable[key] = value

    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=dmrg_parameters,
        access_mode="a",
        group=f"dmrg_first_calc/dmrg_parameters",
        overwrite=False,
    )
    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=dmrg_results_saveable,
        access_mode="a",
        group=f"dmrg_first_calc/dmrg_results",
        overwrite=False,
    )
    del dmrg_results_saveable  # Clear memory
    gc.collect()  # Collect garbage
    dmrg_parameters["restart_dir"] = str(base_mps_storage_folder)

    # Update bond dimension
    sweep_schedule_bond_dims = dmrg_parameters["sweep_schedule_bond_dims"]
    init_state_bond_dimension = dmrg_parameters["init_state_bond_dimension"]

    sweep_schedule_bond_dims, init_state_bond_dimension = update_bond_dim(
        sweep_schedule_bond_dims, init_state_bond_dimension
    )

    dmrg_parameters["sweep_schedule_bond_dims"] = sweep_schedule_bond_dims
    dmrg_parameters["init_state_bond_dimension"] = init_state_bond_dimension
    dmrg_parameters["restart_dir"] = str(base_mps_storage_folder / "dmrg_second_calc")

    # Run DMRG
    dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        dmrg_parameters=dmrg_parameters,
        verbosity=verbosity,
    )

    past_energies_dmrg = np.hstack(
        [energies_dmrg, dmrg_results["dmrg_ground_state_energy"]]
    )
    past_discarded_weights = np.hstack(
        [discarded_weights, dmrg_results["dmrg_discarded_weight"]]
    )
    bond_dims_used = np.hstack(
        [bond_dims_used, dmrg_parameters["sweep_schedule_bond_dims"][-1]]
    )

    # Get initial_ket and ket_optimized and convert them to a representation of tensors
    # See https://github.com/block-hczhai/block2-preview/blob/b4eb5bfe1020ffed6c5e90db444dd878052dd66d/pyblock2/algebra/core.py#L483
    # Perhaps use a dict that has a partial view of dmrg_results and then add representation
    dmrg_results_saveable = {}
    for key, value in dmrg_results.items():
        if key in ["ket_optimized", "initial_ket"]:
            dmrg_results_saveable[key] = repr(value.tensors)
        else:
            dmrg_results_saveable[key] = value

    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=dmrg_parameters,
        access_mode="a",
        group=f"dmrg_second_calc/dmrg_parameters",
        overwrite=False,
    )
    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=dmrg_results_saveable,
        access_mode="a",
        group=f"dmrg_second_calc/dmrg_results",
        overwrite=False,
    )
    del dmrg_results_saveable  # Clear memory
    gc.collect()  # Collect garbage
    dmrg_parameters["restart_dir"] = str(base_mps_storage_folder)

    energy_change = np.inf
    wall_time_loop_ns = time.perf_counter_ns() - wall_time_start_ns
    # Loop
    loop_entry_count = 0
    past_parameters = None
    unmodified_fit_parameters_list = []
    fit_parameters_list = []
    while (
        np.abs(energy_change) > min_energy_change_hartree
        and dmrg_parameters["sweep_schedule_bond_dims"][-1] < max_bond_dimension
        and wall_time_loop_ns < max_time_limit_sec * 1e9
    ):
        loop_entry_count += 1
        (
            dmrg_results,
            energy_estimated,
            fit_parameters,
            unmodified_fit_parameters,
            R_squared,
            past_energies_dmrg,
            past_discarded_weights,
            result_storage_dict,
        ) = dmrg_loop_function(
            one_body_tensor=one_body_tensor,
            two_body_tensor=two_body_tensor,
            dmrg_parameters=dmrg_parameters,
            past_energies_dmrg=past_energies_dmrg,
            past_discarded_weights=past_discarded_weights,
            loop_entry_count=loop_entry_count,
            main_storage_file_path=main_storage_file_path,
            past_parameters=past_parameters,
            verbosity=verbosity,
        )
        wall_time_loop_ns = time.perf_counter_ns() - wall_time_start_ns
        cpu_time_loop_ns = time.process_time_ns() - cpu_time_start_ns
        energy_change = past_energies_dmrg[-1] - past_energies_dmrg[-2]
        discard_weight_change = past_discarded_weights[-1] - past_discarded_weights[-2]
        bond_dims_used = np.hstack(
            [bond_dims_used, result_storage_dict["sweep_schedule_bond_dims"][-1]]
        )
        past_parameters = fit_parameters
        unmodified_fit_parameters_list.append(unmodified_fit_parameters)
        fit_parameters_list.append(fit_parameters)

    if np.abs(energy_change) < min_energy_change_hartree:
        finish_reason = f"Energy change below threshold, limit {min_energy_change_hartree}, achieved {energy_change}"
    elif dmrg_parameters["sweep_schedule_bond_dims"][-1] >= max_bond_dimension:
        finish_reason = f"Maximum bond dimension reached, limit {max_bond_dimension}, achieved {dmrg_parameters['sweep_schedule_bond_dims'][-1]}"
    elif wall_time_loop_ns >= max_time_limit_sec * 1e9:
        finish_reason = f"Time limit reached, limit {max_time_limit_sec} s, achieved {wall_time_loop_ns / 1e9} s"
    loop_results = {
        "energy_estimated": energy_estimated,
        "fit_parameters": fit_parameters,
        "R_squared": R_squared,
        "energies_dmrg": past_energies_dmrg,
        "discarded_weights": past_discarded_weights,
        "result_storage_dict": result_storage_dict,
        "wall_time_loop_ns": wall_time_loop_ns,
        "cpu_time_loop_ns": cpu_time_loop_ns,
        "energy_change": energy_change,
        "discard_weight_change": discard_weight_change,
        "finish_reason": finish_reason,
        "past_energies_dmrg": past_energies_dmrg,
        "past_discarded_weights": past_discarded_weights,
        "bond_dims_used": bond_dims_used,
        "loop_entry_count": loop_entry_count,
        "unmodified_fit_parameters_list": unmodified_fit_parameters_list,
        "fit_parameters_list": fit_parameters_list,
        # "final_dmrg_results": dmrg_results,
    }

    return loop_results


def dmrg_loop_function(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    dmrg_parameters: dict,
    past_energies_dmrg: np.ndarray,
    past_discarded_weights: np.ndarray,
    loop_entry_count: int,
    main_storage_file_path: str,
    past_parameters: np.ndarray = None,
    verbosity: int = 0,
):
    base_mps_storage_folder = Path(dmrg_parameters["restart_dir"])
    # Update bond dimension
    sweep_schedule_bond_dims = dmrg_parameters["sweep_schedule_bond_dims"]
    init_state_bond_dimension = dmrg_parameters["init_state_bond_dimension"]

    sweep_schedule_bond_dims, init_state_bond_dimension = update_bond_dim(
        sweep_schedule_bond_dims, init_state_bond_dimension
    )

    dmrg_parameters["sweep_schedule_bond_dims"] = sweep_schedule_bond_dims
    dmrg_parameters["init_state_bond_dimension"] = init_state_bond_dimension
    dmrg_parameters["restart_dir"] = str(
        base_mps_storage_folder / f"dmrg_loop_{loop_entry_count:03d}"
    )

    # Run DMRG
    dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        dmrg_parameters=dmrg_parameters,
        verbosity=verbosity,
    )

    # Perform extrapolation
    energies_dmrg = np.append(
        past_energies_dmrg, dmrg_results["dmrg_ground_state_energy"]
    )
    discarded_weights = np.append(
        past_discarded_weights, dmrg_results["dmrg_discarded_weight"]
    )
    fit_result_obj, energy_estimated, fit_parameters, R_squared = (
        energy_extrapolation.dmrg_energy_extrapolation(
            energies_dmrg=energies_dmrg,
            independent_vars=discarded_weights,
            extrapolation_type="discarded_weight",
            past_parameters=past_parameters,
            verbosity=verbosity,
        )
    )
    # If DMRG energy is below estimated energy, then use DMRG energy as DMRG is variational
    fit_energy_replaced_by_dmrg = False
    unmodified_fit_parameters = fit_parameters.copy()
    if dmrg_results["dmrg_ground_state_energy"] < energy_estimated:
        energy_estimated = dmrg_results["dmrg_ground_state_energy"]
        fit_parameters[-1] = energy_estimated
        fit_energy_replaced_by_dmrg = True

    # Get initial_ket and ket_optimized and convert them to a representation of tensors
    # See https://github.com/block-hczhai/block2-preview/blob/b4eb5bfe1020ffed6c5e90db444dd878052dd66d/pyblock2/algebra/core.py#L483
    # Perhaps use a dict that has a partial view of dmrg_results and then add representation
    dmrg_results_saveable = {}
    for key, value in dmrg_results.items():
        if key in ["ket_optimized", "initial_ket"]:
            continue
            # dmrg_results_saveable[key+"_storage_folder"] = str(base_mps_storage_folder / f"dmrg_loop_{loop_entry_count:03d}")
            # log.info(
            #     f"Saved {key} as a representation of tensors.------------------------------------"
            # )
            # log.info(f"repr(value.tensors): {repr(value.tensors)}")
            # log.info(
            #     f"repr(value.tensors): {[repr(tensor) for tensor in value.tensors]}"
            # )
            # log.info(
            #     f"repr(value.tensors): {[repr(block) for tensor in value.tensors for block in tensor.blocks]}"
            # )
        else:
            dmrg_results_saveable[key] = value

    # Save data
    result_storage_dict = {
        "sweep_schedule_bond_dims": sweep_schedule_bond_dims,
        "init_state_bond_dimension": init_state_bond_dimension,
        "energy_estimated": energy_estimated,
        "fit_parameters": fit_parameters,
        "R_squared": R_squared,
        "fit_energy_replaced_by_dmrg": fit_energy_replaced_by_dmrg,
        "ket_storage_dir": str(
            base_mps_storage_folder / f"dmrg_loop_{loop_entry_count:03d}"
        ),
    }
    past_energies_dmrg = energies_dmrg
    past_discarded_weights = discarded_weights

    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=result_storage_dict,
        access_mode="a",
        group=f"dmrg_loop_{loop_entry_count:03d}/result_storage_dict",
        overwrite=False,
    )
    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=dmrg_parameters,
        access_mode="a",
        group=f"dmrg_loop_{loop_entry_count:03d}/dmrg_parameters",
        overwrite=False,
    )
    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=dmrg_results_saveable,
        access_mode="a",
        group=f"dmrg_loop_{loop_entry_count:03d}/dmrg_results",
        overwrite=False,
    )
    del dmrg_results_saveable  # Clear memory
    gc.collect()  # Collect garbage
    dmrg_parameters["restart_dir"] = str(base_mps_storage_folder)

    # Return results
    return (
        dmrg_results,
        energy_estimated,
        fit_parameters,
        unmodified_fit_parameters,
        R_squared,
        past_energies_dmrg,
        past_discarded_weights,
        result_storage_dict,
    )


def update_bond_dim(sweep_schedule_bond_dims, init_state_bond_dimension):

    # Increase by 10%
    new_sweep_schedule_bond_dim = np.ceil(sweep_schedule_bond_dims[-1] * 1.1).astype(
        np.int_
    )
    num_sweeps = len(sweep_schedule_bond_dims)
    half_length = num_sweeps // 2
    new_sweep_schedule_bond_dims = [new_sweep_schedule_bond_dim // 2] * half_length + [
        new_sweep_schedule_bond_dim
    ] * (num_sweeps - half_length)

    new_init_state_bond_dimension = int(np.ceil(init_state_bond_dimension * 1.1))
    return new_sweep_schedule_bond_dims, new_init_state_bond_dimension
