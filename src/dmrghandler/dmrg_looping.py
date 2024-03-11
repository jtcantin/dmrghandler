import logging
import time

import numpy as np

import dmrghandler.energy_extrapolation as energy_extrapolation
import dmrghandler.qchem_dmrg_calc as qchem_dmrg_calc

log = logging.getLogger(__name__)


def dmrg_central_loop(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    dmrg_parameters: dict,
    max_bond_dimension: int,
    max_time_limit_sec: int,
    min_energy_change_hartree: float,
    verbosity: int = 0,
):
    wall_time_start_ns = time.perf_counter_ns()
    cpu_time_start_ns = time.process_time_ns()

    # Initial two calculations
    # Run DMRG
    dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        dmrg_parameters=dmrg_parameters,
        verbosity=verbosity,
    )

    energies_dmrg = np.array(dmrg_results["dmrg_ground_state_energy"])
    discarded_weights = np.array(dmrg_results["dmrg_discarded_weight"])
    bond_dims_used = np.array(dmrg_parameters["sweep_schedule_bond_dims"][-1])

    # Update bond dimension
    sweep_schedule_bond_dims = dmrg_parameters["sweep_schedule_bond_dims"]
    init_state_bond_dimension = dmrg_parameters["init_state_bond_dimension"]

    sweep_schedule_bond_dims, init_state_bond_dimension = update_bond_dim(
        sweep_schedule_bond_dims, init_state_bond_dimension
    )

    dmrg_parameters["sweep_schedule_bond_dims"] = sweep_schedule_bond_dims
    dmrg_parameters["init_state_bond_dimension"] = init_state_bond_dimension

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
    past_parameters: np.ndarray = None,
    verbosity: int = 0,
):
    # Update bond dimension
    sweep_schedule_bond_dims = dmrg_parameters["sweep_schedule_bond_dims"]
    init_state_bond_dimension = dmrg_parameters["init_state_bond_dimension"]

    sweep_schedule_bond_dims, init_state_bond_dimension = update_bond_dim(
        sweep_schedule_bond_dims, init_state_bond_dimension
    )

    dmrg_parameters["sweep_schedule_bond_dims"] = sweep_schedule_bond_dims
    dmrg_parameters["init_state_bond_dimension"] = init_state_bond_dimension

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

    # Save data
    result_storage_dict = {
        "sweep_schedule_bond_dims": sweep_schedule_bond_dims,
        "init_state_bond_dimension": init_state_bond_dimension,
        "energy_estimated": energy_estimated,
        "fit_parameters": fit_parameters,
        "R_squared": R_squared,
        "fit_energy_replaced_by_dmrg": fit_energy_replaced_by_dmrg,
    }
    past_energies_dmrg = energies_dmrg
    past_discarded_weights = discarded_weights
    save_data(dmrg_results, result_storage_dict)

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
    new_sweep_schedule_bond_dims = np.ceil(
        np.array(sweep_schedule_bond_dims) * 1.1
    ).astype(np.int_)

    new_init_state_bond_dimension = int(np.ceil(init_state_bond_dimension * 1.1))
    return new_sweep_schedule_bond_dims, new_init_state_bond_dimension


def save_data(dmrg_results, result_storage_dict):
    pass
