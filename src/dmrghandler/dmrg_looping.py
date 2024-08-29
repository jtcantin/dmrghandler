import gc
import inspect
import logging
import os
import shutil

# import shutil
import time
from pathlib import Path

import numpy as np
import pyblock2.tools
from block2 import SU2 as block2_SU2
from block2 import SZ as block2_SZ

# from memory_profiler import profile as mem_profile
from memory_profiler import memory_usage

import dmrghandler.energy_extrapolation as energy_extrapolation
import dmrghandler.hdf5_io as hdf5_io
import dmrghandler.qchem_dmrg_calc as qchem_dmrg_calc
from dmrghandler.profiling import print_system_info

log = logging.getLogger(__name__)


def dmrg_central_loop_mem_tracking(*args, track_mem=False, **kwargs):
    if track_mem:
        # return mem_profile(dmrg_central_loop)(*args, track_mem=track_mem, **kwargs)
        mem_data, retval = memory_usage(
            proc=(dmrg_central_loop, args, kwargs),
            interval=0.1,
            timeout=None,
            timestamps=True,
            include_children=True,
            multiprocess=False,
            max_usage=False,
            retval=True,
            stream=None,
            backend=None,
            max_iterations=None,
        )

        log.info(f"Memory usage (MiB, Timestamp (sec)): {mem_data}")
        # print(f"Memory usage: {mem_data}")
        memory_MiB_list = []
        time_stamp_sec_list = []
        for mem, time_stamp in mem_data:
            memory_MiB_list.append(mem)
            time_stamp_sec_list.append(time_stamp)
        memory_MiB_array = np.array(memory_MiB_list)
        time_stamp_sec_array = np.array(time_stamp_sec_list)

        memory_tracking_data = {
            "memory_MiB_array": memory_MiB_array,
            "time_stamp_array": time_stamp_sec_array,
            "max_memory_MiB": np.amax(memory_MiB_array),
        }
        main_storage_folder_path = kwargs["main_storage_folder_path"]
        main_storage_folder_path = Path(main_storage_folder_path)
        main_storage_file_path = main_storage_folder_path / "dmrg_results.hdf5"
        hdf5_io.save_many_variables_to_hdf5(
            hdf5_filepath=main_storage_file_path,
            variables=memory_tracking_data,
            access_mode="a",
            group=f"memory_tracking_data",
            overwrite=False,
        )
        return retval
    else:
        return dmrg_central_loop(*args, track_mem=track_mem, **kwargs)


def dmrg_central_loop(
    one_body_tensor: np.ndarray,
    two_body_tensor: np.ndarray,
    dmrg_parameters: dict,
    max_bond_dimension: int,
    max_time_limit_sec: int,
    min_energy_change_hartree: float,
    main_storage_folder_path: str,
    verbosity: int = 0,
    mps_final_storage_path="./",
    track_mem=False,
):
    wall_time_start_ns = time.perf_counter_ns()
    cpu_time_start_ns = time.process_time_ns()
    main_storage_folder_path = Path(main_storage_folder_path)
    main_storage_file_path = main_storage_folder_path / "dmrg_results.hdf5"
    # Make directory if it does not exist
    main_storage_folder_path.mkdir(parents=True, exist_ok=True)

    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )

    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables={"parent_folder_name": str(main_storage_file_path.parent.name)},
        access_mode="a",
        group=None,
        overwrite=False,
    )

    if dmrg_parameters["symmetry_type"] == "SZ":
        pyblock2.tools.init(block2_SZ)
        log.info("Initialized pyblock2.tools with SZ symmetry.")
    elif dmrg_parameters["symmetry_type"] == "SU(2)":
        pyblock2.tools.init(block2_SU2)
        log.info("Initialized pyblock2.tools with SU2 symmetry.")
    else:
        raise ValueError(
            f"symmetry_type {dmrg_parameters['symmetry_type']} not recognized"
        )

    # if "restart_dir" in dmrg_parameters and dmrg_parameters["restart_dir"] is not None:
    #     log.warning(
    #         f"restart_dir is ignored in dmrg_central_loop, MPSs will be saved in { main_storage_folder_path / 'mps_storage'}"
    #     )
    #     dmrg_parameters["restart_dir"] = None

    # Initial two calculations
    # Run DMRG
    log.info("Starting first preloop calc")

    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    wall_first_preloop_start_ns = time.perf_counter_ns()
    cpu_first_preloop_start_ns = time.process_time_ns()
    dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc_mem_tracking(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        dmrg_parameters=dmrg_parameters,
        verbosity=verbosity,
        track_mem=track_mem,
    )
    log.info("Finished first preloop calc")
    wall_first_preloop_end_ns = time.perf_counter_ns()
    cpu_first_preloop_end_ns = time.process_time_ns()

    wall_first_preloop_ns = wall_first_preloop_end_ns - wall_first_preloop_start_ns
    cpu_first_preloop_ns = cpu_first_preloop_end_ns - cpu_first_preloop_start_ns

    log.info(f"wall_first_preloop_s: {wall_first_preloop_ns/1e9}")
    log.info(f"cpu_first_preloop_s: {cpu_first_preloop_ns/1e9}")
    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    log.info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    energies_dmrg = np.array(dmrg_results["dmrg_ground_state_energy"])
    discarded_weights = np.array(dmrg_results["dmrg_discarded_weight"])
    bond_dims_used = np.array(dmrg_parameters["sweep_schedule_bond_dims"][-1])
    wall_dmrg_whole_calc_times_s = np.array(wall_first_preloop_ns / 1e9)
    cpu_dmrg_whole_calc_times_s = np.array(cpu_first_preloop_ns / 1e9)
    wall_dmrg_optimization_times_s = np.array(
        dmrg_results["wall_dmrg_optimization_time_s"]
    )
    cpu_dmrg_optimization_times_s = np.array(
        dmrg_results["cpu_dmrg_optimization_time_s"]
    )
    log.info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    save_dmrg_results(
        dmrg_results=dmrg_results,
        dmrg_parameters=dmrg_parameters,
        main_storage_file_path=main_storage_file_path,
        calc_id_str="first_preloop_calc",
        mps_final_storage_path=mps_final_storage_path,
    )
    log.info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )

    if "do_single_calc" in dmrg_parameters.keys() and dmrg_parameters["do_single_calc"]:
        loop_results = {
            "energy_estimated": energies_dmrg,  # energy_estimated,
            "fit_parameters": [0, 0, 0],  # fit_parameters,
            "R_squared": 0,  # R_squared,
            "energies_dmrg": energies_dmrg,  # past_energies_dmrg,
            "discarded_weights": discarded_weights,  # past_discarded_weights,
            # "result_storage_dict": result_storage_dict,
            "wall_time_loop_s": wall_first_preloop_ns / 1e9,
            "cpu_time_loop_s": cpu_first_preloop_ns / 1e9,
            "energy_change": 9999999,  # energy_change,
            "discard_weight_change": 9999999,  # discard_weight_change,
            "finish_reason": "Single calc",  # finish_reason,
            "past_energies_dmrg": [energies_dmrg],  # past_energies_dmrg,
            "past_discarded_weights": [discarded_weights],  # past_discarded_weights,
            "bond_dims_used": [bond_dims_used],
            "loop_entry_count": 0,  # loop_entry_count,
            "unmodified_fit_parameters_list": [
                [0, 0, 0]
            ],  # unmodified_fit_parameters_list,
            "fit_parameters_list": [[0, 0, 0]],  # fit_parameters_list,
            # "final_dmrg_results": dmrg_results,
            "wall_dmrg_whole_calc_times_s": wall_dmrg_whole_calc_times_s,
            "cpu_dmrg_whole_calc_times_s": cpu_dmrg_whole_calc_times_s,
            "wall_dmrg_optimization_times_s": wall_dmrg_optimization_times_s,
            "cpu_dmrg_optimization_times_s": cpu_dmrg_optimization_times_s,
        }
        hdf5_io.save_many_variables_to_hdf5(
            hdf5_filepath=main_storage_file_path,
            variables=loop_results,
            access_mode="a",
            group=f"first_preloop_calc/loop_results",
            overwrite=False,
        )

        result_storage_dict = {
            "sweep_schedule_bond_dims": dmrg_parameters[
                "sweep_schedule_bond_dims"
            ],  # sweep_schedule_bond_dims,
            "init_state_bond_dimension": dmrg_parameters[
                "init_state_bond_dimension"
            ],  # init_state_bond_dimension,
            "energy_estimated": energies_dmrg,  # energy_estimated,
            "fit_parameters": [0, 0, 0],  # fit_parameters,
            "R_squared": 0,  # R_squared,
            "fit_energy_replaced_by_dmrg_bool": False,  # fit_energy_replaced_by_dmrg,
            "wall_extrapolation_s": 0,  # wall_extrapolation_ns / 1e9,
            "cpu_extrapolation_s": 0,  # cpu_extrapolation_ns / 1e9,
        }

        loop_results["result_storage_dict"] = result_storage_dict

        print_system_info(
            f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
        )

        return loop_results

    after_first_preloop_ns = time.perf_counter_ns() - wall_time_start_ns
    if after_first_preloop_ns > max_time_limit_sec * 1e9:
        raise Exception(
            f"First preloop calc took longer than time limit {max_time_limit_sec} s"
        )
    log.info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    # Update bond dimension
    sweep_schedule_bond_dims = dmrg_parameters["sweep_schedule_bond_dims"]
    init_state_bond_dimension = dmrg_parameters["init_state_bond_dimension"]

    sweep_schedule_bond_dims, init_state_bond_dimension = update_bond_dim(
        sweep_schedule_bond_dims, init_state_bond_dimension
    )

    dmrg_parameters["sweep_schedule_bond_dims"] = sweep_schedule_bond_dims
    dmrg_parameters["init_state_bond_dimension"] = init_state_bond_dimension
    log.info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    # Run DMRG
    log.info("Starting second preloop calc")
    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    wall_second_preloop_start_ns = time.perf_counter_ns()
    cpu_second_preloop_start_ns = time.process_time_ns()
    dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc_mem_tracking(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        dmrg_parameters=dmrg_parameters,
        verbosity=verbosity,
        track_mem=track_mem,
    )

    log.info("Finished second preloop calc")
    wall_second_preloop_end_ns = time.perf_counter_ns()
    cpu_second_preloop_end_ns = time.process_time_ns()

    wall_second_preloop_ns = wall_second_preloop_end_ns - wall_second_preloop_start_ns
    cpu_second_preloop_ns = cpu_second_preloop_end_ns - cpu_second_preloop_start_ns

    log.info(f"wall_second_preloop_s: {wall_second_preloop_ns/1e9}")
    log.info(f"cpu_second_preloop_s: {cpu_second_preloop_ns/1e9}")
    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
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
    wall_dmrg_whole_calc_times_s = np.hstack(
        [wall_dmrg_whole_calc_times_s, wall_second_preloop_ns / 1e9]
    )
    cpu_dmrg_whole_calc_times_s = np.hstack(
        [cpu_dmrg_whole_calc_times_s, cpu_second_preloop_ns / 1e9]
    )
    wall_dmrg_optimization_times_s = np.hstack(
        [wall_dmrg_optimization_times_s, dmrg_results["wall_dmrg_optimization_time_s"]]
    )
    cpu_dmrg_optimization_times_s = np.hstack(
        [cpu_dmrg_optimization_times_s, dmrg_results["cpu_dmrg_optimization_time_s"]]
    )

    save_dmrg_results(
        dmrg_results=dmrg_results,
        dmrg_parameters=dmrg_parameters,
        main_storage_file_path=main_storage_file_path,
        calc_id_str="second_preloop_calc",
        mps_final_storage_path=mps_final_storage_path,
    )

    after_second_preloop_ns = time.perf_counter_ns() - wall_time_start_ns
    if after_second_preloop_ns > max_time_limit_sec * 1e9:
        raise Exception(
            f"Getting to after second preloop calc took longer than time limit {max_time_limit_sec} s"
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
        log.info(f"Starting loop {loop_entry_count}")
        print_system_info(
            f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
        )
        wall_dmrg_loop_start_ns = time.perf_counter_ns()
        cpu_dmrg_loop_start_ns = time.process_time_ns()
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
            move_mps_to_final_storage_path=mps_final_storage_path,
            track_mem=track_mem,
        )
        wall_dmrg_loop_end_ns = time.perf_counter_ns()
        cpu_dmrg_loop_end_ns = time.process_time_ns()

        wall_dmrg_loop_ns = wall_dmrg_loop_end_ns - wall_dmrg_loop_start_ns
        cpu_dmrg_loop_ns = cpu_dmrg_loop_end_ns - cpu_dmrg_loop_start_ns

        wall_time_loop_ns = time.perf_counter_ns() - wall_time_start_ns
        cpu_time_loop_ns = time.process_time_ns() - cpu_time_start_ns
        energy_change = past_energies_dmrg[-1] - past_energies_dmrg[-2]
        discard_weight_change = past_discarded_weights[-1] - past_discarded_weights[-2]
        bond_dims_used = np.hstack(
            [bond_dims_used, result_storage_dict["sweep_schedule_bond_dims"][-1]]
        )
        wall_dmrg_whole_calc_times_s = np.hstack(
            [
                wall_dmrg_whole_calc_times_s,
                (wall_dmrg_loop_ns / 1e9) - result_storage_dict["wall_extrapolation_s"],
            ]
        )
        cpu_dmrg_whole_calc_times_s = np.hstack(
            [
                cpu_dmrg_whole_calc_times_s,
                (cpu_dmrg_loop_ns / 1e9) - result_storage_dict["cpu_extrapolation_s"],
            ]
        )
        wall_dmrg_optimization_times_s = np.hstack(
            [
                wall_dmrg_optimization_times_s,
                dmrg_results["wall_dmrg_optimization_time_s"],
            ]
        )
        cpu_dmrg_optimization_times_s = np.hstack(
            [
                cpu_dmrg_optimization_times_s,
                dmrg_results["cpu_dmrg_optimization_time_s"],
            ]
        )
        past_parameters = fit_parameters
        unmodified_fit_parameters_list.append(unmodified_fit_parameters)
        fit_parameters_list.append(fit_parameters)
        log.info(f"Finished loop {loop_entry_count}")
        log.info(f"wall_dmrg_loop_s: {wall_dmrg_loop_ns/1e9}")
        log.info(f"cpu_dmrg_loop_s: {cpu_dmrg_loop_ns/1e9}")
        print_system_info(
            f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
        )

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
        # "result_storage_dict": result_storage_dict,
        "wall_time_loop_s": wall_time_loop_ns / 1e9,
        "cpu_time_loop_s": cpu_time_loop_ns / 1e9,
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
        "wall_dmrg_whole_calc_times_s": wall_dmrg_whole_calc_times_s,
        "cpu_dmrg_whole_calc_times_s": cpu_dmrg_whole_calc_times_s,
        "wall_dmrg_optimization_times_s": wall_dmrg_optimization_times_s,
        "cpu_dmrg_optimization_times_s": cpu_dmrg_optimization_times_s,
    }
    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=loop_results,
        access_mode="a",
        group=f"dmrg_loop_{loop_entry_count:03d}/loop_results",
        overwrite=False,
    )

    loop_results["result_storage_dict"] = result_storage_dict

    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )

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
    move_mps_to_final_storage_path=None,
    track_mem=False,
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
    dmrg_results = qchem_dmrg_calc.single_qchem_dmrg_calc_mem_tracking(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        dmrg_parameters=dmrg_parameters,
        verbosity=verbosity,
        track_mem=track_mem,
    )

    # Perform extrapolation
    energies_dmrg = np.append(
        past_energies_dmrg, dmrg_results["dmrg_ground_state_energy"]
    )
    discarded_weights = np.append(
        past_discarded_weights, dmrg_results["dmrg_discarded_weight"]
    )
    wall_extrapolation_start_ns = time.perf_counter_ns()
    cpu_extrapolation_start_ns = time.process_time_ns()
    fit_result_obj, energy_estimated, fit_parameters, R_squared = (
        energy_extrapolation.dmrg_energy_extrapolation(
            energies_dmrg=energies_dmrg,
            independent_vars=discarded_weights,
            extrapolation_type="discarded_weight",
            past_parameters=past_parameters,
            verbosity=verbosity,
        )
    )
    wall_extrapolation_ns = time.perf_counter_ns() - wall_extrapolation_start_ns
    cpu_extrapolation_ns = time.process_time_ns() - cpu_extrapolation_start_ns
    log.info(f"wall_extrapolation_s: {wall_extrapolation_ns/1e9}")
    log.info(f"cpu_extrapolation_s: {cpu_extrapolation_ns/1e9}")
    # If DMRG energy is below estimated energy, then use DMRG energy as DMRG is variational
    fit_energy_replaced_by_dmrg = False
    unmodified_fit_parameters = fit_parameters.copy()
    if dmrg_results["dmrg_ground_state_energy"] < energy_estimated:
        energy_estimated = dmrg_results["dmrg_ground_state_energy"]
        fit_parameters[-1] = energy_estimated
        fit_energy_replaced_by_dmrg = True

    # Save data
    save_dmrg_results(
        dmrg_results=dmrg_results,
        dmrg_parameters=dmrg_parameters,
        main_storage_file_path=main_storage_file_path,
        calc_id_str=f"dmrg_loop_{loop_entry_count:03d}",
        mps_final_storage_path=move_mps_to_final_storage_path,
    )

    result_storage_dict = {
        "sweep_schedule_bond_dims": sweep_schedule_bond_dims,
        "init_state_bond_dimension": init_state_bond_dimension,
        "energy_estimated": energy_estimated,
        "fit_parameters": fit_parameters,
        "R_squared": R_squared,
        "fit_energy_replaced_by_dmrg_bool": fit_energy_replaced_by_dmrg,
        "wall_extrapolation_s": wall_extrapolation_ns / 1e9,
        "cpu_extrapolation_s": cpu_extrapolation_ns / 1e9,
    }

    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=result_storage_dict,
        access_mode="a",
        group=f"dmrg_loop_{loop_entry_count:03d}/result_storage_dict",
        overwrite=False,
    )

    past_energies_dmrg = energies_dmrg
    past_discarded_weights = discarded_weights

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


def prepare_dmrg_results_for_saving(
    dmrg_results, dmrg_parameters, mps_id_str, main_storage_file_path
):
    dmrg_results_saveable = {}
    for key, value in dmrg_results.items():
        if key in ["ket_optimized", "initial_ket"]:
            dmrg_results_saveable[f"{key}_storage"] = str(
                Path(main_storage_file_path.parent.name)
                / f"mps_storage/{mps_id_str}_{key}"
            )
        elif key in ["dmrg_driver"]:
            continue
        else:
            dmrg_results_saveable[key] = value

    return dmrg_results_saveable


def save_dmrg_results(
    dmrg_results,
    dmrg_parameters,
    main_storage_file_path,
    calc_id_str,
    mps_final_storage_path="./",
):
    dmrg_results_saveable = prepare_dmrg_results_for_saving(
        dmrg_results=dmrg_results,
        dmrg_parameters=dmrg_parameters,
        mps_id_str=calc_id_str,
        main_storage_file_path=main_storage_file_path,
    )
    final_destination = str(
        Path(mps_final_storage_path)
        / Path(main_storage_file_path.parent.parent)
        / Path(dmrg_results_saveable["initial_ket_storage"])
    )
    pyblock2.tools.saveMPStoDir(
        mps=dmrg_results["initial_ket"],
        mpsSaveDir=final_destination,
    )

    final_destination = str(
        Path(mps_final_storage_path)
        / Path(main_storage_file_path.parent.parent)
        / Path(dmrg_results_saveable["ket_optimized_storage"])
    )
    pyblock2.tools.saveMPStoDir(
        mps=dmrg_results["ket_optimized"],
        mpsSaveDir=final_destination,
    )

    # if mps_final_storage_path is not None:

    #     final_destination.parent.mkdir(parents=True, exist_ok=True)
    #     shutil.move(
    #         Path(main_storage_file_path.parent.parent)
    #         / Path(dmrg_results_saveable["initial_ket_storage"]),
    #         final_destination,
    #     )

    #     final_destination.parent.mkdir(parents=True, exist_ok=True)
    #     shutil.move(
    #         Path(main_storage_file_path.parent.parent)
    #         / Path(dmrg_results_saveable["ket_optimized_storage"]),
    #         final_destination,
    #     )

    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=dmrg_parameters,
        access_mode="a",
        group=f"{calc_id_str}/dmrg_parameters",
        overwrite=False,
    )
    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=dmrg_results_saveable,
        access_mode="a",
        group=f"{calc_id_str}/dmrg_results",
        overwrite=False,
    )

    # Copy hdf5 file to mps_final_storage_path
    hdf5_backup_location = Path(mps_final_storage_path) / Path(
        main_storage_file_path.parent / Path("hdf5_backup")
    )
    hdf5_backup_location.mkdir(parents=True, exist_ok=True)
    shutil.copy(main_storage_file_path, hdf5_backup_location)

    driver = dmrg_results["dmrg_driver"]
    # Release the memory
    log.info(f"Releasing the memory from driver")
    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )
    driver.finalize()
    log.info(f"Memory released from driver")
    print_system_info(
        f"{os.path.basename(__file__)} - LINE {inspect.getframeinfo(inspect.currentframe()).lineno}"
    )

    # Remove from dictionary
    try:
        del dmrg_results["ket_optimized"]
        del dmrg_results["initial_ket"]
        del dmrg_results["dmrg_driver"]
    except KeyError:
        pass

    del dmrg_results_saveable  # Clear memory
    gc.collect()  # Collect garbage

    log.info(f"Saved dmrg_results to {main_storage_file_path}")
