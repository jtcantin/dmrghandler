import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def gen_single_node_job_script(submit_dict, submit_script_file_name):

    time_cap_string = submit_dict["time_cap_string"]
    job_name = submit_dict["job_name"]
    email = submit_dict["email"]
    account_name = submit_dict["account_name"]
    tasks_per_node = submit_dict["tasks_per_node"]
    cpus_per_task = submit_dict["cpus_per_task"]
    partition = submit_dict["partition"]
    job_output_file = submit_dict["job_output_file"]
    job_error_file = submit_dict["job_error_file"]
    python_environment_location = submit_dict["python_environment_location"]
    data_files_folder = submit_dict["data_files_folder"]
    data_storage_folder = submit_dict["data_storage_folder"]
    python_run_file = submit_dict["python_run_file"]
    log_folder = submit_dict["log_folder"]

    submit_script_string = f"""#!/bin/bash
#SBATCH --account={account_name}
#SBATCH --nodes=1
#SBATCH --time={time_cap_string}
#SBATCH --job-name={job_name}
#SBATCH --mail-user={email}
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node={tasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
##SBATCH --partition={partition}
#SBATCH --output={job_output_file}
#SBATCH --error={job_error_file}
# asks Slurm to send the USR1 signal 120 seconds before end of the time limit
# Timeout signal handling based on https://documentation.sigma2.no/jobs/guides/cleanup_timeout.html 
#SBATCH --signal=B:USR1@120

# define the handler function
# note that this is not executed here, but rather
# when the associated signal is sent
your_cleanup_function()
{{
    echo "function your_cleanup_function called at $(date)"
    # do whatever cleanup you want here
    #Copy output to project directory
    echo 'current directory'
    pwd
    echo 'files here:'
    ls -lh
    mkdir -p $SCRATCH/{data_storage_folder}
    cp -r {data_storage_folder}/. $SCRATCH/{data_storage_folder}
    echo "output files copied to $SCRATCH/{data_storage_folder}"
    cp -r {log_folder}/. $SCRATCH/{log_folder}
    echo "log files copied to $SCRATCH/{log_folder}"
    cp *.log $SCRATCH/{data_storage_folder}

    #Clean up RAMDISK
    echo 'removing left over files'
    rm -rf {data_storage_folder}
    rm -rf {data_files_folder}
    rm -f *.py
    echo 'files remaining:'
    ls -lhR
}}

export LC_ALL=en_US.UTF-8

# Below error handling from https://unix.stackexchange.com/a/462157
set -eE -o functrace

failure() {{
  local lineno=$1
  local msg=$2
  echo "Failed at $lineno: $msg"
}}
trap 'failure ${{LINENO}} "$BASH_COMMAND"' ERR

#Import Modules
# module load CCEnv
# module load StdEnv
# module load gcc/10
# module load boost cmake libffi fmt 
# module load rust
module load python/3.9
# module load intel/2021.2.0
# module load hdf5/1.10.7

#Activate python environment
source $SCRATCH/{python_environment_location}/bin/activate

echo ' '
echo $SCRATCH
ls $SCRATCH

# Turn off implicit threading in Python, R
export OMP_NUM_THREADS=1

# Info
echo ' '
echo "current directory:"
pwd
# ls

#Help avoid clashes
sleep $[ ( $RANDOM % 10 )  + 1 ]s

datetime=`date +%Y%m%d-%H%M%S.%N`

echo $datetime

#work in RAMdisk
cd $SLURM_TMPDIR

echo ' '
echo "current directory:"
pwd
ls

#Copy over files
mkdir -p {data_files_folder}
cp -r $SCRATCH/{data_files_folder}/. ./{data_files_folder}
echo 'data and python files copied over'

mkdir -p {data_storage_folder}

echo "files currently here:"
ls -lhR

#Get running information
echo ' '
echo 'Job Info'
echo 'SLURM_JOB_NAME:'
echo $SLURM_JOB_NAME
echo 'SLURM_JOB_ID:'
echo $SLURM_JOB_ID
echo 'SLURM_NODE_ALIASES:'
echo $SLURM_NODE_ALIASES
echo 'SLURM_NODEID:'
echo $SLURM_NODEID
echo 'SLURM_NODELIST:'
echo $SLURM_NODELIST
echo 'SLURM_SUBMIT_DIR:'
echo $SLURM_SUBMIT_DIR
echo 'SLURM_TASK_PID:'
echo $SLURM_TASK_PID
echo 'SLURMD_NODENAME:'
echo $SLURMD_NODENAME

#Run code
echo ' '
echo 'Single run of DMRG calculations beginning'
echo ' '
python {python_run_file} &
wait
echo ' '
echo 'Single run of DMRG calculations done'
echo ' '

#Copy output to project directory
echo 'current directory'
pwd
echo 'files here:'
ls -lhR
mkdir -p $SCRATCH/{data_storage_folder}
cp {data_storage_folder}/dmrg_results.hdf5 $SCRATCH/{data_storage_folder}
cp ./dmrghandler.log $SCRATCH/{data_storage_folder}
cp -r {data_storage_folder}/plots $SCRATCH/{data_storage_folder}
echo "output files copied to $SCRATCH/{data_storage_folder}"
#cp -r {log_folder}/. $SCRATCH/{log_folder}
#echo "log files copied to $SCRATCH/{log_folder}"
#cp *.log $SCRATCH/{data_storage_folder}

#Clean up RAMDISK
echo 'removing left over files'
rm -rf {data_storage_folder}
rm -rf {data_files_folder}
rm -f *.py
echo 'files remaining:'
ls -lhR

echo 'Job Completed'
cd $SCRATCH

sleep 2

exit 0
    """
    log.debug(f"submit_script_string: {submit_script_string}")
    with open(submit_script_file_name, "w") as file:
        file.write(submit_script_string)

    log.info("Submit script file written")
    log.info(f"submit_script_file_name: {submit_script_file_name}")


def gen_python_run_script(python_run_file_name, config_file_name):
    python_run_script_string_1 = rf"""
import inspect
import logging
import os
import time

from pathlib import Path

import dmrghandler.dmrg_calc_prepare as dmrg_calc_prepare
import dmrghandler.dmrg_looping as dmrg_looping
import dmrghandler.energy_extrapolation as energy_extrapolation
from dmrghandler.profiling import print_system_info
import dmrghandler.hdf5_io as hdf5_io

# log = logging.getLogger("{Path(config_file_name).stem}")
log = logging.getLogger("dmrghandler")
log.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler("dmrghandler.log")
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - Line %(lineno)d - %(message)s")
fh.setFormatter(formatter)
# add the handlers to the log
log.addHandler(fh)

if __name__ == "__main__":

    log.debug(f"Starting global calculation script")
    print_system_info(
        f"{{os.path.basename(__file__)}} - LINE {{inspect.getframeinfo(inspect.currentframe()).lineno}}"
    )
    wall_time_start_ns = time.perf_counter_ns()
    cpu_time_start_ns = time.process_time_ns()

    config_file = "{config_file_name}"

    (
        one_body_tensor,
        two_body_tensor,
        dmrg_parameters,
        looping_parameters,
        data_config,
    ) = dmrg_calc_prepare.prepare_calc(config_file)

    wall_time_prepare_calc_done_ns = time.perf_counter_ns()
    cpu_time_prepare_calc_done_ns = time.process_time_ns()

    wall_time_prepare_calc_ns = wall_time_prepare_calc_done_ns - wall_time_start_ns
    cpu_time_prepare_calc_ns = cpu_time_prepare_calc_done_ns - cpu_time_start_ns

    wall_time_total_ns = time.perf_counter_ns() - wall_time_start_ns
    cpu_time_total_ns = time.process_time_ns() - cpu_time_start_ns
    log.debug(f"Calculations prepared.")
    log.debug(f"wall_time_prepare_calc_s: {{wall_time_prepare_calc_ns/1E9}}")
    log.debug(f"cpu_time_prepare_calc_s: {{cpu_time_prepare_calc_ns/1E9}}")
    log.debug(f"wall_time_total_s: {{wall_time_total_ns/1E9}}")
    log.debug(f"cpu_time_total_s: {{cpu_time_total_ns/1E9}}")

    print_system_info(
        f"{{os.path.basename(__file__)}} - LINE {{inspect.getframeinfo(inspect.currentframe()).lineno}}"
    )

    

    if len(one_body_tensor) == 2:
        log.debug(f"one_body_tensor: {{one_body_tensor[0].shape}}")
        log.debug(f"one_body_tensor: {{one_body_tensor[1].shape}}")
        log.debug(f"two_body_tensor: {{two_body_tensor[0].shape}}")
        log.debug(f"two_body_tensor: {{two_body_tensor[1].shape}}")
        log.debug(f"two_body_tensor: {{two_body_tensor[2].shape}}")
    else:
        log.debug(f"one_body_tensor: {{one_body_tensor.shape}}")
        log.debug(f"two_body_tensor: {{two_body_tensor.shape}}")
    log.debug(f"dmrg_parameters: {{dmrg_parameters}}")
    log.debug(f"looping_parameters: {{looping_parameters}}")
    log.debug(f"data_config: {{data_config}}")

    max_bond_dimension = looping_parameters["max_bond_dimension"]
    max_time_limit_sec = looping_parameters["max_time_limit_sec"]
    min_energy_change_hartree = looping_parameters["min_energy_change_hartree"]
    main_storage_folder_path = data_config["main_storage_folder_path"]
    loop_results = dmrg_looping.dmrg_central_loop(
        one_body_tensor=one_body_tensor,
        two_body_tensor=two_body_tensor,
        dmrg_parameters=dmrg_parameters,
        max_bond_dimension=max_bond_dimension,
        max_time_limit_sec=max_time_limit_sec,
        min_energy_change_hartree=min_energy_change_hartree,
        main_storage_folder_path=main_storage_folder_path,
        verbosity=2,
        move_mps_to_final_storage_path=os.environ['SCRATCH']
    )

    wall_time_dmrg_loop_done_ns = time.perf_counter_ns()
    cpu_time_dmrg_loop_done_ns = time.process_time_ns()

    wall_time_dmrg_loop_ns = wall_time_dmrg_loop_done_ns - wall_time_prepare_calc_done_ns
    cpu_time_dmrg_loop_ns = cpu_time_dmrg_loop_done_ns - cpu_time_prepare_calc_done_ns

    wall_time_total_ns = time.perf_counter_ns() - wall_time_start_ns
    cpu_time_total_ns = time.process_time_ns() - cpu_time_start_ns
    log.debug(f"Looping complete")
    log.debug(f"wall_time_dmrg_loop_s: {{wall_time_dmrg_loop_ns/1E9}}")
    log.debug(f"cpu_time_dmrg_loop_s: {{cpu_time_dmrg_loop_ns/1E9}}")
    log.debug(f"wall_time_total_s: {{wall_time_total_ns/1E9}}")
    log.debug(f"cpu_time_total_s: {{cpu_time_total_ns/1E9}}")
    print_system_info(
        f"{{os.path.basename(__file__)}} - LINE {{inspect.getframeinfo(inspect.currentframe()).lineno}}"
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
    log.info(f"finish_reason: {{finish_reason}}")
    log.info(f"energy_change: {{energy_change}}")
    log.info(f"discard_weight_change: {{discard_weight_change}}")
    log.info(f"bond_dims_used: {{bond_dims_used}}")
    log.info(f"past_energies_dmrg: {{past_energies_dmrg}}")
    log.info(f"past_discarded_weights: {{past_discarded_weights}}")
    log.info(f"loop_entry_count: {{loop_entry_count}}")
    log.info(f"unmodified_fit_parameters_list: {{unmodified_fit_parameters_list}}")
    log.info(f"fit_parameters_list: {{fit_parameters_list}}")

    main_storage_folder_path = Path(main_storage_folder_path)
    main_storage_file_path = main_storage_folder_path / "dmrg_results.hdf5"
    # Make directory if it does not exist
    main_storage_folder_path.mkdir(parents=True, exist_ok=True)

    final_results_dict = {{
        "final_dmrg_energy": past_energies_dmrg[-1],
        "final_dmrg_bond_dim": bond_dims_used[-1],
        "finish_reason": finish_reason,
        "energy_change": energy_change,
        "discard_weight_change": discard_weight_change,
        "bond_dims_used": bond_dims_used,
        "past_energies_dmrg": past_energies_dmrg,
        "past_discarded_weights": past_discarded_weights,
        "loop_entry_count": loop_entry_count,
        "unmodified_fit_parameters_list": unmodified_fit_parameters_list,
        "fit_parameters_list": fit_parameters_list,
    }}

    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=final_results_dict,
        access_mode="a",
        group="final_dmrg_results",
        overwrite=False,
    )

    # Get final extrapolated energy
    wall_time_extrapolation_start_ns = time.perf_counter_ns()
    cpu_time_extrapolation_start_ns = time.process_time_ns()
    result_obj, energy_estimated, fit_parameters, R_squared = (
        energy_extrapolation.dmrg_energy_extrapolation(
            energies_dmrg=past_energies_dmrg,
            independent_vars=past_discarded_weights,
            extrapolation_type="discarded_weight",
            past_parameters=fit_parameters_list[-1],
            verbosity=2,
        )
    )
    wall_time_extrapolation_done_ns = time.perf_counter_ns()
    cpu_time_extrapolation_done_ns = time.process_time_ns()

    wall_time_extrapolation_ns = wall_time_extrapolation_done_ns - wall_time_extrapolation_start_ns
    cpu_time_extrapolation_ns = cpu_time_extrapolation_done_ns - cpu_time_extrapolation_start_ns

    wall_time_total_ns = time.perf_counter_ns() - wall_time_start_ns
    cpu_time_total_ns = time.process_time_ns() - cpu_time_start_ns
    log.debug(f"Extrapolation complete")
    log.debug(f"wall_time_extrapolation_s: {{wall_time_extrapolation_ns/1E9}}")
    log.debug(f"cpu_time_extrapolation_s: {{cpu_time_extrapolation_ns/1E9}}")
    log.debug(f"wall_time_total_s: {{wall_time_total_ns/1E9}}")
    log.debug(f"cpu_time_total_s: {{cpu_time_total_ns/1E9}}")

    log.info(f"energy_estimated: {{energy_estimated}}")
    log.info(f"fit_parameters: {{fit_parameters}}")
    log.info(f"R_squared: {{R_squared}}")
    log.info(f"result_obj.message: {{result_obj.message}}")
    log.info(f"result_obj.cost: {{result_obj.cost}}")
    log.info(f"result_obj.fun: {{result_obj.fun}}")

    final_extrapolated_results_dict = {{
        "energy_estimated": energy_estimated,
        "fit_parameters": fit_parameters,
        "R_squared": R_squared,
        "result_obj_message": result_obj.message,
        "result_obj_cost": result_obj.cost,
        "result_obj_fun": result_obj.fun,
    }}

    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=final_extrapolated_results_dict,
        access_mode="a",
        group="final_extrapolated_results",
        overwrite=False,
    )

    wall_time_plotting_start_ns = time.perf_counter_ns()
    cpu_time_plotting_start_ns = time.process_time_ns()

    plot_filename_prefix = data_config["plot_filename_prefix"]
    energy_extrapolation.plot_extrapolation(
        discarded_weights=past_discarded_weights,
        energies_dmrg=past_energies_dmrg,
        fit_parameters=fit_parameters,
        bond_dims=bond_dims_used,
        plot_filename=main_storage_folder_path
        / Path("plots")
        / Path(f"{{plot_filename_prefix}}_energy_extrapolation"),
        figNum=0,
    )
    wall_time_plotting_done_ns = time.perf_counter_ns()
    cpu_time_plotting_done_ns = time.process_time_ns()

    wall_time_plotting_ns = wall_time_plotting_done_ns - wall_time_plotting_start_ns
    cpu_time_plotting_ns = cpu_time_plotting_done_ns - cpu_time_plotting_start_ns

    wall_time_total_ns = time.perf_counter_ns() - wall_time_start_ns
    cpu_time_total_ns = time.process_time_ns() - cpu_time_start_ns
    log.debug(f"Plotting complete")
    log.debug(f"wall_time_plotting_s: {{wall_time_plotting_ns/1E9}}")
    log.debug(f"cpu_time_plotting_s: {{cpu_time_plotting_ns/1E9}}")
    log.debug(f"wall_time_total_s: {{wall_time_total_ns/1E9}}")
    log.debug(f"cpu_time_total_s: {{cpu_time_total_ns/1E9}}")

    #Save timing data
    timing_dict = {{
        "wall_time_prepare_calc_s": wall_time_prepare_calc_ns/1e9,
        "cpu_time_prepare_calc_s": cpu_time_prepare_calc_ns/1e9,
        "wall_time_dmrg_loop_s": wall_time_dmrg_loop_ns/1e9,
        "cpu_time_dmrg_loop_s": cpu_time_dmrg_loop_ns/1e9,
        "wall_time_extrapolation_s": wall_time_extrapolation_ns/1e9,
        "cpu_time_extrapolation_s": cpu_time_extrapolation_ns/1e9,
        "wall_time_plotting_s": wall_time_plotting_ns/1e9,
        "cpu_time_plotting_s": cpu_time_plotting_ns/1e9,
        "wall_time_total_s": (time.perf_counter_ns() - wall_time_start_ns)/1e9,
        "cpu_time_total_s": (time.process_time_ns() - cpu_time_start_ns)/1e9,
    }}

    hdf5_io.save_many_variables_to_hdf5(
        hdf5_filepath=main_storage_file_path,
        variables=timing_dict,
        access_mode="a",
        group="timing_data",
        overwrite=False,
    )
    """
    python_run_script_string = python_run_script_string_1

    log.debug(f"python_run_script_string: {python_run_script_string}")
    with open(python_run_file_name, "w") as file:
        file.write(python_run_script_string)
    log.info("Python run script file written")
    log.info(f"python_run_file_name: {python_run_file_name}")


def gen_run_files(submit_dict, config_dict_list):
    for config_dict in config_dict_list:
        data_config = config_dict["data_config"]
        python_run_file_name = data_config["python_run_file"]
        submit_script_file_name = data_config["submit_script_file"]
        config_file_name = data_config["config_file"]

        submit_dict["job_output_file"] = data_config["folder_uuid"] + ".out"
        submit_dict["job_error_file"] = data_config["folder_uuid"] + ".err"
        submit_dict["data_files_folder"] = data_config["data_prep_path"]
        submit_dict["data_storage_folder"] = data_config["main_storage_folder_path"]
        submit_dict["python_run_file"] = python_run_file_name
        submit_dict["log_folder"] = data_config["main_storage_folder_path"]

        gen_single_node_job_script(submit_dict, submit_script_file_name)
        gen_python_run_script(python_run_file_name, config_file_name)


def gen_submit_commands(config_dict_list):
    submit_commands_str = ""
    for config_dict in config_dict_list:
        data_config = config_dict["data_config"]
        original_data_file_path = data_config["original_data_file_path"]
        submit_script_file_name = Path(data_config["submit_script_file"])
        data_prep_path = Path(data_config["data_prep_path"])

        comment_string = f"# {original_data_file_path}\n"

        # change_dir_command = f"cd {submit_script_file_name.parent}\n"
        submit_dir = Path(data_prep_path.parent) / Path("submit_dir")
        submit_dir.mkdir(exist_ok=True, parents=True)
        change_dir_command = f"cd {submit_dir}\n"

        submit_command = f"sbatch {os.path.relpath(submit_script_file_name.resolve(),start=submit_dir.resolve())}\n"

        log.info(comment_string)
        log.info(f"change_dir_command: {change_dir_command}")
        log.info(f"submit_command: {submit_command}")
        submit_commands_str += comment_string
        submit_commands_str += change_dir_command
        submit_commands_str += submit_command

    return submit_commands_str
