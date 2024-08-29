from pathlib import Path

import dmrghandler.config_io as config_io
import dmrghandler.slurm_scripts as slurm_scripts

# This file will generate configuration and submit files,
# along with submission commands
# For the first run, log into Niagara and go to your SCRATCH directory
# Then, install Python and dmrghandler via the following commands:
# $ module load python/3.9
# $ python -m venv env_dmrg_thresholding
# $ source env_dmrg_thresholding/bin/activate
# $ pip install git+https://github.com/jtcantin/dmrghandler

# Then, put in your email in line 129 of this file and
# copy this file to your SCRATCH directory
# To run the script, use the following commands:
# $ python prepare_calcs_small_to_med_catalysts.py

# Then, use the output submission commands to submit the jobs

# For any future logins to Niagara, go to SCRATCH
# and then use the following commands to run:
# $ module load python/3.9
# $ source env_dmrg_thresholding/bin/activate
# $ python prepare_calcs_small_to_med_catalysts.py

data_files_folder = Path("./fcidumps_benchmark/")

data_file_list_file = [
    # "",
    # "FCIDUMP_L_4x4_Ut_2.0",  # Fermi-Hubbard model
    # "FCIDUMP_L_4x4_Ut_4.0",
    # "FCIDUMP_L_4x4_Ut_8.0",
    # "FCIDUMP_L_6x6_Ut_2.0",
    # "FCIDUMP_L_6x6_Ut_4.0",
    # "FCIDUMP_L_6x6_Ut_8.0",
    # "FCIDUMP_b_cc-pvdz",  # Benzene
    # "FCIDUMP_b_sto-3g",
    # "FCIDUMP_d_1.68_b_cc-pvdz-dk_ne_12",  # Cr2
    # "FCIDUMP_d_1.68_b_cc-pvdz-dk_ne_28",
    # "FCIDUMP_d_1.68_b_sto-3g_ne_12",
    # "FCIDUMP_d_1.68_b_sto-3g_ne_28",
    # "FCIDUMP_d_2.0_b_cc-pvdz-dk_ne_12",
    # "FCIDUMP_d_2.0_b_cc-pvdz-dk_ne_28",
    # "FCIDUMP_d_2.0_b_sto-3g_ne_12",
    # "FCIDUMP_d_2.0_b_sto-3g_ne_28",
    # "FCIDUMP_d_3.0_b_cc-pvdz-dk_ne_12",
    # "FCIDUMP_d_3.0_b_cc-pvdz-dk_ne_28",
    # "FCIDUMP_d_3.0_b_sto-3g_ne_12",
    # "FCIDUMP_d_3.0_b_sto-3g_ne_28",
    # "FCIDUMP_g_bent_b_cc-pvdz",  # Ozone
    # "FCIDUMP_g_bent_b_cc-pvtz",
    # "FCIDUMP_g_bent_b_sto-3g",
    # "FCIDUMP_g_ring_b_cc-pvdz",
    # "FCIDUMP_g_ring_b_cc-pvtz",
    # "FCIDUMP_g_ring_b_sto-3g",
    "fcidump.59_5_16_noncan_0.2_new",
    "fcidump.60_5_16_noncan_0.2_new",
    "fcidump.61_3_15_af_noncan_0.2_new",
    "fcidump.62_3_15_af_noncan_0.2_new",
    "fcidump.63_5_15_af_ts_noncan_0.2_new",
    "fcidump.64_5_15_af_ts_noncan_0.2_new",
    "fcidump.65_5_15_af_noncan_0.2_new",
    "fcidump.66_5_15_af_noncan_0.2_new",
]


data_file_list = []
for data_file in data_file_list_file:
    data_file_path = Path(data_files_folder) / Path(data_file)
    data_file_list.append(str(data_file_path))
print(f"data_file_list: {data_file_list}")

config_dict = {
    "plot_filename_prefix_list": [
        # "",
        # "L_4x4_Ut_2.0",
        # "L_4x4_Ut_4.0",
        # "L_4x4_Ut_8.0",
        # "L_6x6_Ut_2.0",
        # "L_6x6_Ut_4.0",
        # "L_6x6_Ut_8.0",
        # "b_cc-pvdz",
        # "b_sto-3g",
        # "d_1.68_b_cc-pvdz-dk_ne_12",
        # "d_1.68_b_cc-pvdz-dk_ne_28",
        # "d_1.68_b_sto-3g_ne_12",
        # "d_1.68_b_sto-3g_ne_28",
        # "d_2.0_b_cc-pvdz-dk_ne_12",
        # "d_2.0_b_cc-pvdz-dk_ne_28",
        # "d_2.0_b_sto-3g_ne_12",
        # "d_2.0_b_sto-3g_ne_28",
        # "d_3.0_b_cc-pvdz-dk_ne_12",
        # "d_3.0_b_cc-pvdz-dk_ne_28",
        # "d_3.0_b_sto-3g_ne_12",
        # "d_3.0_b_sto-3g_ne_28",
        # "g_bent_b_cc-pvdz",
        # "g_bent_b_cc-pvtz",
        # "g_bent_b_sto-3g",
        # "g_ring_b_cc-pvdz",
        # "g_ring_b_cc-pvtz",
        # "g_ring_b_sto-3g",
        "59_5_16_noncan_0.2_new",
        "60_5_16_noncan_0.2_new",
        "61_3_15_af_noncan_0.2_new",
        "62_3_15_af_noncan_0.2_new",
        "63_5_15_af_ts_noncan_0.2_new",
        "64_5_15_af_ts_noncan_0.2_new",
        "65_5_15_af_noncan_0.2_new",
        "66_5_15_af_noncan_0.2_new",
    ],
    "main_storage_folder_path_prefix": "./data_storage",
    "max_bond_dimension_list": [6000],
    "max_time_limit_sec_list": [23.5 * 3600],  # 23.5 hrs
    "min_energy_change_hartree_list": [0.5e-4],
    "extrapolation_type_list": ["discard_weights"],
    "starting_bond_dimension_list": [
        4,
    ],
    "max_num_sweeps_list": [80],
    "energy_convergence_threshold_list": [1e-8],
    "sweep_schedule_bond_dims_parameters": [
        [(2, 4), (1, 5)]
    ],  # (division_factor, count),
    # e.g. [(2, 4), (1, 5)] and init_bd of 3 ->[1, 1, 1, 1, 3, 3, 3, 3, 3]
    "sweep_schedule_noise_list": [[1e-4] * 4 + [1e-5] * 4 + [0]],
    "sweep_schedule_davidson_threshold_list": [[1e-10] * 9],
    "init_state_bond_dimension_division_factor_list": [2],
    "init_state_seed_list": [
        # 658724484,
        # 944398351,
        # 852965221,
        # 263856242,
        # 659349335,
        # 440264380,
        # 380311945,
        # 832349004,
        # 900300072,
        # 974244990,
        # 850604342,
        # 266578604,
        # 304729279,
        # 882788950,
        # 915946814,
        # 632724145,
        # 267099520,
        # 119641569,
        # 442296275,
        # 495763738,
        # 446907089,
        # 889544,
        # 141309035,
        # 913425049,
        # 179408518,
        # 783342046,
        266578604, #MnMono (8)
        304729279,
        882788950,
        915946814,
        632724145,
        267099520,
        119641569,
        442296275,
    ],
    "initial_mps_method_list": ["random"],
    "factor_half_convention_list": [True],
    # "symmetry_type_list": ["SZ", "SZ", "SU(2)", "SU(2)"],
    "symmetry_type_list": ["SU(2)"],
    "num_threads_list": [40],
    "n_mkl_threads_list": [40],
    "track_mem": [False],
    "reordering_method_list": [
        # "fiedler, exchange matrix"
        # "fiedler, interaction matrix, SU(2) calc",
        "gaopt, exchange matrix",
        # "gaopt, interaction matrix",
        # "gaopt, interaction matrix, SU(2) calc",
    ],
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
    "stack_mem": 150 * 1024 * 1024 * 1024,  # =150GiB
    "stack_mem_ratio": 0.4,
}


config_files_list, config_dict_single_file_list = config_io.gen_config_files(
    data_file_list=data_file_list,
    config_dict=config_dict,
    dmrg_advanced_config=dmrg_advanced_config,
    config_file_prefix="config_gsee_benchmark_coarse_set_run3",
)
print(f"config_files_list: {config_files_list}")
# print(f"config_dict_single_file_list: {config_dict_single_file_list}")

submit_dict = {
    "time_cap_string": "00-23:59:00",
    "job_name": "dmrg_gsee_benchmark_coarse_set_run3",
    "email": "email@email.ca",
    "account_name": "rrg-example",
    "tasks_per_node": "1",
    "cpus_per_task": "40",
    "partition": "debug",  # The script will not be run on the debug partition from this setting, so ignore this line
    "python_environment_location": "env_dmrg_thresholding",
}

slurm_scripts.gen_run_files(submit_dict, config_dict_single_file_list)

submit_commands = slurm_scripts.gen_submit_commands(config_dict_single_file_list)

print(submit_commands)
