import h5py
import numpy as np


def get_data_from_incomplete_processing(data_file):
    # Get DMRG energy, bond dimensions, and truncation error for each loop
    dmrg_energies = []
    bond_dimensions = []
    discarded_weights = []
    with h5py.File(data_file, "r") as file_obj:
        # Load for first preloop calculation
        preloop = file_obj["first_preloop_calc"]["dmrg_results"]
        dmrg_energy = float(preloop["dmrg_ground_state_energy"][()])
        bond_dimension = int(preloop["sweep_bond_dims"][()][-1])
        discarded_weight = float(preloop["sweep_max_discarded_weight"][()][-1])
        dmrg_energies.append(dmrg_energy)
        bond_dimensions.append(bond_dimension)
        discarded_weights.append(discarded_weight)

        # Load for second preloop calculation
        preloop = file_obj["second_preloop_calc"]["dmrg_results"]
        dmrg_energy = float(preloop["dmrg_ground_state_energy"][()])
        bond_dimension = int(preloop["sweep_bond_dims"][()][-1])
        discarded_weight = float(preloop["sweep_max_discarded_weight"][()][-1])
        dmrg_energies.append(dmrg_energy)
        bond_dimensions.append(bond_dimension)
        discarded_weights.append(discarded_weight)

        # Load for main loop calculations
        for i in range(1, 100):
            group_name = f"dmrg_loop_{i:03d}"
            if group_name not in file_obj:
                last_loop = i - 1
                print(f"Last loop included = {last_loop}")

                break
            loop = file_obj[group_name]["dmrg_results"]
            dmrg_energy = float(loop["dmrg_ground_state_energy"][()])
            bond_dimension = int(loop["sweep_bond_dims"][()][-1])
            discarded_weight = float(loop["sweep_max_discarded_weight"][()][-1])
            dmrg_energies.append(dmrg_energy)
            bond_dimensions.append(bond_dimension)
            discarded_weights.append(discarded_weight)

        if "final_dmrg_results" in file_obj:
            print("Processed results available")

            final = file_obj["final_dmrg_results"]
            processed_dmrg_energies = final["past_energies_dmrg"][()]
            processed_bond_dimensions = final["bond_dims_used"][()]
            processed_discarded_weights = final["past_discarded_weights"][()]

            print("Checking that processed results match raw results.")
            assert np.allclose(dmrg_energies, processed_dmrg_energies)
            assert np.allclose(bond_dimensions, processed_bond_dimensions)
            assert np.allclose(discarded_weights, processed_discarded_weights)

    num_loops = last_loop
    num_dmrg_calculations = len(dmrg_energies)
    assert num_loops == num_dmrg_calculations - 2
    return (
        dmrg_energies,
        bond_dimensions,
        discarded_weights,
        num_loops,
        num_dmrg_calculations,
    )
