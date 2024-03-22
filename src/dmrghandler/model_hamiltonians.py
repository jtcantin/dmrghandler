import numpy as np


def banded_tight_binding_model(
    num_orbitals, orbital_energies, hopping_amplitude, bandwidth
):
    """
    Function to generate a tight binding model with a chosen bandwidth.

    """

    # Diagonal part of the one-body tensor
    one_body_tensor = np.diag(orbital_energies)

    # Off-diagonal parts of the one-body tensor
    for band_iter in range(1, bandwidth):
        one_body_tensor += np.diag(
            hopping_amplitude * np.ones(num_orbitals - band_iter), band_iter
        )
        one_body_tensor += np.diag(
            hopping_amplitude * np.ones(num_orbitals - band_iter), -band_iter
        )

    return one_body_tensor


def save_one_body_fcidump_file(one_body_tensor, num_orbitals, num_electrons, filename):
    """
    Function to save a one-body tensor to an FCIDUMP file.

    """
    pass
    # Open the file
    # with open(filename
