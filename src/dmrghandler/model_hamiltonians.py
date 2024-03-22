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
    for band_iter in range(1, bandwidth + 1):
        one_body_tensor += np.diag(
            hopping_amplitude * np.ones(num_orbitals - band_iter), band_iter
        )
        one_body_tensor += np.diag(
            hopping_amplitude * np.ones(num_orbitals - band_iter), -band_iter
        )

    return one_body_tensor


def random_orbital_rotation(one_body_tensor, seed):
    """
    Function to generate a random orbital rotation matrix.

    """

    # Set the seed for reproducibility
    np.random.seed(seed)

    # Generate a random unitary matrix
    num_orbitals = one_body_tensor.shape[0]
    random_unitary = np.random.rand(num_orbitals, num_orbitals) + 1j * np.random.rand(
        num_orbitals, num_orbitals
    )
    random_unitary, _ = np.linalg.qr(random_unitary)

    return random_unitary


def get_one_body_term_ground_state(one_body_tensor, num_electrons):
    # Diagonalize the one-body tensor
    eigenvalues, eigenvectors = np.linalg.eigh(one_body_tensor)

    # Sort the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Get the ground state
    num_fully_occupied_orbitals = num_electrons // 2
    singly_occupied_orbital = bool(num_electrons % 2)
    ground_state_energy = (
        2 * np.sum(eigenvalues[:num_fully_occupied_orbitals])
        + singly_occupied_orbital * eigenvalues[num_fully_occupied_orbitals]
    )

    return ground_state_energy, eigenvalues, eigenvectors
