import logging

import scipy.linalg
import scipy.stats

log = logging.getLogger(__name__)
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

module_verbosity = 2
module_ftol = 1e-8
module_xtol = 1e-8
module_gtol = 1e-12
log_shift = 1e-16
neg_log_threshold = 1e-7


def dmrg_energy_extrapolation(
    energies_dmrg: List[float],
    independent_vars: List[float],
    extrapolation_type: str = "discarded_weight",
    past_parameters: List[float] = None,
    verbosity=module_verbosity,
):
    """
    Extrapolate the energy to the exact limit of the MPS

    Args:
        energies: list of energies
        independent_vars: list of independent variables
        extrapolation_type: type of extrapolation to use
    Returns:
        result_obj: scipy.optimize.OptimizeResult object
        energy_estimated: extrapolated energy
        fit_parameters: list of fit parameters
        R_squared: R^2 value of the fit
    """

    if extrapolation_type == "discarded_weight":
        result_obj = discarded_weight_extrapolation(
            discarded_weights=independent_vars,
            energies_dmrg=energies_dmrg,
            past_parameters=past_parameters,
            verbosity=verbosity,
        )
        fit_parameters = result_obj.x
        energy_estimated = fit_parameters[-1]
        rel_energies = (energies_dmrg - energy_estimated) / energy_estimated
        if (rel_energies <= 0).any():
            for iiter, val in enumerate(rel_energies):
                if val <= 0:
                    rel_energies[iiter] = np.abs(val) + 1e-16
                    # if np.abs(val) < neg_log_threshold:
                    #     rel_energies[iiter] = np.abs(val) + 1e-16
                    # # print(f"val: {val}")
                    # else:
                    #     raise ValueError(
                    #         f"Relative energy is less than or equal to zero. energies_dmrg: {energies_dmrg}, energy_estimated: {energy_estimated}\n"
                    #         f"rel_energies: {rel_energies}\n"
                    #     )
        """      This block is the new version, but not passing tests   
        rel_energies = np.abs((energies_dmrg - energy_estimated) / energy_estimated)
        # if (rel_energies <= 0).any():
        #     for iiter, val in enumerate(rel_energies):
        #         if val <= 0:
        #             rel_energies[iiter] = np.abs(val) + 1e-16
        #             # if np.abs(val) < neg_log_threshold:
        #             #     rel_energies[iiter] = np.abs(val) + 1e-16
        #             # # print(f"val: {val}")
        #             # else:
        #             #     raise ValueError(
        #             #         f"Relative energy is less than or equal to zero. energies_dmrg: {energies_dmrg}, energy_estimated: {energy_estimated}\n"
        #             #         f"rel_energies: {rel_energies}\n"
        #             #     )
        """
        ln_rel_energies = np.log(rel_energies)
        # if np.isnan(ln_rel_energies).any():
        #
        # raise ValueError(
        #     f"ln_rel_energies has NaN values. energies_dmrg: {energies_dmrg}, energy_estimated: {energy_estimated}\n"
        #     f"ln_rel_energies: {ln_rel_energies}\n"
        #     f"(energies_dmrg - energy_estimated): {(energies_dmrg - energy_estimated)}\n"
        #     f"(energies_dmrg - energy_estimated) / energy_estimated :{(energies_dmrg - energy_estimated) / energy_estimated}"
        # )
        R_squared = calc_coefficient_of_determination(
            x_data=independent_vars,
            y_data=ln_rel_energies,
            predictor_fcn=discarded_weight_predictor,
            predictor_fcn_args=(fit_parameters[0], fit_parameters[1]),
        )
    else:
        raise ValueError(f"Extrapolation type {extrapolation_type} not recognized")

    return result_obj, energy_estimated, fit_parameters, R_squared


def discarded_weight_extrapolation(
    discarded_weights: List[float],
    energies_dmrg: List[float],
    past_parameters,
    verbosity=module_verbosity,
):
    """
    Extrapolate the energy to the exact limit of the MPS using
    the discarded weight. The functional form is:
    ln ΔE_rel =a+b ln(δϵ)
    where ΔE_rel is the relative energy difference, δϵ is the discarded weight,
    a and b are the fit parameters, and ΔE_rel = (E_DMRG - E_estimated)/E_estimated.
    E_estimated is treated as a free parameter in the fit.

    Args:
        energies_dmrg: list of energies from DMRG
        discarded_weights: list of discarded_weights

    Returns:
        energy_estimated: extrapolated energy
        fit_parameters: list of fit parameters
        R_squared: R^2 value of the fit
    """
    if past_parameters is None:
        initial_guess = [0.25, 1.4, energies_dmrg[-1]]
        # initial_guess = [0.25, 1.4, np.min(energies_dmrg) - 1e-3]
    else:
        initial_guess = past_parameters

    # print(f"Initial guess: {initial_guess}")
    result_obj = scipy.optimize.least_squares(
        fun=discarded_weight_residuals_function,
        x0=initial_guess,
        # jac=discarded_weight_residuals_gradient_matrix,
        jac="3-point",
        bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),  # alpha, b, E_estimated
        # np.inf >= alpha = exp(a) >= 0
        # b >= 0 as it is required δϵ^b -> 0 as δϵ -> 0
        method="dogbox",  # Recommended by Scipy for small problems with bounds
        ftol=module_ftol,
        xtol=module_xtol,
        gtol=module_gtol,
        x_scale="jac",  # Set scale to be based on the size of the gradient components
        loss="soft_l1",  # Means the cost function is
        # loss="linear",  # Means the cost function is
        # # Σ_i  (E_estimated*(α[δϵ_i]^b) + E_estimated - E_dmrg_i)^2
        f_scale=1.0,
        max_nfev=100 * 3,  # 100*3 is the default value of max function evaluations
        diff_step=None,
        tr_solver="exact",  # Supposed to be the best for small problems with dense jacobians
        tr_options={},
        jac_sparsity=None,
        verbose=verbosity,
        args=(discarded_weights, energies_dmrg),
        kwargs={},
    )

    return result_obj


def discarded_weight_cost_function(param_vec, discarded_weights, energies_dmrg):
    """
    Function to minimize for the discarded weight extrapolation.
    Has the form:
        0 = Σ_i  (E_estimated*(α[δϵ_i]^b) + E_estimated - E_dmrg_i)^2

    Args:
        param_vec: list of fit parameters [alpha, b, E_estimated]
        discarded_weights: list of discarded_weights
        energies_dmrg: list of DMRG energies
        alpha: fit parameter
        b: fit parameter
        E_estimated: fit parameter, energy to be estimated

    Returns:
        cost: value of the cost function
    """
    alpha, b, E_estimated = param_vec
    # discarded_weights, energies_dmrg = args
    weight_to_b = np.power(discarded_weights, b)
    alpha_X_weight_to_b = alpha * weight_to_b
    return np.sum(
        np.power(E_estimated * alpha_X_weight_to_b + E_estimated - energies_dmrg, 2)
    )


def discarded_weight_residuals_function_ln(
    param_vec, discarded_weights, energies_dmrg, local_log_shift=1e-16
):
    """
    Residuals for the discarded weight extrapolation.
    Has the form:
        ln(ΔE_rel) =a+b ln(δϵ)
        where ΔE_rel = |(E_DMRG - E_estimated)/E_estimated|
        E_estimated is treated as a free parameter in the fit.

    Args:
        param_vec: list of fit parameters [a, b, E_estimated], length n=3
        discarded_weights: list of discarded_weights
        energies_dmrg: list of DMRG energies
        a: fit parameter
        b: fit parameter
        E_estimated: fit parameter, energy to be estimated

    Returns:
        residuals: darray (of lenght len(discarded_weights)) of the residuals
    """
    a, b, E_estimated = param_vec
    # discarded_weights, energies_dmrg = args
    linear_term = a + b * np.log(discarded_weights)
    energy_term = (energies_dmrg - E_estimated) / E_estimated
    return np.abs(np.log(np.abs(energy_term) + local_log_shift) - linear_term) ** 2


def discarded_weight_residuals_function(param_vec, discarded_weights, energies_dmrg):
    """
    Residuals for the discarded weight extrapolation.
    Has the form:
        E_estimated*(α[δϵ_i]^b) + E_estimated - E_dmrg_i

    Args:
        param_vec: list of fit parameters [alpha, b, E_estimated], length n=3
        discarded_weights: list of discarded_weights
        energies_dmrg: list of DMRG energies
        alpha: fit parameter
        b: fit parameter
        E_estimated: fit parameter, energy to be estimated

    Returns:
        residuals: darray (of lenght len(discarded_weights)) of the residuals
    """
    alpha, b, E_estimated = param_vec
    # discarded_weights, energies_dmrg = args
    weight_to_b = np.power(discarded_weights, b)
    alpha_X_weight_to_b = alpha * weight_to_b
    return (E_estimated * alpha_X_weight_to_b) + E_estimated - energies_dmrg


def discarded_weight_gradient_vector(param_vec, discarded_weights, energies_dmrg):
    """
    Gradient of the cost function for the discarded weight extrapolation.
    Has the form:
        ∇ = [dC/dα, dC/db, dC/dE_estimated]

    Args:
        param_vec: list of fit parameters [alpha, b, E_estimated]
        discarded_weights: list of discarded_weights
        energies_dmrg: list of DMRG energies
        alpha: fit parameter
        b: fit parameter
        E_estimated: fit parameter, energy to be estimated

    Returns:
        gradient: list of the gradient of the cost function
    """
    alpha, b, E_estimated = param_vec
    weight_to_b = np.power(discarded_weights, b)
    alpha_X_weight_to_b = alpha * weight_to_b
    central_term = E_estimated * alpha_X_weight_to_b + E_estimated - energies_dmrg
    dC_dalpha = 2 * np.sum(central_term * (E_estimated * weight_to_b))
    # np.log is the natural logarithm
    dC_db = 2 * np.sum(
        central_term * (E_estimated * alpha_X_weight_to_b * np.log(discarded_weights))
    )
    dC_dE_estimated = 2 * np.sum(central_term * (alpha_X_weight_to_b + 1))
    return [dC_dalpha, dC_db, dC_dE_estimated]


def discarded_weight_residuals_gradient_matrix(
    param_vec, discarded_weights, energies_dmrg
):
    """
    Gradient of the residuals r_i for the discarded weight extrapolation.
    Has the form:
        ∇ = [dr_i/dα, dr_i/db, dr_i/dE_estimated]

    Args:
        param_vec: list of fit parameters [alpha, b, E_estimated]
        discarded_weights: list of discarded_weights
        energies_dmrg: list of DMRG energies
        alpha: fit parameter
        b: fit parameter
        E_estimated: fit parameter, energy to be estimated

    Returns:
        gradient_matrix: (m,n) array of the gradient of the cost function.
            m = len(discarded_weights), n = 3
    """
    alpha, b, E_estimated = param_vec
    weight_to_b = np.power(discarded_weights, b)
    alpha_X_weight_to_b = alpha * weight_to_b
    central_term = E_estimated * alpha_X_weight_to_b + E_estimated - energies_dmrg
    dr_dalpha = central_term * (E_estimated * weight_to_b)
    # np.log is the natural logarithm
    dr_db = central_term * (
        E_estimated * alpha_X_weight_to_b * np.log(discarded_weights)
    )

    dr_dE_estimated = central_term * (alpha_X_weight_to_b + 1)
    gradient_matrix = np.vstack([dr_dalpha, dr_db, dr_dE_estimated]).T
    return gradient_matrix


def calc_coefficient_of_determination(
    x_data, y_data, predictor_fcn, predictor_fcn_args
):
    """
    Calculate the coefficient of determination (R^2) for a set of data and a predictor function.

    Args:
        x_data: list of independent variables
        y_data: list of dependent variables
        predictor_fcn: function that predicts the dependent variable from the independent variable

    Returns:
        R_squared: R^2 value of the fit
    """
    y_mean = np.mean(y_data)
    y_predicted = predictor_fcn(x_data, *predictor_fcn_args)
    SS_tot = np.sum((y_data - y_mean) ** 2) + 1e-16
    SS_res = np.sum((y_data - y_predicted) ** 2) + 1e-16
    R_squared = 1 - SS_res / SS_tot
    if np.isnan(R_squared):
        raise ValueError(
            f"R^2 is NaN. y_data: {y_data}, y_predicted: {y_predicted}, SS_res: {SS_res}, SS_tot: {SS_tot}"
        )
    return R_squared


def discarded_weight_predictor(discarded_weights, alpha, b):
    """
    Predict the relative energy from the discarded weight using the extrapolation function.
    ln ΔE_rel =a+b ln(δϵ)

    Args:
        discarded_weights: list of discarded_weights
        alpha: fit parameter, α = exp(a)
        b: fit parameter
        E_estimated: fit parameter, energy to be estimated

    Returns:
        predicted_relative energies: list of predicted ln ΔE_rel
    """
    a = np.log(alpha)  # α = exp(a), np.log is the natural logarithm
    ln_del_E_rel = a + b * np.log(discarded_weights)
    return ln_del_E_rel


def discarded_weight_predictor_ln(discarded_weights, a, b):
    """
    Predict the relative energy from the discarded weight using the extrapolation function.
    ln ΔE_rel =a+b ln(δϵ)

    Args:
        discarded_weights: list of discarded_weights
        a: fit parameter
        b: fit parameter
        E_estimated: fit parameter, energy to be estimated

    Returns:
        predicted_relative energies: list of predicted ln ΔE_rel
    """
    ln_del_E_rel = a + b * np.log(discarded_weights)
    return ln_del_E_rel


def plot_extrapolation(
    discarded_weights: List[float],
    energies_dmrg: List[float],
    fit_parameters: List[float],
    bond_dims: List[int] = None,
    plot_filename: str = None,
    figNum: int = 0,
):
    """
    Plot the extrapolation results.

    Args:
        discarded_weights: list of discarded_weights
        energies_dmrg: list of energies from DMRG
        fit_parameters: list of fit parameters
        plot_file: file to save the plot
    """
    l_w = 2.0
    fs = 35
    fs2 = 30
    fs3 = fs - 10
    discarded_weights_local = discarded_weights + 1e-30
    # ln ΔE_rel vs ln(δϵ) plot

    label_1 = r"$\log(\delta \epsilon)$"
    label_2 = r"$\log(\Delta E_{\mathrm{rel}})$"

    # plt.clf()
    fig = plt.figure(figNum, facecolor="white", figsize=6 * np.array([3, 2.25]))
    ax = plt.gca()

    rel_energies = np.abs(energies_dmrg - fit_parameters[-1]) / np.abs(
        fit_parameters[-1]
    )
    log.info(f"fit_parameters,plotting: {fit_parameters}")
    log.info(f"energies_dmrg,plotting: {energies_dmrg}")
    log.info(f"rel_energies,plotting: {rel_energies}")
    ax.plot(
        np.log(discarded_weights_local),
        np.log(rel_energies),
        "o",
        label="Data",
    )

    predicted_values = discarded_weight_predictor(
        discarded_weights_local, fit_parameters[0], fit_parameters[1]
    )
    ax.plot(np.log(discarded_weights_local), predicted_values, label="Fit", marker="s")

    ax.legend(fontsize=fs2)
    plt.xlabel(label_1, fontsize=fs, labelpad=10)
    # plt.ylabel(r"$\log(\mathrm{Extremum of Signal})$", fontsize=fs)
    plt.ylabel(label_2, fontsize=fs)

    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    # ax.xaxis.set_ticks([2,4,6,8,10,12,14,16,18,20])
    # plt.ylim(top=100)
    # plt.ylim(bottom=0.98,top=1.05)

    ax.tick_params(axis="both", width=l_w, which="both")
    ax.tick_params(axis="both", length=5, which="major")
    ax.tick_params(axis="both", length=3, which="minor")
    ##ax.yaxis.set_tick_params(width=l_w)
    ax.spines["top"].set_linewidth(l_w)
    ax.spines["bottom"].set_linewidth(l_w)
    ax.spines["left"].set_linewidth(l_w)
    ax.spines["right"].set_linewidth(l_w)

    # ax.legend(prop={'size': 20},
    # loc='upper left',
    # ncol=2,
    # )

    # ax.legend(bbox_to_anchor=(0.995, 0.9),
    # bbox_transform=ax.transAxes,
    # loc='upper right',
    # fontsize=fs2,
    # )

    fig.tight_layout()

    if plot_filename is not None:
        plot_filename_path = Path(str(plot_filename) + "_lnDeltaE.pdf")
        plot_filename_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_filename_path, format="pdf", dpi=300)

    # E_DMRG vs δϵ plot
    ################################
    fig = plt.figure(figNum + 1, facecolor="white", figsize=6 * np.array([3, 2.25]))
    ax = plt.gca()
    ax.plot(np.log(discarded_weights_local), energies_dmrg, "o", label="Data")
    predicted_values = (
        (fit_parameters[2])
        * fit_parameters[0]
        * np.power(discarded_weights_local, fit_parameters[1])
    ) + fit_parameters[2]
    ax.plot(np.log(discarded_weights_local), predicted_values, label="Fit", marker="s")
    ax.legend(fontsize=fs2)
    plt.xlabel(label_1, fontsize=fs, labelpad=10)
    plt.ylabel("E_DMRG", fontsize=fs)

    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    # ax.xaxis.set_ticks([2,4,6,8,10,12,14,16,18,20])
    # plt.ylim(top=100)
    # plt.ylim(bottom=0.98,top=1.05)

    ax.tick_params(axis="both", width=l_w, which="both")
    ax.tick_params(axis="both", length=5, which="major")
    ax.tick_params(axis="both", length=3, which="minor")
    ##ax.yaxis.set_tick_params(width=l_w)
    ax.spines["top"].set_linewidth(l_w)
    ax.spines["bottom"].set_linewidth(l_w)
    ax.spines["left"].set_linewidth(l_w)
    ax.spines["right"].set_linewidth(l_w)

    # ax.legend(prop={'size': 20},
    # loc='upper left',
    # ncol=2,
    # )

    # ax.legend(bbox_to_anchor=(0.995, 0.9),
    # bbox_transform=ax.transAxes,
    # loc='upper right',
    # fontsize=fs2,
    # )

    fig.tight_layout()

    if plot_filename is not None:
        plot_filename_path = Path(str(plot_filename) + "_EDMRG.pdf")
        plot_filename_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_filename_path, format="pdf", dpi=300)

    # E_DMRG vs 1/bond_dimension plot
    ################################
    fig = plt.figure(figNum + 2, facecolor="white", figsize=6 * np.array([3, 2.25]))
    ax = plt.gca()
    ax.plot(np.log(1 / np.array(bond_dims)), energies_dmrg, "o", label="Data")
    # predicted_values = (
    #     (fit_parameters[2])
    #     * fit_parameters[0]
    #     * np.power(1 / np.array(bond_dims), fit_parameters[1])
    # ) + fit_parameters[2]
    # ax.plot(1 / np.array(bond_dims), predicted_values, label="Fit", marker="s")
    ax.legend(fontsize=fs2)
    plt.xlabel(r"$1/\mathrm{bond\ dimension}$", fontsize=fs, labelpad=10)
    plt.ylabel("E_DMRG", fontsize=fs)

    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    # ax.xaxis.set_ticks([2,4,6,8,10,12,14,16,18,20])
    # plt.ylim(top=100)
    # plt.ylim(bottom=0.98,top=1.05)

    ax.tick_params(axis="both", width=l_w, which="both")
    ax.tick_params(axis="both", length=5, which="major")
    ax.tick_params(axis="both", length=3, which="minor")
    ##ax.yaxis.set_tick_params(width=l_w)
    ax.spines["top"].set_linewidth(l_w)
    ax.spines["bottom"].set_linewidth(l_w)
    ax.spines["left"].set_linewidth(l_w)
    ax.spines["right"].set_linewidth(l_w)

    # ax.legend(prop={'size': 20},
    # loc='upper left',
    # ncol=2,
    # )

    # ax.legend(bbox_to_anchor=(0.995, 0.9),
    # bbox_transform=ax.transAxes,
    # loc='upper right',
    # fontsize=fs2,
    # )

    fig.tight_layout()

    if plot_filename is not None:
        plot_filename_path = Path(str(plot_filename) + "_EDMRG_bond_dims.pdf")
        plot_filename_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_filename_path, format="pdf", dpi=300)


def bond_dimension_fitting(
    bond_dims, dmrg_energies, exact_energy, extrap_threshold=1e-3
):
    # Using Eqn 14 of http://dx.doi.org/10.1016/j.cpc.2014.01.019
    y_values = np.array(dmrg_energies)
    y_values = np.log(y_values - exact_energy)  # natural log

    x_values = np.array(bond_dims)
    x_values = (np.log(x_values)) ** 2  # natural log

    # Fit the data to a line using scipy
    result = scipy.stats.linregress(x_values, y_values)
    slope = result.slope
    intercept = result.intercept
    R_squared = result.rvalue**2
    slope_stderr = result.stderr
    intercept_stderr = result.intercept_stderr

    # Extrapolate the bond dimension to E - E_exact = extrap_threshold
    exponent = np.sqrt((np.log(extrap_threshold) - intercept) / slope)

    # Ensure that the exponent is real and positive
    assert np.isreal(exponent), f"Exponent is complex: {exponent}"
    assert (
        exponent >= 0
    ), f"Exponent is negative: {exponent}, {extrap_threshold}, {intercept}, {slope}, (x_values, y_values): ({x_values}, {y_values})"

    extrapolated_bd = np.exp(exponent)

    # Calculate the error on the extrapolated bond dimension
    max_intercept = intercept + 1.96 * intercept_stderr
    min_intercept = intercept - 1.96 * intercept_stderr
    max_slope = slope + 1.96 * slope_stderr
    min_slope = slope - 1.96 * slope_stderr
    pos_pos_bd = np.exp(np.sqrt((np.log(extrap_threshold) - max_intercept) / max_slope))
    pos_neg_bd = np.exp(np.sqrt((np.log(extrap_threshold) - min_intercept) / max_slope))
    neg_pos_bd = np.exp(np.sqrt((np.log(extrap_threshold) - max_intercept) / min_slope))
    neg_neg_bd = np.exp(np.sqrt((np.log(extrap_threshold) - min_intercept) / min_slope))
    max_bd = max(pos_pos_bd, pos_neg_bd, neg_pos_bd, neg_neg_bd)
    min_bd = min(pos_pos_bd, pos_neg_bd, neg_pos_bd, neg_neg_bd)

    return (
        slope,
        intercept,
        R_squared,
        slope_stderr,
        intercept_stderr,
        extrapolated_bd,
        max_bd,
        min_bd,
    )


def discarded_weight_linear_fitting(discarded_weights, dmrg_energies):
    y_values = np.array(dmrg_energies)
    x_values = np.array(discarded_weights)

    # Fit the data to a line using scipy
    result = scipy.stats.linregress(x_values, y_values)
    slope = result.slope
    intercept = result.intercept
    R_squared = result.rvalue**2
    slope_stderr = result.stderr
    intercept_stderr = result.intercept_stderr

    return slope, intercept, R_squared, slope_stderr, intercept_stderr
