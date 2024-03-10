import unittest

import numpy as np
import numpy.testing as npt

# Import the module you want to test
import dmrghandler.energy_extrapolation as energy_extrapolation

test_rtol = 1e-5
test_atol = 1e-8
test_rsquared_rtol = 1e-1
test_rsquared_atol = 1e-1


class TestExtrapolation(unittest.TestCase):
    def test_whole_fit_discarded_weight(self):
        """
        Test the whole fit of the discarded weight extrapolation
        """
        # Generate some fake data
        # discarded_weights = [5e-1, 1e-1, 1e-2, 1e-3, 1e-5]
        # exact_energy = -0.5
        # alpha = 1.0  # alpha = exp(a), a = -0.6931471805599453
        # b = 1.5

        # Set up rng
        rng = np.random.default_rng(12345)
        rng_range = 0.5
        discarded_weights_base = np.array([5e-1, 1e-1, 1e-2, 1e-3, 1e-5, 1e-7])
        discarded_weights_2dList = [
            discarded_weights_base,
            discarded_weights_base
            * (1 + rng.uniform(-rng_range, rng_range, len(discarded_weights_base))),
            discarded_weights_base
            * (1 + rng.uniform(-rng_range, rng_range, len(discarded_weights_base))),
            discarded_weights_base
            * (1 + rng.uniform(-rng_range, rng_range, len(discarded_weights_base))),
        ]
        exact_energy_list = [-0.5, -1.354, -8.654]
        alpha_list = [1.0, 0.5, 0.29874]
        b_list = [1.5, 1.0, 3.18564]

        for discarded_weights in discarded_weights_2dList:
            for exact_energy in exact_energy_list:
                for alpha in alpha_list:
                    for b in b_list:
                        # print(
                        #     "-------------------------------------------------------------"
                        # )
                        # print(
                        #     "-------------------------------------------------------------"
                        # )
                        # print(f"discarded_weights: {discarded_weights}")
                        # print(f"exact_energy: {exact_energy}")
                        # print(f"alpha: {alpha}")
                        # print(f"b: {b}")

                        lnDE_rel = energy_extrapolation.discarded_weight_predictor(
                            discarded_weights, alpha, b
                        )
                        energies_dmrg = exact_energy * (1 + np.exp(lnDE_rel))

                        # print(f"lnDE_rel: {lnDE_rel}")
                        # print(f"energies_dmrg: {energies_dmrg}")

                        # Fit the data
                        # past_parameters = [0.55, 1.4, -0.54]
                        past_parameters = None

                        result_obj, energy_estimated, fit_parameters, R_squared = (
                            energy_extrapolation.dmrg_energy_extrapolation(
                                energies_dmrg=energies_dmrg,
                                independent_vars=discarded_weights,
                                extrapolation_type="discarded_weight",
                                past_parameters=past_parameters,
                                verbosity=0,
                            )
                        )

                        # Check the fit parameters
                        # print(f"Fit parameters: {fit_parameters}")
                        # print(f"R^2 = {R_squared}")
                        npt.assert_allclose(
                            fit_parameters[0], alpha, rtol=test_rtol, atol=test_atol
                        )
                        npt.assert_allclose(
                            fit_parameters[1], b, rtol=test_rtol, atol=test_atol
                        )
                        npt.assert_allclose(
                            energy_estimated,
                            exact_energy,
                            rtol=test_rtol,
                            atol=test_atol,
                        )
                        # print(f"R^2 = {R_squared}")
                        # These R^2 values, based on the ln ln formulation, are not always good
                        # npt.assert_allclose(
                        #     R_squared,
                        #     1.0,
                        #     rtol=test_rsquared_rtol,
                        #     atol=test_rsquared_atol,
                        # )


if __name__ == "__main__":
    unittest.main()
