import numpy as np


standard_prior_off_diag = 0.  # note that "Psi" should be a symmetric, pos. def matrix (will be checked)
standard_prior_params = {"mu": np.array([0.16, -15.9]),
                         "Psi": np.array([[0.01**2, standard_prior_off_diag],
                                          [standard_prior_off_diag, 0.32**2]]),
                         "kappa": 1, "nu": 4}

setA_prior_params = {"mu": np.array([0.16, -15.9]),
                     "Psi": np.array([[0.01**2, standard_prior_off_diag],
                                      [standard_prior_off_diag, 0.32**2]]),
                     "kappa": 1, "nu": 4}

setB_prior_params = {"mu": np.array([0.16, -15.9]),
                     "Psi": np.array([[0.01**2, standard_prior_off_diag],
                                      [standard_prior_off_diag, 0.32**2]]),
                     "kappa": 1, "nu": 4}

setC_prior_params = {"mu": np.array([0.16, -15.9]),
                    "Psi": np.array([[0.01**2, standard_prior_off_diag],
                                  [standard_prior_off_diag, 0.32**2]]),
                    "kappa": 1, "nu": 4}