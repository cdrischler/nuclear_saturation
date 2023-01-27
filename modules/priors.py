import numpy as np


standard_prior_off_diag = 0.  # note that "Psi" should be a symmetric, pos. def matrix (will be checked)
standard_prior_params = {"mu": np.array([0.16, -15.9]),
                         "Psi": np.array([[0.01**2, standard_prior_off_diag],
                                          [standard_prior_off_diag, 0.32**2]]),
                         "kappa": 1, "nu": 10}

setA_prior_params = {"mu": np.array([0.16, -15.9]),
                     "Psi": np.array([[0.006**2, standard_prior_off_diag],
                                      [standard_prior_off_diag, 0.62**2]]),
                     "kappa": 1, "nu": 4, "label": "Set A"}

setB_prior_params = {"mu": np.array([0.16, -15.9]),
                     "Psi": np.array([[0.006**2, standard_prior_off_diag],
                                      [standard_prior_off_diag, 0.62**2]]),
                     "kappa": 10, "nu": 4, "label": "Set B"}

setC_prior_params = {"mu": np.array([0.16, -15.9]),
                     "Psi": np.array([[0.006**2, standard_prior_off_diag],
                                      [standard_prior_off_diag, 0.62**2]]),
                     "kappa": 10, "nu": 10, "label": "Set C"}

setD_prior_params = {"mu": np.array([0.16, -15.9]),  # normal distribution
                     "Psi": np.array([[0.006**2, standard_prior_off_diag],
                                      [standard_prior_off_diag, 0.62**2]]),
                     "kappa": 1/100, "nu": 100, "label": "Set D"}

used_prior_sets = [setA_prior_params, setB_prior_params, setC_prior_params, setD_prior_params]

#%%
