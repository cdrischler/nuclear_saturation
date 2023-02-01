import numpy as np


standard_prior_off_diag = 0.  # note that "Psi" should be a symmetric, pos. def matrix (will be checked)
standard_prior_params = {"mu": np.array([0.16, -15.9]),
                         "Psi": np.array([[0.01**2, standard_prior_off_diag],
                                          [standard_prior_off_diag, 0.32**2]]),
                         "kappa": 1, "nu": 10, "label": "Standard Prior"}


setA_prior_params = {"mu": np.array([0.16, -15.9]),
                     "Psi": np.array([[0.008**2, standard_prior_off_diag],
                                      [standard_prior_off_diag, 0.32**2]]),
                     "kappa": 3, "nu": 3, "label": "Set A"}

setB_prior_params = {"mu": np.array([0.16, -15.9]),
                     "Psi": np.array([[0.006**2, standard_prior_off_diag],
                                      [standard_prior_off_diag, 0.62**2]]),
                     "kappa": 3, "nu": 3, "label": "Set B"}

setC_prior_params = {"mu": np.array([0.16, -15.9]),
                     "Psi": np.array([[0.006**2, standard_prior_off_diag],
                                      [standard_prior_off_diag, 0.62**2]]),
                     "kappa": 3, "nu": 3, "label": "Set C"}

used_prior_sets = [setA_prior_params, setB_prior_params, setC_prior_params]

# Note (taken from https://en.wikipedia.org/wiki/Conjugate_prior)

# * mean was estimated from \kappa _{0} observations
# with sample mean {\boldsymbol {\mu }}_{0};

# * covariance matrix was estimated from \nu _{0} observations
# with sample mean {\boldsymbol {\mu }}_{0} and with sum of
# pairwise deviation products {\boldsymbol {\Psi }}=\nu _{0}{\boldsymbol {\Sigma }}_{0}

# The latter implies for the prior predictive that the covariance matrix
# goes to zero as nu goes to infinity; i.e., the bivariate t distribution
# collapses to point (=normal distribution with zero covariance matrix)

#%%
