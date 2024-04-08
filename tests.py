def test_plot_confregion_univariate_t():
    """
    simple unit test to make sure the limit nu --> infty of the studen t-distribution (Gaussian) is correct.
    Raises an error if not.
    """
    mu = 4
    Sigma = 2
    nu = 90000  # hence, approx. a normal distribution
    alpha = (0.6826894921370859, 0.9544997361036416, 0.9973002039367398)
    from modules.plot_helpers import plot_confregion_univariate_t 
    conf_intervals = plot_confregion_univariate_t(mu=mu, Sigma=Sigma, nu=nu, alpha=alpha, validate=True)
    

def test_fit_bivariate_t(df=7, M=None, mu=None, size=1000000, tol=1e-2, print_status=True):
    """
    simple test of `fit_bivariate_t()` using brute-force sampling `size` times. 
    `df`, `M`, `mu` specify the dof, shape matrix, and mean vector of the t-distribution used to generate
    mock data;

    raises an error if the test doesn't pass
    """
    import numpy as np
    nu_limits = (3, 8)
    if np.isfinite(df) and df > nu_limits[1]:
        print(f"requested (finite) df={df} too high")
        return

    if M is None:
        M = np.array([[1, 0], [0, 2]])
    if mu is None:
        mu = np.array([1, 2])
    if np.isfinite(df):
        is_normal_distr = False
        from scipy.stats import multivariate_t
        data = multivariate_t.rvs(df=df, loc=mu, shape=M, size=size)
    else:
        is_normal_distr = True
        from scipy.stats import multivariate_normal
        data = multivariate_normal.rvs(mean=mu, cov=M, size=size)

    from modules.plot_helpers import fit_bivariate_t 
    fit = fit_bivariate_t(data, nu_limits=nu_limits, print_status=print_status)

    stat_mu = np.abs(mu-fit["mu"]) < tol
    stat_psi = np.abs(M-fit["Psi"]) < tol
    stat_nu = np.abs(df-fit["nu"]) < tol if not is_normal_distr else True
    assert np.any(stat_mu) and np.any(stat_psi) and stat_nu

from modules.StatisticalModel import StatisticalModel
from modules.priors import used_prior_sets
from modules.SaturationAnalysis import drischler_satbox

def test_analysis():
    for prior_set in used_prior_sets:
        model = StatisticalModel(data=drischler_satbox.data_frame, prior_params=prior_set)
        fig, _ = model.plot_predictives(validate=True)

def test_2():
    for prior_set in used_prior_sets:
        model = StatisticalModel(data=drischler_satbox.data_frame, prior_params=prior_set)
        model.sample(num_samples=1, kind="predictive_y", based_on="posterior", random_state=None, validate=True)

def test_3():
    for prior_set in used_prior_sets:
        model = StatisticalModel(data=drischler_satbox.data_frame, prior_params=prior_set)
        for dist in ("posterior", "prior"):
            assert model.sanity_check(num_samples=100000, based_on=dist, 
                                      do_print=False, quantile_values=None, atol=5e-2, rtol=0.)


if __name__ == "__main__":
    test_fit_bivariate_t()