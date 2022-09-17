import numpy as np
from scipy.stats import invwishart, multivariate_normal, multivariate_t


class StatisticalModel:
    def __init__(self, data, quantities, prior_params):
        self.data = data
        self.quantities = quantities
        self.prior_params = prior_params

    @property
    def n(self):
        """
        Number of data points used for inference
        """
        return len(self.data)

    @property
    def d(self):
        """
        Dimensionality of the statistical model;
        for the saturation point, d = 2 [i.e, n0, E0/A]
        """
        return len(self.quantities)

    @property
    def sample_mean(self):
        """
        returns the sample mean vector \bar{y} of length d
        """
        return self.data[self.quantities].mean()

    @property
    def sum_squared_dev(self):
        """
        returns the sum of the squared deviations matrix, i.e., the symmetric S matrix (d x d)
        """
        diff = self.data[self.quantities] - self.sample_mean
        return np.sum([np.outer(row.to_numpy(), row.to_numpy()) for irow, row in diff.iterrows()], axis=0)

    @staticmethod
    def _validate_matrix(mat, raise_error=False, rtol=1e-05, atol=1e-08):
        """
        Checks that the matrix `mat` is symmetric and positive semi-definite
        :param mat: matrix
        :param raise_error: raise error if not symmetric and positive semi-definite
        :param rtol: relative tolerance used for comparison
        :param atol: absolute tolerance used for comparision
        :return: returns boolean result of the validation
        """
        stat_sym = np.allclose(mat, mat.T, rtol=rtol, atol=atol)
        stat_pos_semi_def = np.all(np.linalg.eigvals(mat) >= 0)
        stat = stat_sym and stat_pos_semi_def
        if not stat and raise_error:
            raise ValueError("Non-symmetric matrix encountered.")
        else:
            return stat

    @property
    def posterior_params(self):
        """
        Parameters of the posterior distribution; both the posterior and prior are normal inverse-Wishart distributions
        For the analytic expressions, see
            * https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution#Posterior_distribution_of_the_parameters
            * Equations (250) through (254) in Murphy's notes: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
            * page 73 (pdf page 83) in Gelman et al.'s book: http://www.stat.columbia.edu/~gelman/book/BDA3.pdf

        :return: parameters of the distribution
        """
        ret = dict()
        ret["kappa"] = self.prior_params["kappa"] + self.n
        ret["nu"] = self.prior_params["nu"] + self.n
        ret["mu"] = np.array([self.prior_params["kappa"], self.n]) @ np.array([self.prior_params["mu"], self.sample_mean]) /ret["kappa"]
        diff = self.sample_mean - self.prior_params["mu"]
        ret["Psi"] = self.prior_params["Psi"] + self.sum_squared_dev + (self.prior_params["kappa"] * self.n / ret["kappa"]) * np.outer(diff, diff)
        StatisticalModel._validate_matrix(ret["Psi"], raise_error=True)
        return ret

    def sample_mu_Sigma(self, num_samples=1, based_on="posterior", random_state=None):
        """
        Samples the posterior pr(\mu, \Sigma), which is a normal-inverse-Wishart distribution, in a two-step process.
        For more information on the sampling, see:
            * https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution#Generating_normal-inverse-Wishart_random_variates
            * page 73 (pdf page 83) in Gelman et al.'s book: http://www.stat.columbia.edu/~gelman/book/BDA3.pdf

        :param num_samples: number of samples
        :based_on: either "posterior" or "prior"
        :param random_state: state of the random number generator
        :return: returns `num_samples` samples of either the posterior or prior
        """
        if based_on not in ("posterior", "prior"):
            raise ValueError(f"Got unknown prior/posterior request '{based_on}'.")
        params = self.posterior_params if based_on=="posterior" else self.prior_params
        Sigmas = invwishart.rvs(df=params["nu"], scale=params["Psi"], size=num_samples, random_state=None)
        # TODO: note that `params["Psi"]` vs `np.linalg.inv(params["Psi"])` in the previous line (see also TODO below)
        if num_samples == 1:
            Sigmas = np.expand_dims(Sigmas, 0)
        mus = np.array([multivariate_normal.rvs(mean=params["mu"], cov=Sigma/params["kappa"],
                                                size=1, random_state=random_state) for Sigma in Sigmas])
        return mus, Sigmas

    def sample(self, num_samples=1, kind="predictive_y", based_on="posterior", random_state=None, validate=True):
        """
        Samples prior, posterior, or marginal distribution functions, if available
        For the analytic expressions, see
            * https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution#Posterior_distribution_of_the_parameters
            * Chapter 9 in Murphy's notes: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
            * see page 73 (pdf page 83) in Gelman et al.'s book: http://www.stat.columbia.edu/~gelman/book/BDA3.pdf

        :param num_samples: number of requested samples
        :param kind: "predictive_y", "marginal_mu", "marginal_Sigma", or "none"
        :param based_on: either "posterior" or "prior"
        :param random_state: state of random number generator
        :param validate: validate that the covariance matrix is symmetric and positive semi-definite
        :return: samples in an array with columns associated with (density, energy/particle)
        """
        # validate inputs for 'kind' and 'based_on'
        if kind not in ("predictive_y", "marginal_mu", "marginal_Sigma", "none"):
            raise ValueError(f"Got unknown 'kind' request '{kind}'.")
        if based_on not in ("posterior", "prior"):
            raise ValueError(f"Got unknown 'based_on' request '{based_on}'.")
        #if kind =="predictive_y" and based_on == "prior":  # TODO: Is our implementation of the prior predictive correct?
        #    raise ValueError("Predictive prior not implemented.")

        # select parameter sets based on posterior or prior
        params = self.posterior_params if based_on == "posterior" else self.prior_params

        # sample requested distribution and return results
        if kind in ("none", "marginal_Sigma"):
            mus, Sigmas = self.sample_mu_Sigma(num_samples=num_samples, based_on=based_on, random_state=random_state)
            if kind == "none":  # plain prior or posterior requested; i.e., no marginalization involved
                return mus, Sigmas
            else:  # marginal_Sigma
                return Sigmas  # Eq. (255) in Murphy's notes #TODO: understand here why Murphy and Gelman have the matrix inverse but not Wikipedia
        else:  # "predictive_y" or "marginal_mu"
            extra_factor = (params["kappa"]+1) if kind=="predictive_y" else 1.
            shape_matrix = params["Psi"]*extra_factor/(params["kappa"]*(params["nu"]-self.d+1))
            if validate:
                StatisticalModel._validate_matrix(shape_matrix, raise_error=True)
            return multivariate_t.rvs(df=params["nu"] - self.d + 1,
                                      loc=params["mu"],
                                      shape=shape_matrix,
                                      size=num_samples, random_state=random_state)  # Eqs. (256) and (258) in Murphy's notes

    def sample_predictive_bf(self, num_samples=1, based_on="posterior", random_state=None):
        """
        Samples the posterior predictive brute-force in the two-step process described in
        https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution#Posterior_distribution_of_the_parameters

        :param num_samples: number of requested samples
        :param based_on: either using the prior or posterior
        :param random_state: state of random number generator
        :return: `num_samples` samples in an array with columns associated with (density, energy/particle)
        """
        mus, Sigmas = self.sample_mu_Sigma(num_samples=num_samples, based_on=based_on, random_state=random_state)
        return np.array([multivariate_normal.rvs(mean=mu, cov=Sigma, size=1) for mu, Sigma in zip(mus, Sigmas)])

    def sanity_check(self, num_samples=10000, based_on="posterior", do_print=False, quantile_values=None, atol=5e-2, rtol=0.):
        """
        Checks that the quantiles obtained from sampling the (analytic) posterior predictive matches the
        brute-force calculation described in https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution#Posterior_distribution_of_the_parameters

        :param num_samples: number of samples
        :param based_on: either using the prior or posterior
        :param do_print: print intermediate results
        :param quantile_values: requested quantiles for comparison
        :param atol: absolute tolerance of comparison
        :param rtol: relative tolerance of comparison
        :return: boolean result of the comparison
        """
        quantile_values = quantile_values if quantile_values else (0.05, 0.5, 0.95)
        samples = self.sample(num_samples=num_samples, kind="predictive_y", based_on=based_on)
        quantiles_cp = np.quantile(samples, q=quantile_values, axis=0)
        samples_bf = self.sample_predictive_bf(num_samples=num_samples, based_on=based_on)
        quantiles_bf = np.quantile(samples_bf, q=quantile_values, axis=0)
        stat = np.allclose(quantiles_cp, quantiles_bf, atol=atol, rtol=rtol)
        if do_print:
            print(f"sanity check based on predictive '{based_on}'")
            print("quantiles conj. prior:", quantiles_cp)
            print("quantiles brute-force:", quantiles_bf)
            print("passed:", stat)
        return stat