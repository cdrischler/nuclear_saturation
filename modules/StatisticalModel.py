import numpy as np
import pandas as pd
from scipy.stats import invwishart, multivariate_normal, multivariate_t
import matplotlib.pyplot as plt
from modules.plot_helpers import colors
from modules.plot_helpers import plot_confregion_bivariate_t
from modules.priors import standard_prior_params


class StatisticalModel:
    def __init__(self, data, quantities=None, prior_params=None):
        assert data is None or isinstance(data, pd.DataFrame), "`data` for StatisticalModel must be None or a Dataframe"
        self.data = data
        self.quantities = quantities if quantities else ["rho0", "E/A"]
        self.prior_params = prior_params if prior_params else standard_prior_params
        self._validate_matrix(self.prior_params["Psi"], raise_error=True)

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
        returns the sum of the squared deviations matrix, i.e., the symmetric S matrix (d x d);
        in other words, returns the un-normalized covariance matrix
        """
        diff = self.data[self.quantities] - self.sample_mean
        return np.sum([np.outer(row.to_numpy(), row.to_numpy()) for irow, row in diff.iterrows()], axis=0)
        # or just # return (len(self.data) - 1) * np.cov(self.data[self.quantities[0]], self.data[self.quantities[1]])

    @staticmethod
    def _validate_matrix(mat, raise_error=False, rtol=1e-05, atol=1e-08):
        """
        Checks that the matrix `mat` is symmetric and positive semi-definite
        :param mat: matrix
        :param raise_error: raise error if not symmetric and positive semi-definite
        :param rtol: relative tolerance used for comparison
        :param atol: absolute tolerance used for comparison
        :return: returns boolean result of the validation
        """
        stat_sym = np.allclose(mat, mat.T, rtol=rtol, atol=atol)
        stat_pos_semi_def = np.all(np.linalg.eigvals(mat) >= 0)
        stat = stat_sym and stat_pos_semi_def
        if not stat and raise_error:
            raise ValueError("Non-symmetric and/or non-pos.-def. matrix encountered.")
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

        # if no data is provided, by definition, the posterior equals the prior
        if self.data is None:
            return self.prior_params

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

        # select parameter sets based on posterior or prior
        params = self.posterior_params if based_on == "posterior" else self.prior_params

        # sample requested distribution and return results
        if kind in ("none", "marginal_Sigma"):
            mus, Sigmas = self.sample_mu_Sigma(num_samples=num_samples, based_on=based_on, random_state=random_state)
            if kind == "none":  # plain prior or posterior requested; i.e., no marginalization involved
                return mus, Sigmas
            else:  # marginal_Sigma
                return Sigmas  # Eq. (255) in Murphy's notes # TODO: understand here why Murphy and Gelman have the matrix inverse but not Wikipedia
        else:  # "predictive_y" or "marginal_mu"
            extra_factor = (params["kappa"]+1) if kind=="predictive_y" else 1.
            shape_matrix = params["Psi"]*extra_factor/(params["kappa"]*(params["nu"]-self.d+1))
            if validate:
                StatisticalModel._validate_matrix(shape_matrix, raise_error=True)
            return multivariate_t.rvs(df=params["nu"] - self.d + 1,
                                      loc=params["mu"],
                                      shape=shape_matrix,
                                      size=num_samples, random_state=random_state)  # Eqs. (256) and (258) in Murphy's notes

    def sample_predictive_bf(self, num_samples=1, num_samples_mu_Sigma=1,
                             return_predictive_only=True, based_on="posterior", random_state=None):
        """
        Samples the posterior predictive brute-force in the two-step process described in
        https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution#Posterior_distribution_of_the_parameters

        :param num_samples: number of requested samples of the predictive prior/posterior
        :param num_samples_mu_Sigma: number of requested samples of (mu, Sigma)
        :param based_on: either using the prior or posterior
        :param return_predictive_only: returns prior/posterior predictive samples but not the associated samples of (mu, Sigma)
        :param random_state: state of random number generator
        :return: `num_samples` samples in an array with columns associated with (density, energy/particle)
        or pandas DataFrame with full set of results (predictive_y, my, Sigma) depending on the argument `return_predictive_only`
        """
        mus, Sigmas = self.sample_mu_Sigma(num_samples=num_samples_mu_Sigma, based_on=based_on, random_state=random_state)
        predictive = np.array([multivariate_normal.rvs(mean=mu, cov=Sigma, size=num_samples) for mu, Sigma in zip(mus, Sigmas)])
        if return_predictive_only:
            return predictive
        else:
            return pd.DataFrame(data={"predictive rho0": predictive[:, 0],
                                      "predictive E/A": predictive[:, 1],
                                      "mu rho0": mus[:, 0], "mu E/A": mus[:, 1],
                                      "Sigma (0,0)": Sigmas[:, 0, 0],
                                      "Sigma (0,1)": Sigmas[:, 0, 1],  # symmetric matrix
                                      "Sigma (1,1)": Sigmas[:, 1, 1]})

    def sanity_check(self, num_samples=100000, based_on="posterior", do_print=False, quantile_values=None, atol=5e-2, rtol=0.):
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
        samples_bf = self.sample_predictive_bf(num_samples_mu_Sigma=num_samples, based_on=based_on)
        quantiles_bf = np.quantile(samples_bf, q=quantile_values, axis=0)
        stat = np.allclose(quantiles_cp, quantiles_bf, atol=atol, rtol=rtol)
        if do_print:
            print(f"sanity check based on predictive '{based_on}'")
            print("quantiles conj. prior:", quantiles_cp)
            print("quantiles brute-force:", quantiles_bf)
            print("passed:", stat)
        return stat

    def predictives_params(self, based_on="posterior" ):
        params = self.posterior_params if based_on == "posterior" else self.prior_params
        # Eq. (258) in Murphy's notes: https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        df = params["nu"] - self.d + 1
        shape_matrix = params["Psi"]*(params["kappa"]+1)/(params["kappa"]*df)
        return df, params["mu"], shape_matrix

    def plot_predictives(self, plot_data=True, levels=None,
                         set_xy_limits=True, set_xy_lbls=True, place_legend=True, validate=False):
        """
        plots the confidence regions associated with the predictive prior (left panel)
        and prior predictive (right panel); these are bivariate t distributions

        :param plot_data: add scatter plot of underlying data (boolean)
        :param levels: confidence level, array-like
        :param set_xy_limits: set x/y axis limits to default values
        :param set_xy_lbls: set x/y axis labels to default labels
        :param place_legend: adds a legend to the panels
        :param validate: validate confidence ellipses
        :return: None
        """
        from plot_helpers import cm
        fig, axs = plt.subplots(1, 2, figsize=(17.8*cm, 8.6*cm), sharex=True, sharey=True,
                                tight_layout=True)
        fig.tight_layout(pad=1.5)

        for ibon, bon in enumerate(("prior", "posterior")):
            ax = axs[ibon]
            df, mu, shape_matrix = self.predictives_params(bon)
            plot_confregion_bivariate_t(ax=ax, mu=mu,
                            Sigma=shape_matrix, nu=df,
                            alpha=levels, alpha_unit="decimal", num_pts=10000000,
                            plot_scatter=False, validate=validate)
            ax.set_title(f"{bon} predictive")
            if plot_data and self.data is not None:
                ax.scatter(self.data["rho0"], self.data["E/A"], s=4, c=colors[5])

            from modules.SaturationAnalysis import drischler_satbox
            drischler_satbox.plot(ax=ax, plot_scatter=False, plot_box_estimate=True,
                                  add_axis_labels=False, place_legend=False)

        axs[0].text(0.05, 0.9, f"{self.prior_params['label']}", transform=axs[0].transAxes)

        if set_xy_limits:
            self.set_xy_lim(axs[0])

        if set_xy_lbls:
            for ax in axs:
                ax.set_xlabel(self.xlabel)
            axs[0].set_ylabel(self.ylabel)

        if place_legend:
            axs[1].legend(ncol=2, title="confidence level", prop={'size': 8}, frameon=False)

        return fig, axs

    def plot_predictives_corner(self, plot_data=True, levels=None,
                                set_xy_limits=True, set_xy_lbls=True, place_legend=True, validate=False):
        """

        :param axs: axes used for plotting, array-like
        :return: None
        """
        ret_array = []
        for ibon, bon in enumerate(("prior", "posterior")):
            from plot_helpers import cm
            fig, axs = plt.subplots(2, 2, figsize=(9*cm, 1.2*8.6*cm), sharex=False, sharey=False)

            # upper right panel
            axs[0, 1].grid(False)
            axs[0, 1].axis('off')
            axs[0, 1].text(0.05, 0.9, f"{bon} predictive", transform=axs[0, 1].transAxes)  #transAxes)
            axs[0, 1].text(0.05, 0.83, f"({self.prior_params['label'].lower()})", transform=axs[0, 1].transAxes)  # transAxes)
            fig.tight_layout(pad=.5)

            # lower left panel
            df, mu, shape_matrix = self.predictives_params(bon)
            plot_confregion_bivariate_t(ax=axs[1, 0], mu=mu,
                                        Sigma=shape_matrix, nu=df,
                                        alpha=levels, alpha_unit="decimal", num_pts=10000000,
                                        plot_scatter=False, validate=validate, edgecolor='k', facecolor="None")

            if plot_data and self.data is not None:
                axs[1, 0].scatter(self.data["rho0"], self.data["E/A"], s=4, c=colors[5])

            from modules.SaturationAnalysis import drischler_satbox
            drischler_satbox.plot(ax=axs[1, 0], plot_scatter=False, plot_box_estimate=True,
                                  add_axis_labels=False, place_legend=False)
            box_params = drischler_satbox.box_estimate()

            # diagonal panels
            R = np.linalg.cholesky(shape_matrix)
            from scipy.stats import t
            for idiag, diag in enumerate(np.diag(axs)):
                sigma = np.linalg.norm(R[idiag, :])
                if idiag == 0:
                    x = np.linspace(0.12, 0.20, 1000)
                    y = t.pdf(x, df, loc=mu[idiag], scale=sigma) * sigma
                else:
                    y = np.linspace(-18, -12, 1000)
                    x = t.pdf(y, df, loc=mu[idiag], scale=sigma) * sigma

                diag.plot(x, y, c="darkgray", ls='-', lw=2,
                          alpha=1, label='t pdf')

            # t.cdf(x, df, loc=0, scale=1)  # TODO: plot C.I. for marginal distributions

            # empirical saturation range
            axs[0, 0].axvspan(box_params["rho0"][0]-box_params["rho0"][1],
                         box_params["rho0"][0]+box_params["rho0"][1],
                         zorder=-1, alpha=0.5, color='lightgray')

            axs[1, 1].axhspan(box_params["E/A"][0]-box_params["E/A"][1],
                              box_params["E/A"][0]+box_params["E/A"][1],
                              zorder=-1, alpha=0.5, color='lightgray')

            axs[0,0].set_xlim(0.13, 0.18)
            axs[1,0].set_xlim(0.13, 0.18)

            axs[1,0].set_ylim(-16.5, -15.00)
            axs[1,1].set_ylim(-16.5, -15.00)

            axs[0,0].axes.xaxis.set_ticklabels([])
            # axs[0,0].axes.yaxis.set_ticklabels([])
            # axs[1,1].axes.xaxis.set_ticklabels([])
            axs[1,1].axes.yaxis.set_ticklabels([])

            axs[1, 0].set_xlabel(self.xlabel)
            axs[1, 0].set_ylabel(self.ylabel)

            if place_legend:
                axs[1, 0].legend(ncol=2, title="confidence level",
                                 prop={'size': 8}, frameon=False,
                                 bbox_to_anchor=(1.8, 1.5), loc='center')

            ret_array.append([fig, axs])
        return ret_array

    @staticmethod
    def set_xy_lim(ax):
        ax.set_xlim(0.145, 0.175)
        ax.set_ylim(-16.5, -15.00)

    @property
    def xlabel(self):
        return 'Saturation Density $n_0$ [fm$^{-3}$]'

    @property
    def ylabel(self):
        return 'Saturation Energy $E_0/A$ [MeV]'

    def set_xy_lbls(self, ax):
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)


def model_from_scenario(scenario, quantities=None, prior_params=None):
    sampled_dft_constraints = pd.DataFrame()
    # print(scenario.label)
    for task in scenario.configs:
        if task.sample_from_model:
            sampled_dft_constraints = task.data_set.sample_from_model(df=sampled_dft_constraints,
                                                                      **task.sample_kwargs,
                                                                      **task.sample_from_model_kwargs)
        else:
            sampled_dft_constraints = task.data_set.sample(df=sampled_dft_constraints, **task.sample_kwargs)
    return StatisticalModel(data=sampled_dft_constraints,
                            quantities=quantities, prior_params=prior_params)
#%%
