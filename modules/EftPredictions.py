import pandas as pd
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import matplotlib as mpb
import corner
import numpy as np
from modules.plot_helpers import cm


class EftPredictions:
    def __init__(self, filename=None, show_result=False):
        filename = filename if filename else "data/satpoints_predicted.csv"
        self.data = self.read_data(filename)
        self.model, self.trace = self.fit(show_result=show_result)
        self.x_validate = np.linspace(0.14, 0.20, 10)
        self.posterior_predictive_sampled = self.posterior_predictive(self.x_validate)

    def read_data(self, filename):
        data = pd.read_csv(filename)
        return data[((data["method"] == "MBPT") & (data["mbpt_order"] == 4)) | (data["method"] == "MBPT*")]

    def fit(self, draws=10000, tune=2000, target_accept=.95, show_result=False):
        print("Performing Bayesian linear regression on EFT predictions (Coester Band)")
        with pm.Model() as model:
            x_data = pm.Data("x_data", self.data["n0"])
            beta_0 = pm.Normal("beta_0", mu=1.5, sd=1)
            beta_1 = pm.Normal("beta_1", mu=-100, sd=50)
            sigma = pm.InverseGamma("sigma", alpha=6, beta=5)
            y = pm.Normal('y', mu=beta_0 + beta_1 * x_data, sd=sigma, observed=self.data["En0"])

            step = pm.NUTS(target_accept=target_accept)
            trace = pm.sample(draws=draws, tune=tune, step=step,
                              return_inferencedata=True,
                              progressbar=True)  # , nuts_kwargs=dict(target_accept=0.90))
            if show_result:
                labels = ["beta_0", "beta_1", "sigma"]
                pm.plot_trace(trace, labels)
                plt.show()
        return model, trace

    @property
    def summary(self):
        return az.summary(self.trace)

    def corner_plot(self):
        with self.model:
            names = ["beta_0", "beta_1", "sigma"]
            labels = [r"$\beta_0$", r"$\beta_1$", r"$\sigma$"]
            figure = mpb.figure.Figure(figsize=(1.05*17.88*cm, 2.5*8.6*cm))
            corner.corner(self.trace, var_names=names, labels=labels,  # truths={**physical_point, "error": sigma},
                          quantiles=(0.025, 0.5, 0.975),
                          title_quantiles=(0.025, 0.5, 0.975),
                          show_titles=True, title_fmt=".4f", title_kwargs={"fontsize": 8}, fig=figure)
            return figure

    def posterior_predictive(self, x_validate=None):
        if x_validate is None:
            return self.posterior_predictive_sampled

        with self.model:
            pm.set_data({"x_data": x_validate}, model=self.model)
            posterior_predict = pm.sample_posterior_predictive(self.trace)
            return posterior_predict

    def plot(self, ax=None, level=0.95, plot_scatter=True, x_validate=None):
        if ax is None:
            ax = plt.gca()

        if x_validate is None:
            x_validate = self.x_validate
            posterior_predict = self.posterior_predictive_sampled
        else:
            posterior_predict = self.posterior_predictive(x_validate)

        lower = np.quantile(posterior_predict["y"], q=0.5-level/2, axis=0)
        upper = np.quantile(posterior_predict["y"], q=0.5+level/2, axis=0)
        ax.fill_between(x_validate, lower, upper, alpha=0.3, label=f"Coester band ({level:.0f}\%)")
        if plot_scatter:
            ax.scatter(self.data["n0"], self.data["En0"])
        ax.plot(x_validate, np.median(posterior_predict["y"], axis=0))  #, label="Coester band")

        # ax.set_xlim(0.145, 0.175)
        # ax.set_ylim(-16.5, -14.7)
        # ax.set_xlabel('Saturation Density $n_0$ [fm$^{-3}$]')
        # ax.set_ylabel('Saturation Energy $E_0/A$ [MeV]')
        # ax.set_title("Empirical saturation box")
        # ax.legend(ncol=2, loc="upper center", prop={'size': 6})  #TODO: don't overwrite settings from DataSets
