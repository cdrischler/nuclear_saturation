from tqdm import tqdm
from modules.plot_helpers import cm
import matplotlib.pyplot as plt
import pandas as pd
from modules.StatisticalModel import model_from_scenario
import corner
import numpy as np
import os
import arviz as az
from modules.DataSets import GenericDataSet, NormDistDataSet, KernelDensityEstimate
from modules.EftPredictions import *
from modules.StatisticalModel import StatisticalModel, multivariate_t, multivariate_normal, standard_prior_params
import matplotlib.backends.backend_pdf
from modules.DataSets import DataSetSampleConfig, Scenario
from modules.StatisticalModel import model_from_scenario


DEFAULT_DFT_CONSTRAINTS = {
    "Dutra_skyrme": GenericDataSet(filenames=["satpoints_dutra_skyrme.csv"]),
    "Kortelainen": GenericDataSet(filenames=["satpoints_kortelainen.csv"]),
    "Brown": GenericDataSet(filenames=["brown/satpoints_brown.csv"]),
    "Dutra_rmf": GenericDataSet(filenames=["satpoints_rmf_dutra_2014.csv"]),
    "FSU": NormDistDataSet(set_specifier="fsu_rmf"),
    "Schunck": KernelDensityEstimate(set_specifier="schunck"),
    "Giuliani": KernelDensityEstimate(set_specifier="giuliani")
}


class SaturationAnalysis:
    def __init__(self, pdf_output_path="./pdf"):
        self.pdf_output_path = pdf_output_path
        if not os.path.exists(pdf_output_path):
            os.mkdir(pdf_output_path)
        self.drischler_satbox = GenericDataSet(filenames=["satpoints_dutra_skyrme.csv", "satpoints_kortelainen.csv"])
        self.eft_predictions = EftPredictions(show_result=True)

    def plot_contraints(self, dft_constraints=None, eft=True, dft_conf_level=0.8647,
                        eft_conf_level=0.95, eft_plot_scatter=True):
        pdf = matplotlib.backends.backend_pdf.PdfPages(f"{self.pdf_output_path}/tt.pdf")
        fig, ax = plt.subplots(1, 1, figsize=(1.25*6.8*cm, 1.2*6.8*cm))
        self.drischler_satbox.plot(ax=ax, plot_scatter=False, plot_box_estimate=True, marker_size=8,
                                   add_legend=False, add_axis_labels=False, exclude=None)

        dft_constraints = DEFAULT_DFT_CONSTRAINTS if dft_constraints is None else dft_constraints
        for key, val in dft_constraints.items():
            val.plot(ax=ax, level=dft_conf_level)
            pdf.savefig(fig)
        if eft:
            self.eft_predictions.plot(ax=ax, level=eft_conf_level, plot_scatter=eft_plot_scatter)
            pdf.savefig(fig)
        pdf.close()
        # return fig, ax

    def plot_individual_models(self, num_points=3000, dft_constraints=None):
        dft_constraints = DEFAULT_DFT_CONSTRAINTS if dft_constraints is None else dft_constraints
        for lbl, val in tqdm(dft_constraints.items(), desc="Iterating over DFT constraints"):
            sample_kwargs = dict(exclude=None,
                                 num_points="all" if isinstance(val, GenericDataSet) else num_points,
                                 num_distr="all")

            scenario = Scenario(label=f"{lbl}-only",
                                configs=[DataSetSampleConfig(data_set=val, sample_kwargs=sample_kwargs)])
            self.multiverse(scenario=scenario, num_realizations=1, quantities=None, prior_params=None)

    def multiverse(self, scenario=None, num_realizations=5, quantities=None, prior_params=None, progressbar=True):
        pdf = matplotlib.backends.backend_pdf.PdfPages(f"{self.pdf_output_path}/{scenario.label}.pdf")
        samples = pd.DataFrame()
        for irealiz in tqdm(range(num_realizations), desc="Multiverse sampling", disable=progressbar):
            # set up canvas
            fig, ax = plt.subplots(1, 1, figsize=(6.8*cm, 6.8*cm), constrained_layout=True)
            self.drischler_satbox.plot(ax=ax, plot_scatter=False,
                                       plot_box_estimate=True, add_legend=False, add_axis_labels=False)

            model = model_from_scenario(scenario, quantities=quantities, prior_params=prior_params)
            model.data.plot(ax=ax, x="rho0", y="E/A", linestyle="None",
                            marker="o", c="k", ms=1, zorder=20, alpha=0.2)
            # print(model.sanity_check(num_samples=100000, based_on="prior", do_print=True))

            # store data from current universe (for universe-averaging later on)
            tmp = model.sample_predictive_bf(return_predictive_only=False, based_on="posterior",
                                             num_samples=1, num_samples_mu_Sigma=10000)  # 100000
            tmp["universe"] = irealiz
            samples = pd.concat((samples, tmp))

            # plot predictive prior and predictive posterior (right panel)
            model.plot(ax=ax, num_samples=50000, plot_data=False,
                       set_xy_limits=False, set_xy_lbls=False, place_legend=False)  #500000
            ax.set_xlim(0.135, 0.185)
            ax.set_ylim(-17.0, -14.7)
            pdf.savefig(fig)

        # plot multi-universe average of the posterior predictive
        names = ["predictive rho0", "predictive E/A"]
        labels = [r"$n_0$", r"$E_0/A$"]
        n_std = 2  # e.g., 86% will correspond to 2 sigma in 2 dimensions
        fig, axs = plt.subplots(2, 2, figsize=(2*6.8*cm, 2*6.8*cm))

        data = az.from_dict(posterior={lbl: samples[lbl] for lbl in names})
        corner.corner(data, # var_names=names,
                      labels=labels,
                      quantiles=(0.025, 0.5, 0.975),  # TODO: 2 sigma hard-coded
                      title_quantiles=(0.025, 0.5, 0.975),
                      levels=(1 - np.exp(-n_std**2 / 2),),
                      bins=200,
                      plot_datapoints=False, plot_density=False,
                      show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 8}, fig=fig)
        self.drischler_satbox.plot(ax=axs[1, 0], plot_scatter=False, plot_box_estimate=True,
                                   add_legend=False, add_axis_labels=False)
        self.eft_predictions.plot(ax=axs[1, 0])
        # self.plot_contraints()

        axs[0, 0].set_xlim(0.145, 0.175)
        axs[1, 0].set_xlim(0.145, 0.175)
        axs[1, 0].set_ylim(-16.5, -14.7)
        axs[1, 1].set_xlim(-16.5, -14.7)

        pdf.savefig(fig)
        pdf.close()
#%%
