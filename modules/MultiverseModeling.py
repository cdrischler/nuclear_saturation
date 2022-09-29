from tqdm import tqdm
from modules.plot_helpers import cm
import matplotlib.pyplot as plt
import pandas as pd
from modules.StatisticalModel import model_from_scenario
import corner
import numpy as np


def func(scenario, pdf, num_realizations=5, quantities=None, prior_params=None):
    samples = pd.DataFrame()
    for irealiz in tqdm(range(num_realizations), desc="Multi-universe sampling"):
        # set up canvas
        fig, axs = plt.subplots(1, 2, figsize=(2*6.8*cm, 6.8*cm), constrained_layout=True)

        axs[0].set_xlim(0.145, 0.175)
        axs[0].set_ylim(-16.5, -14.7)

        model = model_from_scenario(scenario, quantities=quantities, prior_params=prior_params)
        model.data.plot(ax=axs[0], x="rho0", y="E/A", linestyle="None", marker="o")
        # print(model.sanity_check(num_samples=100000, based_on="prior", do_print=True))

        # store data from current universe (for universe-averaging later on)
        tmp = model.sample_predictive_bf(return_predictive_only=False, based_on="posterior",
                                         num_samples=1, num_samples_mu_Sigma=100000)  # 100000
        tmp["universe"] = irealiz
        samples = pd.concat((samples, tmp))

        # plot predictive prior and predictive posterior (right panel)
        model.plot(ax=axs[1], num_samples=5000)  #500000
        pdf.savefig(fig)

    # plot multi-universe average of the posterior predictive
    names = ["predictive rho0", "predictive E/A"]
    labels = [r"$n_0$", r"$E_0/A$"]
    n_std = 2 # e.g., 86% will correspond to 2 sigma in 2 dimensions
    fig, axs = plt.subplots(2, 2, figsize=(2*6.8*cm, 2*6.8*cm))
    corner.corner(samples, var_names=names, labels=labels,
                  quantiles=(0.025, 0.5, 0.975),  # TODO: 2 sigma hard-coded
                  title_quantiles=(0.025, 0.5, 0.975),
                  levels=(1 - np.exp(-n_std**2 / 2),),
                  bins=200,
                  plot_datapoints=False, plot_density=True,
                  show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 8}, fig=fig)
    #drischler.plot(ax=axs[1, 0], plot_scatter=False, plot_box_estimate=True,
    #               add_legend=False, add_axis_labels=False)
    #eft_pred.plot(ax=axs[1, 0])

    axs[0, 0].set_xlim(0.145, 0.175)
    axs[1, 0].set_xlim(0.145, 0.175)
    axs[1, 0].set_ylim(-16.5, -14.7)
    axs[1, 1].set_xlim(-16.5, -14.7)

    pdf.savefig(fig)

#%%
