from tqdm import tqdm
from modules.plot_helpers import cm
import matplotlib.pyplot as plt
import pandas as pd
import corner
import numpy as np
import os
import arviz as az
from modules.DataSets import GenericDataSet, NormDistDataSet, KernelDensityEstimate
from modules.EftPredictions import *
from modules.StatisticalModel import StatisticalModel, multivariate_t, multivariate_normal, standard_prior_params
import matplotlib.backends.backend_pdf
from modules.DataSets import Scenario
from modules.priors import label_filename


DEFAULT_DFT_CONSTRAINTS = {
    "Dutra_skyrme": GenericDataSet(filenames=["satpoints_dutra_skyrme.csv"]),
    "Kortelainen": GenericDataSet(filenames=["satpoints_kortelainen.csv"]),
    "Brown": GenericDataSet(filenames=["brown/satpoints_brown.csv"]),
    "Dutra_rmf": GenericDataSet(filenames=["satpoints_dutra_rmf.csv"]),
    "FSU": NormDistDataSet(set_specifier="fsu_rmf"),
    "SV-min": NormDistDataSet(set_specifier="sv-min"),
    "TOV": NormDistDataSet(set_specifier="tov"),
    "Schunck": KernelDensityEstimate(set_specifier="schunck"),
    "Giuliani": KernelDensityEstimate(set_specifier="giuliani")
}


drischler_satbox = GenericDataSet(filenames=["satpoints_dutra_skyrme.csv", "satpoints_kortelainen.csv"])


class SaturationAnalysis:
    def __init__(self, prestore_eft_fit=False, pdf_output_path="./pdf"):
        self.pdf_output_path = pdf_output_path
        if not os.path.exists(pdf_output_path):
            os.mkdir(pdf_output_path)
        self.drischler_satbox = drischler_satbox  # GenericDataSet(filenames=["satpoints_dutra_skyrme.csv", "satpoints_kortelainen.csv"])
        self.eft_predictions = EftPredictions(show_result=True) if prestore_eft_fit else None

    def plot_constraints(self, dft_constraints=None, eft=False, dft_conf_level=0.95,
                         eft_conf_level=0.95, eft_plot_scatter=True):
        pdf = matplotlib.backends.backend_pdf.PdfPages(f"{self.pdf_output_path}/constraints.pdf")
        fig, ax = plt.subplots(1, 1, figsize=(1.25*6.8*cm, 1.2*6.8*cm))
        self.drischler_satbox.plot(ax=ax, plot_scatter=False, plot_box_estimate=True, marker_size=8,
                                   place_legend=False, add_axis_labels=False, exclude=None)

        dft_constraints = DEFAULT_DFT_CONSTRAINTS if dft_constraints is None else dft_constraints
        additional_legend_handles = []
        for key, val in dft_constraints.items():
            handles = val.plot(ax=ax, level=dft_conf_level, additional_legend_handles=additional_legend_handles)
            if handles is not None:
                additional_legend_handles.append(handles)
            pdf.savefig(fig)
        if eft:
            if self.eft_predictions is None:
                self.eft_predictions = EftPredictions(show_result=True)
            self.eft_predictions.plot(ax=ax, level=eft_conf_level, plot_scatter=eft_plot_scatter)
            pdf.savefig(fig)
        pdf.close()
        # return fig, ax

    def plot_individual_models(self, num_points=3000, dft_constraints=None, prior_params=None):
        dft_constraints = DEFAULT_DFT_CONSTRAINTS if dft_constraints is None else dft_constraints
        for lbl, val in tqdm(dft_constraints.items(), desc="Iterating over DFT constraints"):
            sample_kwargs = dict(exclude=None,
                                 num_points="all" if isinstance(val, GenericDataSet) else num_points,
                                 num_distr="all")

            scenario = Scenario(label=f"{lbl}-only", datasets=[val])
            self.multiverse(scenario=scenario, num_realizations=1, plot_fitted_conf_regions=True,
                            quantities=None, prior_params=prior_params)

    def multiverse(self, scenario=None, num_realizations=10, num_pts_per_realization=1, sample_replace=True,
                   levels=None, quantities=None, parallel_eval=True,
                   prior_params=None, progressbar=True, debug=False, bins=100,
                   plot_fitted_conf_regions=True, plot_iter_results=False, close_figures=True,
                   num_samples=1, num_samples_mu_Sigma=10000000, **kwargs):
        if levels is None:
            levels = np.array((0.5, 0.8, 0.95, 0.99))
        levels = np.atleast_1d(levels)

        file_output = f"{self.pdf_output_path}/{scenario.label_plain}_{label_filename(prior_params['label'])}_"
        file_output += f"{num_samples_mu_Sigma}_{num_realizations}.pdf"
        pdf = matplotlib.backends.backend_pdf.PdfPages(file_output)

        num_points = num_realizations*num_pts_per_realization
        use_kwargs = dict(df=None, replace=sample_replace,
                          exclude=None, num_points=num_points,
                          num_pts_per_distr=1, num_distr="all")
        sampled_dft_constraints = [dset.sample(**use_kwargs) for dset in scenario.datasets]

        mc_iter = []
        for j in range(num_realizations):
            start = j*num_pts_per_realization
            end = start + num_pts_per_realization
            out = [sampled_dft_constraints[i][start:end] for i in range(len(sampled_dft_constraints))]
            mc_iter.append(pd.concat(out))

        # return mc_iter
        import time
        if parallel_eval:
            from multiprocessing import Pool, cpu_count
            from functools import partial
            num_workers = int(cpu_count()/2)
            print("Number of workers : ", num_workers)
            ct = time.perf_counter()
            with Pool(processes=num_workers) as pool:
                out = pool.map(partial(test, quantities=quantities, prior_params=prior_params,
                               num_samples=num_samples, num_samples_mu_Sigma=num_samples_mu_Sigma),
                         mc_iter) #, chunksize=1000)

            samples = pd.concat(out)
            print(f"+++++{time.perf_counter()-ct}")
            ct = time.perf_counter()
        #
        #
        else:
            samples = []
            for irealiz in tqdm(range(num_realizations), desc="MC sampling", disable=not progressbar):
                # set up canvas (and draw saturation box)
                ct = time.perf_counter()
                # print(mc_iter[irealiz])
                # model = model_from_scenario(scenario, quantities=quantities, prior_params=prior_params)
                model = StatisticalModel(data=mc_iter[irealiz],
                                         quantities=quantities, prior_params=prior_params)

                # print(f"+++++{time.perf_counter()-ct}")
                ct = time.perf_counter()

                # store data from current universe (for universe-averaging later on)
                tmp = model.sample_predictive_bf(return_predictive_only=False, based_on="posterior",
                                                 num_samples=num_samples,
                                                 num_samples_mu_Sigma=num_samples_mu_Sigma)  # 100000

                # print(f"+++++{time.perf_counter()-ct}")
                ct = time.perf_counter()

                tmp["universe"] = irealiz
                # print(tmp)
                samples.append(tmp)

                # print(f"+++++{time.perf_counter()-ct}")
                ct = time.perf_counter()

                # plot predictive prior and predictive posterior (right panel)
                if plot_iter_results:
                    fig, axs = model.plot_predictives(plot_data=True, levels=levels, num_pts=10000000,
                                                     set_xy_limits=True, set_xy_lbls=True, place_legend=True,
                                                     validate=False)
                    pdf.savefig(fig)
                    if close_figures:
                        plt.close(fig=fig)

                    figsAxs = model.plot_predictives_corner(plot_data=True, levels=levels, show_box_in_marginals=False,
                                                            place_legend=True, validate=False)
                    for figAx in figsAxs:
                        pdf.savefig(figAx[0])
                        if close_figures:
                            plt.close(fig=figAx[0])
            samples = pd.concat(samples)

        # plot multi-universe average of the posterior predictive (corner plot)
        use_level = 0.95
        names = ["predictive rho0", "predictive E/A"]
        labels = ['Sat. Density $n_0$ [fm$^{-3}$]', 'Sat. Energy $E_0/A$ [MeV]']
        fig, axs = plt.subplots(2, 2, figsize=(9*cm, 1.2*8.6*cm))

        data = az.from_dict(posterior={lbl: samples[lbl] for lbl in names})
        corner.corner(data,  # var_names=names,
                      labels=labels,
                      quantiles=(0.5-use_level/2, 0.5+use_level/2),
                      verbose=debug,
                      # title_quantiles=None,  # bug in current version of `corner`
                      levels=levels,
                      facecolor="none",
                      bins=bins,
                      # color='r',
                      labelpad=-0.0,
                      plot_datapoints=False, plot_density=False,
                      no_fill_contours=True, fill_contours=None,
                      title_fmt=".3f", title_kwargs={"fontsize": 8}, fig=fig)

        axs[0, 1].text(0.05, 0.9, f"posterior predictive", transform=axs[0, 1].transAxes)  #transAxes)
        axs[0, 1].text(0.05, 0.82, f"({prior_params['label']})", transform=axs[0, 1].transAxes)  # transAxes)

        # fix bug in `corner`, see https://github.com/dfm/corner.py/issues/107
        title_template = r"${{{1}}} \pm {{{2}}}$ {{{3}}} ({{{4:.0f}}}\%)"
        for iname, name in enumerate(names):
            mu = np.mean(samples[name])
            disp = mu - np.percentile(samples[name], (1-use_level)/2*100)  # t distribution is symmetric about the mean
            if debug:
                print("expected means and two-sided errors", mu, disp, mu-disp, mu+disp)

            if iname == 0:
                quantity = "n_0"
                unit = "fm$^{-3}$"
                title_fmt = ".3f"
            else:
                quantity = r"\frac{E_0}{A}"
                unit = "MeV"
                title_fmt = ".2f"

            fmt = "{{0:{0}}}".format(title_fmt).format
            title = title_template.format(quantity, fmt(mu), fmt(disp), unit, use_level*100)
            axs[iname, iname].set_title(title, fontsize=7)

        self.drischler_satbox.plot(ax=axs[1, 0], plot_scatter=False, plot_box_estimate=True,
                                   place_legend=False, add_axis_labels=False)  # , zorder=60, facecolor="none")
        # self.eft_predictions.plot(ax=axs[1, 0])
        # from plot_helpers import colors
        # axs[1, 0].scatter(samples[names[0]], samples[names[1]], s=4, c=colors[4])  # avoid: a LOT of samples!

        for row in axs[:, 0]:
            row.set_xlim(0.145, 0.175)

        axs[1, 0].set_ylim(-16.5, -14.7)  # not that the axes are different
        axs[1, 1].set_xlim(-16.5, -14.7)

        # fit bivariate t distribution to samples; only accurate for large numbers of sampling points
        from plot_helpers import fit_bivariate_t
        fit = fit_bivariate_t(samples[names].to_numpy(), alpha_fit=0.68, nu_limits=(3, 60),
                                  tol=1e-3, print_status=debug)
        if debug:
            print("fit results", fit)
            cov_est = np.cov(samples[names[0]], samples[names[1]])
            print("est cov matrix:", cov_est)

            # expected values
            if num_realizations == 1:
                print("expected values:")
                df, mu, shape_matrix = model.predictives_params("posterior")
                print("exp df:", df)
                print("exp mean:", mu)
                print("exp cov matrix:", shape_matrix * df / (df-2))

        if plot_fitted_conf_regions:
            from plot_helpers import plot_confregion_bivariate_t
            plot_confregion_bivariate_t(fit["mu"], fit["Psi"], fit["df"],
                                        ax=axs[1, 0], alpha=levels, alpha_unit="decimal", num_pts=10000000,
                                        plot_scatter=False, validate=False, zorder=100)  # linestyle=":"
            axs[1, 0].legend(ncol=2, title="confidence level (fit)", prop={'size': 6}, frameon=False)

        pdf.savefig(fig)
        pdf.close()
        if close_figures:
            plt.close(fig=fig)
        print(f"Results written to '{file_output}'")
        return fit


def visualize_priors(prior_params_list, levels=None, plot_satbox=True):
    prior_params_list = np.atleast_1d(prior_params_list)
    num_priors = len(prior_params_list)

    fig, axs = plt.subplots(1, num_priors, figsize=(7, 6.4*cm), tight_layout=True)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    for iprior_params, prior_params in enumerate(prior_params_list):
        ax = axs[iprior_params] if num_priors > 1 else axs

        if plot_satbox:
            drischler_satbox.plot(ax=ax, plot_scatter=False, plot_box_estimate=True, marker_size=8,
                                  place_legend=False, add_axis_labels=False, exclude=None)

        triv_model = StatisticalModel(data=None, prior_params=prior_params)
        df, mu, shape_matrix = triv_model.predictives_params("prior")
        from modules.plot_helpers import plot_confregion_bivariate_t
        plot_confregion_bivariate_t(ax=ax, mu=mu,
                                    Sigma=shape_matrix, nu=df,
                                    alpha=levels, alpha_unit="decimal", num_pts=10000000,
                                    plot_scatter=False, validate=False)

        ax.set_title(f"Prior {prior_params['label']}")

        ax.set_xlim(0.14, 0.18)
        ax.set_ylim(-18., -14.)

        ax.set_xlabel('Sat. Density $n_0$ [fm$^{-3}$]')

        if iprior_params == 0:
            ax.set_ylabel('Sat. Energy $E_0/A$ [MeV]')
        else:
            ax.set_yticklabels([])

        if iprior_params == 2:
            ax.legend(ncol=2, prop={'size': 7}, frameon=False)
    return fig, axs


def test(data, quantities, prior_params, num_samples, num_samples_mu_Sigma):
        model = StatisticalModel(data=data,
                                 quantities=quantities, prior_params=prior_params)
        tmp = model.sample_predictive_bf(return_predictive_only=False, based_on="posterior",
                                         num_samples=num_samples,
                                         num_samples_mu_Sigma=num_samples_mu_Sigma)  # 100000
        # tmp["universe"] = irealiz
        return tmp

#%%
