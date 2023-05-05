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
import time
from multiprocessing import Pool, cpu_count, current_process
from functools import partial


__DEFAULT_DFT_CONSTRAINTS = [
    GenericDataSet(set_specifier="dutra_skyrme", filenames=["satpoints_dutra_skyrme.csv"]),
    GenericDataSet(set_specifier="kortelainen", filenames=["satpoints_kortelainen.csv"]),
    GenericDataSet(set_specifier="brown", filenames=["brown/satpoints_brown.csv"]),
    GenericDataSet(set_specifier="dutra_rmf", filenames=["satpoints_dutra_rmf.csv"]),
    NormDistDataSet(set_specifier="fsu_rmf"),
    NormDistDataSet(set_specifier="reinhard"),
    KernelDensityEstimate(set_specifier="mcdonnell"),
    KernelDensityEstimate(set_specifier="schunck"),
    KernelDensityEstimate(set_specifier="giuliani")
]
DEFAULT_DFT_CONSTRAINTS = {elem.set_specifier: elem for elem in __DEFAULT_DFT_CONSTRAINTS}

drischler_satbox = GenericDataSet(
    set_specifier="drischler_satbox",
    filenames=["satpoints_dutra_skyrme.csv", "satpoints_kortelainen.csv"]
)
# drischler_satbox = DEFAULT_DFT_CONSTRAINTS["dutra_skyrme"] + DEFAULT_DFT_CONSTRAINTS["kortelainen"]


class SaturationAnalysis:
    def __init__(self, prestore_eft_fit=False, pdf_output_path="./pdf", samples_output_path="./samples"):
        self.pdf_output_path = pdf_output_path
        self.samples_output_path = samples_output_path
        for path in (pdf_output_path, samples_output_path):
            if not os.path.exists(path):
                os.mkdir(path)
        self.drischler_satbox = drischler_satbox
        self.eft_predictions = EftPredictions(show_result=True) if prestore_eft_fit else None

    def plot_constraints(self, dft_constraints=None, eft=False, dft_conf_level=0.95,
                         eft_conf_level=0.95, eft_plot_scatter=True):
        pdf = matplotlib.backends.backend_pdf.PdfPages(f"{self.pdf_output_path}/constraints.pdf")
        fig, ax = plt.subplots(1, 1, figsize=(1.25*6.8*cm, 1.3*6.8*cm))
        self.drischler_satbox.plot(ax=ax, plot_scatter=False, plot_box_estimate=True, marker_size=8,
                                   place_legend=False, add_axis_labels=False, exclude=None)

        dft_constraints = DEFAULT_DFT_CONSTRAINTS if dft_constraints is None else dft_constraints
        additional_legend_handles = []
        ax.text(0.72, 0.40, 'Skyrme', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.text(0.15, 0.35, 'RMF', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
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
            scenario = Scenario(label=f"{lbl}-only", datasets=[val])
            self.mc_iterate(scenario=scenario, num_realizations=num_points, plot_fitted_conf_regions=True,
                            quantities=None, prior_params=prior_params)

    @staticmethod
    def __sample_dft_realizations(datasets, num_realizations, num_pts_per_dft_model, sample_replace,
                                  random_state=None):
        num_points = num_realizations*num_pts_per_dft_model
        use_kwargs = dict(df=None, replace=sample_replace,
                          random_state=random_state,
                          exclude=None, num_points=num_points)
        sampled_dft_constraints = [dset.sample(**use_kwargs) for dset in datasets]
        mc_iter = []
        for j in range(num_realizations):
            start = j*num_pts_per_dft_model
            end = start + num_pts_per_dft_model
            out = [sampled_dft_constraints[i][start:end] for i in range(len(sampled_dft_constraints))]
            mc_iter.append(pd.concat(out))
        return mc_iter

    @staticmethod
    def sample_mix_models_batch(batch_size, num_pts_per_dft_model, sample_replace, scenario,
                                quantities, prior_params, num_samples_mu_Sigma, file_prefix):
        # step 0: init random number generator for parallel computing (pass job id along with batch_size)
        # to ensure the results are reproducible (and not completely random)
        global rng_global, worker_id, root_seed
        if isinstance(batch_size, tuple):
            ibatch, batch_size = batch_size
            rng = np.random.default_rng([ibatch, root_seed])
            # https://numpy.org/devdocs/reference/random/parallel.html#sequence-of-integer-seeds
        else:
            # if no job id is passed, then use the random number generator of the work,
            # which will not lead to reproducible results
            rng = rng_global

        # step 1: pre-store all (random) DFT realizations
        # print(f"\tGenerating {batch_size} DFT realizations", flush=True)
        ct = time.perf_counter()
        dft_realizations = SaturationAnalysis.__sample_dft_realizations(
            num_realizations=batch_size,
            num_pts_per_dft_model=num_pts_per_dft_model,
            sample_replace=sample_replace,
            datasets=scenario.datasets,
            random_state=rng
        )
        print(f"Required time for generating all DFT realizations [{worker_id}]: {time.perf_counter()-ct:.6f} s", flush=True)

        # step 2: construct models for these realizations and sample from their posterior distributions
        ct = time.perf_counter()
        iter_func = partial(SaturationAnalysis.sample_from_model, quantities=quantities, prior_params=prior_params,
                            num_samples_mu_Sigma=num_samples_mu_Sigma, random_state=rng, file_prefix=file_prefix)
        out = map(iter_func, enumerate(dft_realizations))
        samples = pd.concat(out)
        print(f"Required time for sampling {batch_size} mixture models [{worker_id}]: {time.perf_counter()-ct:.6f} s", flush=True)
        return samples

    @staticmethod
    def sample_from_model(data, quantities, prior_params, num_samples_mu_Sigma, random_state, file_prefix):
        i_iter, data = data
        model = StatisticalModel(data=data, quantities=quantities, prior_params=prior_params)
        # diagnostics: plot predictive posterior of the trained model
        if file_prefix is not None:
            levels = np.array((0.5, 0.8, 0.95, 0.99))
            # fig, axs = model.plot_predictives(plot_data=True, levels=levels, num_pts=10000000,
            #                                    set_xy_limits=True, set_xy_lbls=True, place_legend=True,
            #                                    validate=False)
            # fig.savefig(f"{file_prefix}_predictives_{worker_id}_{i_iter}.pdf")
            # plt.close(fig)
            figsAxs = model.plot_predictives_corner(plot_data=True, levels=levels, show_box_in_marginals=False,
                                                    place_legend=True, validate=False)
            figsAxs[1][0].savefig(f"{file_prefix}_posterior_predictives_{worker_id}_{i_iter}.pdf")
            for ifigAx, figAx in enumerate(figsAxs):
                plt.close(figAx[0])

        # option 1: sample (mu, Sigma, y') brute-force in a multi-step process
        # return model.sample_predictive_bf(return_predictive_only=False, based_on="posterior",
        #                                   num_samples_mu_Sigma=num_samples_mu_Sigma, random_state=random_state)

        # option 2: sample just y' from the analytically given posterior predictive (faster)
        tmp = model.sample(num_samples=num_samples_mu_Sigma, kind="predictive_y", based_on="posterior",
                           random_state=random_state, validate=False)
        tmp = np.atleast_2d(tmp)
        return pd.DataFrame(data={"predictive rho0": tmp[:, 0], "predictive E/A": tmp[:, 1]})

    def mc_iterate(self, scenario=None, num_realizations=1000000, num_pts_per_dft_model=1, sample_replace=True,
                   levels=None, quantities=None, prior_params=None, bins=120, debug=True, pdf=None,
                   plot_fitted_conf_regions=True, plot_iter_results=False, store_samples=True,
                   req_num_workers=10, num_batch_per_worker=1, num_samples_mu_Sigma=10):
        ct = time.perf_counter()
        # step 1: determine batch sizes
        max_num_workers = cpu_count()//2
        num_workers = np.min((req_num_workers, max_num_workers))
        print(f"Number of workers used for mixture model sampling: {num_workers} (max: {max_num_workers})")
        batch_sizes = [len(elem) for elem in np.array_split(range(num_realizations),
                                                            num_batch_per_worker*num_workers) if len(elem) > 0]
        print(f"Sampling using {len(batch_sizes)} batches with sizes:", batch_sizes)

        # step 2: create samples
        if plot_iter_results and num_realizations <= 20:
            file_prefix = f"{self.pdf_output_path}/diagnostics_"
            file_prefix += f"{scenario.label_plain}_{label_filename(prior_params['label'])}_"
            file_prefix += f"num_postersamples{num_samples_mu_Sigma}_num_mciter_{num_realizations}"
        else:
            file_prefix = None
        iter_func = partial(SaturationAnalysis.sample_mix_models_batch,
                            num_pts_per_dft_model=num_pts_per_dft_model,
                            sample_replace=sample_replace,
                            scenario=scenario,
                            quantities=quantities,
                            prior_params=prior_params,
                            num_samples_mu_Sigma=num_samples_mu_Sigma,
                            file_prefix=file_prefix
                            )

        with Pool(processes=num_workers, initializer=worker_init) as pool:
            out = pool.map(iter_func, enumerate(batch_sizes, start=1))  # , chunksize=1)
        samples = pd.concat(out)
        print(f"Required time for generating all {len(samples)} posterior samples: {time.perf_counter()-ct:.6f} s", flush=True)

        # step 3: plot samples
        if levels is None:
            levels = np.array((0.5, 0.8, 0.95, 0.99))
        levels = np.atleast_1d(levels)
        file_output = f"{self.pdf_output_path}/mc_output_{req_num_workers}_{scenario.label_plain}_"
        file_output += f"{label_filename(prior_params['label'])}_num_postersamples_{num_samples_mu_Sigma}"
        file_output += f"_num_mciter_{num_realizations}.pdf"
        fit = self.plot_samples(samples=samples, levels=levels, bins=bins, store_samples=store_samples,
                                prior_params=prior_params, plot_fitted_conf_regions=plot_fitted_conf_regions,
                                add_info=None, debug=debug, pdf=pdf, file_output=file_output)
        return fit

    def plot_samples(self, samples, debug, levels, bins, prior_params, plot_fitted_conf_regions, 
                     file_output, store_samples=True, add_info=None, pdf=None):
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
        if add_info is not None:
            axs[0, 1].text(0.05, 0.74, f"({add_info})", transform=axs[0, 1].transAxes)  # transAxes)

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
                quantity = r"E_0"
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

        axs[1, 0].set_ylim(-16.5, -14.7)  # note that the axes are different
        axs[1, 1].set_xlim(-16.5, -14.7)

        # fit bivariate t distribution to samples; only accurate for large numbers of sampling points
        from plot_helpers import fit_bivariate_t
        fit = fit_bivariate_t(samples[names].to_numpy(), alpha_fit=0.68, nu_limits=(3, 60),
                                  tol=1e-3, print_status=debug)

        if plot_fitted_conf_regions:
            from plot_helpers import plot_confregion_bivariate_t
            plot_confregion_bivariate_t(fit["mu"], fit["Psi"], fit["nu"],
                                        ax=axs[1, 0], alpha=levels, alpha_unit="decimal", num_pts=10000000,
                                        plot_scatter=False, validate=False, zorder=100)  # linestyle=":"
            axs[1, 0].legend(ncol=2, title="confidence level (fit)", prop={'size': 6}, frameon=False)

        pdf_not_provided = type(pdf) != matplotlib.backends.backend_pdf.PdfPages
        if pdf_not_provided:
            pdf = matplotlib.backends.backend_pdf.PdfPages(file_output)
            print(f"Writing results to '{file_output}'")
        pdf.savefig(fig)
        plt.close(fig=fig)
        print(fit)
        if pdf_not_provided:
            pdf.close()

        # store samples if requested
        if store_samples:
            if pdf_not_provided:
                filename = file_output
            else:
                filename = pdf._file.fh.name
            filename = filename.replace("mc_output", "samples").replace(".pdf", ".h5").replace("pdf/", f"{self.samples_output_path}/")
            samples.to_hdf(filename, key='samples', mode='w', complevel=6)
            print(f"Samples written to '{filename}'.")
        return fit

    @staticmethod
    def _get_mc_output_filename(kwargs):
        filename = f"pdf/mc_{kwargs['scenario'].label}_num_postersamples_{kwargs['num_samples_mu_Sigma']}_"
        filename += f"num_mciter_{kwargs['num_realizations']}_numworkers_{kwargs['req_num_workers']}.pdf"
        return filename

    def mc_run_scenario(self, scenario, used_prior_sets, results=None, **input_kwargs):
        dflt_kwargs = dict(
            scenario=scenario,
            num_realizations=1,
            num_pts_per_dft_model=1,
            num_samples_mu_Sigma=100000,
            req_num_workers=10,
            sample_replace=True,
            plot_iter_results=False,
            debug=False,
            plot_fitted_conf_regions=True,
            store_samples=True
        )
        kwargs = {**dflt_kwargs, **input_kwargs}
        print("Using the following configuration:")
        for key, val in kwargs.items():
            print(f"{key}: {val}")
        filename=self._get_mc_output_filename(kwargs)
        # print(f"writing to '{filename}'")
        pdf = matplotlib.backends.backend_pdf.PdfPages(filename=filename)
        from StatisticalModel import latex_it
        for prior_set in used_prior_sets:
            fit = self.mc_iterate(pdf=pdf, prior_params=prior_set, **kwargs)
            fit["config"] = kwargs
            if isinstance(results, dict): 
                results[prior_set["label"]][scenario.label] = fit
            latex_it(fit, title=f"fit predictive params: {prior_set['label']}")
        pdf.close()


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
            ax.set_ylabel('Sat. Energy $E_0$ [MeV]')
        else:
            ax.set_yticklabels([])

        if iprior_params == 2:
            ax.legend(ncol=2, prop={'size': 7}, frameon=False)
    return fig, axs


def worker_init(status_msg=True):
    global worker_id, rng_global, root_seed
    worker_id = os.getpid()
    name = current_process().name
    id = current_process()._identity
    if status_msg:
        print(f"Worker with pid {worker_id} [{name} {id}] initialized.", flush=True)
    root_seed = 0x8c3c010cb4754c905776bdac5ee7501
    rng_global = np.random.default_rng([worker_id, root_seed])
    # https://numpy.org/devdocs/reference/random/parallel.html#sequence-of-integer-seeds

#%%
