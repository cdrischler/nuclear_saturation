import sys; sys.path.append("./modules")
from abc import ABC, abstractmethod
from plot_helpers import plot_rectangle, confidence_ellipse_mean_cov, colors, colorset, flatui
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
import glob
import wget
import py7zr
import pandas as pd
import corner
from scipy.stats import multivariate_normal
from StatisticalModel import StatisticalModel
from dataclasses import dataclass


class DataSet(ABC):
    def __init__(self, set_specifier, filenames):
        self.set_specifier = set_specifier
        self.filenames = [filenames] if isinstance(filenames, str) else filenames
        self.data_frame = None

    def humanize_class_labels(self, clbl):  #TODO
        trans = dict(
            dutra_skyrme="Dutra et al. (\'12)",
            kortelainen="Kortelainen et al. (\'14)",
            brown="Brown (\'21)",
            dutra_rmf="Dutra et al. (\'14)"
        )
        default_value = clbl[0].upper()
        default_value = default_value.replace("_", " ")
        return trans.get(clbl, default_value).replace(" et al.", "+")  # " $\it{et~al.}$")

    def get_data_frame(self, exclude=None, exclude_in_col="label"):
        if exclude is None:
            return self.data_frame
        else:
            return self.data_frame.loc[~self.data_frame[exclude_in_col].isin(np.atleast_1d(exclude))]

    def get_statistical_model(self, exclude=None, num_points=1, num_pts_per_distr=1,
                              num_distr="all", replace=True, quantities=None, prior_params=None,
                              random_state=None):
        if isinstance(self, GenericDataSet):
            data = self.get_data_frame(exclude=exclude)
        else:
            data = self.sample(df=None, num_points=num_points, num_distr=num_distr, random_state=random_state,
                               num_pts_per_distr=num_pts_per_distr, replace=replace, exclude=exclude)
        return StatisticalModel(data=data, quantities=quantities, prior_params=prior_params)

    def sample_from_model(self, df=None, num_samples=1, kind="predictive_y", based_on="posterior", random_state=None, validate=True, **kwargs):
        model = self.get_statistical_model(**kwargs)
        ret = model.sample(num_samples=num_samples, kind=kind, based_on=based_on, random_state=random_state, validate=validate)
        ret = pd.DataFrame(data=ret, columns=["rho0", "E/A"])
        if df is not None:
            ret = pd.concat((df, ret))
        return ret

    @staticmethod
    def read_csv_files(filenames, comment="#", data_path="data", dtype=None,
                       search_pattern=r"satpoints_(\w*).csv"):
        dtype = {"label": str, "rho0": np.float64, "E/A": np.float64} if dtype is None else dtype
        data = pd.DataFrame()
        for file in filenames:
            data_read = pd.read_csv(f"{data_path}/{file}", comment=comment, dtype=dtype)
            data_read["class"] = re.search(search_pattern, file).group(1)
            # `class` could be changed later on, if re-classification is needed;
            # by default, all constraints in one file are considered members of one class
            data_read["file"] = file
            data = pd.concat([data, data_read], copy=False, ignore_index=True)
        return data

    @abstractmethod
    def sample(self, **kwargs):
        pass

    @abstractmethod
    def plot(self, **kwargs):
        pass

    @staticmethod
    def set_axes_ranges(ax):
        ax.set_xlim(0.146, 0.172)
        ax.set_ylim(-16.45, -15.45)

    @staticmethod
    def set_axes_labels(ax, set_title=True):
        ax.set_xlabel('Sat. Density $n_0$ [fm$^{-3}$]')
        ax.set_ylabel('Sat. Energy $E_0$ [MeV]')
        if set_title:
            ax.set_title("Nuclear Saturation: Empirical Constraints")

    @staticmethod
    def legend(ax, ncol=2, loc="upper left", out_of_frame=True, **kwargs):
        ax.legend(ncol=ncol, loc=loc,  # title="empirical constraints",
                  frameon=True, framealpha=1, edgecolor="0.8",
                  prop={'size': 6},
                  bbox_to_anchor=(-0.03, -0.15) if out_of_frame else None,  # bottom
                  # bbox_to_anchor=(1.03, .7),  # right
                  **kwargs)


class GenericDataSet(DataSet):
    def __init__(self, set_specifier, filenames):
        super().__init__(set_specifier=set_specifier, filenames=filenames)
        self.data_frame = super().read_csv_files(self.filenames)

    def __add__(self, other):
        return GenericDataSet(set_specifier=self.set_specifier + other.set_specifier,
                              filenames=self.filenames + other.filenames)

    def sample(self, df=None, num_points=1, replace=True, exclude=None, random_state=None):
        ret = self.get_data_frame(exclude=exclude)
        if isinstance(num_points, int):
            ret = ret.sample(num_points, replace=replace, random_state=random_state)
        elif num_points not in (None, "all"):
            raise ValueError(f"'num_points has to be int, 'None', or 'all', got '{num_points}'")
        if df is not None:
            ret = pd.concat((df, ret))
        return ret

    def plot(self, ax=None, plot_scatter=True, plot_box_estimate=False, marker_size=24,
             place_legend=True, add_axis_labels=True, exclude=None, zorder=-5, annotate=False,
             facecolor='lightgray', edgecolor='gray', legend_out_of_frame=True, **kwargs):
        if ax is None:
            ax = plt.gca()

        if plot_box_estimate:
            res = self.box_estimate(exclude=exclude)
            center_uncert = [[res["rho0"][i], res["E/A"][i]] for i in range(2)]
            plot_rectangle(center_uncert[0], center_uncert[1], ax=ax,
                           facecolor=facecolor, edgecolor=edgecolor,
                           alpha=0.4, zorder=zorder, **kwargs)

        if plot_scatter:
            from modules.plot_helpers import mpt_default_colors  # use matplotlib default colors
            dframe = self.get_data_frame(exclude=exclude)
            for iclbls, clbls in enumerate(dframe["class"].unique()):
                color = mpt_default_colors[iclbls] if annotate else None
                idxs = np.where(dframe["class"] == clbls)
                ax.scatter(dframe.iloc[idxs]["rho0"], dframe.iloc[idxs]["E/A"],
                           zorder=100, edgecolor="k", lw=0.5, color=color,
                           s=marker_size, label=self.humanize_class_labels(clbls))
                if annotate:
                    for index, row in (dframe.iloc[idxs].sort_values(by="E/A")).iterrows():
                        # ax.text(x=row["rho0"]*0+0.15, y=row["E/A"], s=row["label"], fontsize=6, color=color)
                        alpha1s = 0.6
                        bbox_dict = dict(boxstyle="round,pad=0.5", fc=color, alpha=alpha1s, ec="none", lw=1)
                        ax.annotate(row["label"].replace("-", "--"),
                            xy=(row["rho0"], row["E/A"]),
                            xytext=(0.15, -15.6-index*0.06), textcoords='data',
                            arrowprops={'arrowstyle':'-','color':color, 'alpha':alpha1s,'relpos':(1., 0.5),
                                        'shrinkA':5, 'shrinkB':5, 'patchA':None, 'patchB':None,
                                        'connectionstyle':"angle3,angleA=-11,angleB=100"},
                            horizontalalignment='left', verticalalignment='bottom',
                            rotation=0, size=5, zorder=1, bbox=bbox_dict)
                

        if add_axis_labels:
            super().set_axes_labels(ax, set_title=False)
        if place_legend:
            largs = dict(ax=ax, ncol=1, loc="lower right", out_of_frame=legend_out_of_frame) if annotate else dict(ax=ax)
            super().legend(**largs)
        super().set_axes_ranges(ax)

    def box_estimate(self, print_result=False, exclude=None):
        result = dict()
        for dim in ("rho0", "E/A"):
            df = self.get_data_frame(exclude=exclude)[dim]
            uncert = (df.max() - df.min())/2.
            central = df.min() + uncert
            result[dim] = (central, uncert)

        # print results if requested
        if print_result:
            print("Empirical saturation point from Drischler et al. (2016):")
            print(f"n0 = {result['rho0'][0]:.3f} +/- {result['rho0'][1]:.3f} fm^(-3)")
            print(f"E0/A = {result['E/A'][0]:.1f} +/- {result['E/A'][1]:.1f} MeV")

        return result


class NormDistDataSet(DataSet):
    def __init__(self, set_specifier):
        filenames, data = self.get_data(set_specifier)
        super().__init__(set_specifier=set_specifier, filenames=filenames)
        self.data_frame = data

    @staticmethod
    def get_data(set_specifier):
        data = []; labels = []
        if set_specifier == "fsu_rmf":
            files = sorted(glob.glob("data/Piekarewicz/*/CovEllipse.com"))
            for file in files:
                labels.append(re.search(r"(\w+)/Cov", file).group(1))
                data.append(open(file, 'r').readlines()[3].strip().split(","))
            data = pd.DataFrame(data, columns=("mean rho0", "mean E/A", "sigma rho0", "sigma E/A", "rho"),
                                dtype=np.float64)
            data["label"] = labels
        elif set_specifier == "reinhard":
            files = ["data/Reinhard/CovEllipse.dat"]*2
            data = pd.read_csv(files[0])
        else:
            raise ValueError(f"unknown `set_specifier` '{set_specifier}'")

        data["class"] = set_specifier
        data["file"] = files
        return files, data

    @staticmethod
    def from_row_to_mean_cov(row):
        mean = row[["mean rho0", "mean E/A"]].to_numpy()
        offdiag = row["sigma rho0"] * row["sigma E/A"] * row["rho"]
        cov = np.array([[row["sigma rho0"]**2, offdiag], [offdiag, row["sigma E/A"]**2]])
        return mean, cov

    def sample(self, df=None, num_points=1, replace=True, exclude=None, random_state=None):
        data = self.get_data_frame(exclude=exclude)
        data.reset_index()
        counts = data.sample(num_points, replace=replace, random_state=random_state).index.value_counts()
        # does not preserve order (we don't need to) but is much faster

        ret = []
        for irow, row in data.iterrows():
            mean, cov = NormDistDataSet.from_row_to_mean_cov(row)
            if irow not in counts.index:
                continue
            samples = multivariate_normal.rvs(mean=mean, cov=cov, size=counts.loc[irow],
                                              random_state=random_state)
            samples = np.atleast_2d(samples)
            result_row = pd.DataFrame(data=samples, columns=("rho0", "E/A"))
            for lbl in ("class", "label", "file"):
                result_row[lbl] = row[lbl]
            ret.append(result_row)
        ret = pd.concat(ret)

        if len(ret) != num_points:
            raise ValueError("Generate number of samples doesn't match requested number.")
        if df is not None:
            ret = pd.concat((df, ret))
        return ret

    def plot(self, ax=None, level=0.95, marker_size=8, add_legend=True,
             add_axis_labels=True, exclude=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        for irow, row in self.get_data_frame(exclude=exclude).iterrows():
            mean, cov = NormDistDataSet.from_row_to_mean_cov(row)
            n_std = np.sqrt(-np.log(1.-level)*2)
            if row["class"] == "reinhard":
                use_color_palette = flatui[-2::-1]
                facecolor = "none"
                edgecolor = use_color_palette[irow]
            else:
                use_color_palette = colorset
                facecolor = use_color_palette[irow]
                edgecolor = facecolor
            confidence_ellipse_mean_cov(mean=mean, cov=cov,
                                        ax=ax, n_std=n_std,
                                        facecolor=facecolor,
                                        edgecolor=edgecolor,
                                        label=f'{row["label"]} ({level*100:.0f}\%)')

        if add_axis_labels:
            super().set_axes_labels(ax)
        if add_legend:
            super().legend(ax)
        super().set_axes_ranges(ax)


class KernelDensityEstimate(DataSet):
    def __init__(self, set_specifier=None):
        filenames, data = self.get_data(set_specifier)
        super().__init__(set_specifier=set_specifier, filenames=filenames)
        self.data_frame = data

    @staticmethod
    def get_data(set_specifier):
        if set_specifier == "schunck":
            files = glob.glob("data/Schunck/samples?.csv")
            delimiter = ","
            column_names = ["rho0", "E/A"]
        elif set_specifier == "giuliani":
            output_folder = "data/giuliani"
            relative_filepath = "PlotSamples/100k_theta.dat"
            fullpath = f"{output_folder}/{relative_filepath}"
            if not os.path.exists(fullpath):
                print("trying to download Giuliani et al.'s data file from the internet (may take a while)")
                url = 'https://figshare.com/ndownloader/files/37611122'
                compressed_file = wget.download(url, out=".")
                with py7zr.SevenZipFile(compressed_file, 'r') as archive:
                    # allfiles = archive.getnames()
                    archive.extract(targets=relative_filepath, path=output_folder)
            files = [fullpath]
            delimiter = " "
            column_names = ["unknown0", "kf", "E/A"] + [f"unknown{i}" for i in range(1, 5+1)]
        elif set_specifier == "mcdonnell":
            files = ["data/1000ParamsSkyrme.csv"]
            delimiter = ","
            column_names = ["rho0", "E/A"]
        else:
            raise ValueError(f"unknown `set_specifier` '{set_specifier}'")

        def calc_density(kf, g=2):
            return 2*g*kf**3/(6*np.pi**2)
        data = []
        for file in sorted(files):
            data_read = pd.read_csv(file, comment="#", names=column_names, skiprows=0, delimiter=delimiter)
            data_read["file"] = file
            if set_specifier == "giuliani":
                data_read = pd.DataFrame(data={"rho0": calc_density(data_read["kf"]),
                                               "E/A": data_read["E/A"]})
                id = 1
            elif set_specifier == "schunck":
                id = int(re.search(r"samples(\d)", file).group(1))
            elif set_specifier == "mcdonnell":
                id = 1
            else:
                raise ValueError(f"unknown `set_specifier` '{set_specifier}'")

            data_read["class"] = set_specifier  # f"{set_specifier}:{id}"
            data_read["label"] = data_read["class"]
            data.append(data_read)
        data = pd.concat(data)
        return files, data

    def sample(self, df=None, num_points=1, replace=True, exclude=None, random_state=None):
        data = self.get_data_frame(exclude=exclude)
        class_lbl = pd.DataFrame(data["class"].unique(), columns=["class"])
        counts = class_lbl.sample(num_points, replace=replace, random_state=random_state).index.value_counts()
        # does not preserve order (we don't need to) but is much faster
        ret = []
        for irow, row in class_lbl.iterrows():
            if irow not in counts.index:
                continue
            dframe_filtered = data[data["class"] == row["class"]]
            samples = dframe_filtered.sample(counts.loc[irow], replace=replace, random_state=random_state)
            for lbl in ("class", "label"):
                samples[lbl] = row["class"]
            ret.append(samples)
        ret = pd.concat(ret)

        if len(ret) != num_points:
            raise ValueError("Generate number of samples doesn't match requested number.")
        if df is not None:
            ret = pd.concat((df, ret))
        return ret

    def plot(self, ax=None, level=0.95, fill=False, plot_scatter=False, marker_size=8,
             add_legend=True, add_axis_labels=True, exclude=None, use_seaborn=False, **kwargs):
        if ax is None:
            ax = plt.gca()

        if "additional_legend_handles" not in kwargs.keys():
            additional_legend_handles = []
        else:
            additional_legend_handles = kwargs["additional_legend_handles"]

        import matplotlib.patches as mpatches
        for icls, cls in enumerate(self.data_frame["class"].unique()):
            dframe_filtered = self.get_data_frame(exclude=exclude)
            mask = dframe_filtered["class"] == cls

            if "schunck" in cls.lower():
                use_colorset = sorted(colors[3:])
                label = "Schunck $\it{et~al.}$ (\'20" + f", {level*100:.0f}\%)"
                bins = 16
            elif "giuliani" in cls.lower():
                use_colorset = sorted(flatui)
                label = "Giuliani $\it{et~al.}$ (\'22" + f", {level*100:.0f}\%)"
                bins = 30
            elif "mcdonnell" in cls.lower():
                use_colorset = ["0.5"]
                label = "McDonnell $\it{et~al.}$ (\'15" + f", {level*100:.0f}\%)"
                bins = 10
            else:
                use_colorset = sorted(colorset)
                label = cls
                bins = 20

            color = use_colorset[-icls]
            
            if use_seaborn:
                levels = 1. - np.atleast_1d(level)
                if fill:
                    levels = np.append(levels, 1.)
                sns.kdeplot(ax=ax, x=dframe_filtered[mask]["rho0"].to_numpy(), 
                            y=dframe_filtered[mask]["E/A"].to_numpy(),
                            fill=fill, levels=levels,
                            # label=f"{cls} ({(1-levels[0])*100:.0f}\%)",
                            legend=False,
                            color=color
                            )
                # Note: sns.kdeplot() seems to have issues with displaying the handles in legends
            else:
                corner.hist2d(x=dframe_filtered[mask]["rho0"].to_numpy(),
                              y=dframe_filtered[mask]["E/A"].to_numpy(),
                              bins=bins, range=None, weights=None,
                              levels=np.atleast_1d(level), smooth=None, ax=ax, color=color,
                              quiet=False, plot_datapoints=plot_scatter, plot_density=False,
                              plot_contours=True, no_fill_contours=True, fill_contours=None,
                              contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
                              pcolor_kwargs=None, new_fig=False)
                
            patch = mpatches.Patch(edgecolor=color, facecolor="none",
                                    label=label.replace(" $\it{et~al.}$", "+"))
            additional_legend_handles.append(patch)

            #if plot_scatter:
            #    ax.scatter(x=dframe_filtered[mask]["rho0"], y=dframe_filtered[mask]["E/A"], label="scatter")

        handles, labels = ax.get_legend_handles_labels()
        for item in additional_legend_handles:
            handles.append(item)

        if add_axis_labels:
            super().set_axes_labels(ax)
        if add_legend:
            super().legend(ax, handles=handles)
        super().set_axes_ranges(ax)


@dataclass
class Scenario:
    label: str
    datasets: list

    @property
    def label_plain(self):
        return self.label.replace(" ", "-").lower()
#%%
