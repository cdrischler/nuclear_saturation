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
    def __init__(self, filenames):
        self.filenames = [filenames] if isinstance(filenames, str) else filenames
        self.data_frame = None

    def get_data_frame(self, exclude=None, exclude_in_col="label"):
        if exclude is None:
            return self.data_frame
        else:
            return self.data_frame.loc[~self.data_frame[exclude_in_col].isin(np.atleast_1d(exclude))]

    def get_statistical_model(self, exclude=None, num_points=1, num_pts_per_distr=1,
                              num_distr="all", replace=True, quantities=None, prior_params=None,
                              random_state=None, **kwargs):
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
    def set_axes_labels(ax):
        ax.set_xlabel('Sat. Density $n_0$ [fm$^{-3}$]')
        ax.set_ylabel('Sat. Energy $E_0/A$ [MeV]')
        ax.set_title("Nuclear Saturation: Empirical Constraints")

    @staticmethod
    def legend(ax, **kwargs):
        ax.legend(ncol=2, loc="upper left",  # title="empirical constraints",
                  frameon=False, prop={'size': 6}, **kwargs)


class GenericDataSet(DataSet):
    def __init__(self, filenames):
        super().__init__(filenames=filenames)
        self.data_frame = super().read_csv_files(self.filenames)

    def __add__(self, other):
        return GenericDataSet(filenames=self.filenames + other.filenames)

    def sample(self, df=None, num_points=1, replace=True, exclude=None, random_state=None, **kwargs):
        ret = self.get_data_frame(exclude=exclude)
        if isinstance(num_points, int):
            ret = self.get_data_frame(exclude=exclude).sample(num_points, replace=replace, random_state=random_state)
        elif num_points not in (None, "all"):
            raise ValueError(f"'num_points has to be int, 'None', or 'all', got '{num_points}'")
        if df is not None:
            ret = pd.concat((df, ret))
        return ret

    def plot(self, ax=None, plot_scatter=True, plot_box_estimate=False, marker_size=8,
             place_legend=True, add_axis_labels=True, exclude=None, zorder=-5,
             facecolor='lightgray', edgecolor='gray', **kwargs):
        if ax is None:
            ax = plt.gca()

        if plot_box_estimate:
            res = self.box_estimate(exclude=exclude)
            center_uncert = [[res["rho0"][i], res["E/A"][i]] for i in range(2)]
            plot_rectangle(center_uncert[0], center_uncert[1], ax=ax,
                           facecolor=facecolor, edgecolor=edgecolor,
                           alpha=0.4, zorder=zorder, **kwargs)

        if plot_scatter:
            dframe = self.get_data_frame(exclude=exclude)
            for clbls in dframe["class"].unique():
                idxs = np.where(dframe["class"] == clbls)
                ax.scatter(dframe.iloc[idxs]["rho0"], dframe.iloc[idxs]["E/A"],
                           zorder=100, edgecolor="k", lw=0.5,  #color="k",
                           s=marker_size, label=self.humanize_class_labels(clbls))
        if add_axis_labels:
            super().set_axes_labels(ax)
        if place_legend:
            super().legend(ax)
        super().set_axes_ranges(ax)

    def humanize_class_labels(self, clbl):
        trans = dict(
            dutra_skyrme="Dutra et al. (\'12)",
            kortelainen="Kortelainen et al. (\'14)",
            brown="Brown (\'21)",
            dutra_rmf="Dutra et al. (\'14)"
        )
        default_value = clbl[0].upper()
        default_value = default_value.replace("_", " ")
        return trans.get(clbl, default_value).replace(" et al.", "+")  # " $\it{et~al.}$")

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
    def __init__(self, filenames=None, set_specifier=None):
        filenames, data = NormDistDataSet.get_data(filenames, set_specifier)
        super().__init__(filenames=filenames)
        self.data_frame = data

    @staticmethod
    def get_data(filenames, set_specifier):
        use_filenames = bool(filenames)
        use_specifier = bool(set_specifier)
        if use_filenames == use_specifier:
            raise ValueError("Need to specify either `filenames` or `set_specifier`")

        data = []; labels = []
        if use_filenames:
            files = filenames
        else:
            if set_specifier == "fsu_rmf":
                files = sorted(glob.glob("data/Piekarewicz/*/CovEllipse.com"))
            elif set_specifier in ("sv-min", "tov"):
                files = ["data/Reinhard/CovEllipse.dat"]
            else:
                raise ValueError(f"unknown `set_specifier` '{set_specifier}'")

        if set_specifier == "fsu_rmf":
            for file in files:
                labels.append(re.search(r"(\w+)/Cov", file).group(1))
                data.append(open(file, 'r').readlines()[3].strip().split(","))
            data = pd.DataFrame(data, columns=("mean rho0", "mean E/A", "sigma rho0", "sigma E/A", "rho"), dtype=np.float64)
            data["label"] = labels
        else:
            data = pd.read_csv(files[0])
            label = {"sv-min": "SV-min", "tov": "TOV"}[set_specifier]
            data = data.loc[data['label'] == label]
            data["label"] = label

        data["class"] = set_specifier
        data["file"] = files
        return files, data

    @staticmethod
    def from_row_to_mean_cov(row):
        mean = row[["mean rho0", "mean E/A"]].to_numpy()
        offdiag = row["sigma rho0"] * row["sigma E/A"] * row["rho"]
        cov = np.array([[row["sigma rho0"]**2, offdiag], [offdiag, row["sigma E/A"]**2]])
        return mean, cov

    def sample(self, df=None, num_points=1, num_distr="all", num_pts_per_distr=None, replace=True, exclude=None,
               random_state=None, **kwargs):
        data = self.get_data_frame(exclude=exclude)
        if isinstance(num_distr, int):
            data = data.sample(num_distr, replace=replace, random_state=random_state)
        elif num_distr != "all":
            raise ValueError(f"'num_distr' should be int or 'all', got '{num_distr}'.")
        num_distr = len(data)

        if num_pts_per_distr is None:
            num_pts_per_distr = int(np.max([1, num_points/num_distr]))

        ret = pd.DataFrame()
        for irow, row in data.iterrows():
            mean, cov = NormDistDataSet.from_row_to_mean_cov(row)
            samples = multivariate_normal.rvs(mean=mean, cov=cov, size=num_pts_per_distr)
            samples = np.atleast_2d(samples)
            result_row = pd.DataFrame(data=samples, columns=("rho0", "E/A"))
            for lbl in ("class", "label", "file"):
                result_row[lbl] = row[lbl]
            ret = pd.concat((ret, result_row))

        if num_points is not None:
            ret = ret.sample(n=num_points, replace=len(ret) < num_points, random_state=random_state)

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
            if row["class"] in ("sv-min", "tov"):
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
    def __init__(self, filenames=None, set_specifier=None):
        filenames, data = KernelDensityEstimate.get_data(filenames, set_specifier)
        super().__init__(filenames=filenames)
        self.data_frame = data
        self.set_specifier = set_specifier

    @staticmethod
    def get_data(filenames, set_specifier):
        use_filenames = bool(filenames)
        use_specifier = bool(set_specifier)
        if use_filenames == use_specifier:
            raise ValueError("Need to specify either `filenames` or `set_specifier`")

        if use_filenames:
            files = filenames
        else:
            if set_specifier == "schunck":
                files = glob.glob("data/Schunck/samples?.csv")
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
            else:
                raise ValueError(f"unknown `set_specifier` '{set_specifier}'")

        data = pd.DataFrame()
        if set_specifier == "giuliani":
            delimiter = " "
            column_names = ["unknown0", "kf", "E/A"] + [f"unknown{i}" for i in range(1, 5+1)]
        else:
            delimiter = ","
            column_names = ["rho0", "E/A"]

        for file in sorted(files):
            data_read = pd.read_csv(file, comment="#", names=column_names, skiprows=0, delimiter=delimiter)
            data_read["file"] = file
            if set_specifier == "giuliani":
                def calc_density(kf, g=2):
                    return 2*g*kf**3/(6*np.pi**2)
                data_read = pd.DataFrame(data={"rho0": calc_density(data_read["kf"]),
                                               "E/A": data_read["E/A"]})
                id = 1
            else:
                id = int(re.search(r"samples(\d)", file).group(1))

            data_read["class"] = f"{set_specifier}:{id}"
            data_read["label"] = data_read["class"]
            data = pd.concat([data, data_read])
        return files, data

    def sample(self, df=None, num_points=1, num_distr="all", num_pts_per_distr=None,
               replace=True, exclude=None, random_state=None, **kwargs):
        class_lbl = self.data_frame["class"].unique()
        if isinstance(num_distr, int):
            class_lbl = np.random.choice(class_lbl, num_distr, replace=replace)
        elif num_distr != "all":
            raise ValueError(f"'num_distr' should be int or 'all', got '{num_distr}'.")
        num_distr = len(class_lbl)
        print(self.set_specifier, random_state)
        if num_pts_per_distr is None:
            num_pts_per_distr = int(np.max([1, num_points/num_distr]))

        ret = pd.DataFrame()
        dframe_filtered = self.get_data_frame(exclude=exclude)
        for cls in class_lbl:
            tmp = dframe_filtered[dframe_filtered["class"] == cls]
            samples = tmp.sample(num_pts_per_distr, replace=replace, random_state=random_state)
            # print(cls, samples)
            for lbl in ("class", "label"):
                samples[lbl] = cls
            ret = pd.concat((ret, samples))

        if num_points is not None:
            ret = ret.sample(n=num_points, replace=len(ret) < num_points, random_state=random_state)

        if df is not None:
            ret = pd.concat((df, ret))
        return ret

    def plot(self, ax=None, level=0.95, num_distr=1, fill=True, plot_scatter=False, marker_size=8,
             add_legend=True, add_axis_labels=True, exclude=None, use_seaborn=False, **kwargs):
        if ax is None:
            ax = plt.gca()

        if "additional_legend_handles" not in kwargs.keys():
            print("yeah")
            additional_legend_handles = []
        else:
            additional_legend_handles = kwargs["additional_legend_handles"]

        import matplotlib.patches as mpatches
        for icls, cls in enumerate(self.data_frame["class"].unique()[:num_distr]):
            dframe_filtered = self.get_data_frame(exclude=exclude)
            mask = dframe_filtered["class"] == cls
            if use_seaborn:
                levels = 1. - np.atleast_1d(level)
                if fill:
                    levels = np.append(levels, 1.)
                sns.kdeplot(ax=ax, x=dframe_filtered[mask]["rho0"], y=dframe_filtered[mask]["E/A"],
                            fill=fill, levels=levels,
                            label=f"{cls} ({(1-levels[0])*100:.0f}\%)",
                            legend=True,
                            color=colors[-icls])
                # Note: sns.kdeplot() seems to have issues with displaying the handles in legends
            else:
                if "schunck" in cls.lower():
                    use_colorset = sorted(colors)
                    label = "Schunck $\it{et~al.}$ (\'20" + f", {level*100:.0f}\%)"
                elif "giuliani" in cls.lower():
                    use_colorset = sorted(flatui)
                    label = "Giuliani $\it{et~al.}$ (\'22" + f", {level*100:.0f}\%)"
                else:
                    use_colorset = sorted(colorset)
                    label = cls

                color = use_colorset[-icls]
                corner.hist2d(x=dframe_filtered[mask]["rho0"].to_numpy(),
                              y=dframe_filtered[mask]["E/A"].to_numpy(),
                              bins=25, range=None, weights=None,
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
