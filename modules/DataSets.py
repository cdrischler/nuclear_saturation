import sys; sys.path.append("./modules")
from abc import ABC, abstractmethod
from plot_helpers import plot_rectangle, confidence_ellipse_mean_cov, colors, colorset
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
from dataclasses import dataclass, field


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
                              num_distr="all", replace=True, quantities=None, prior_params=None, **kwargs):
        if isinstance(self, GenericDataSet):
            data = self.get_data_frame(exclude=exclude)
        else:
            data = self.sample(df=None, num_points=num_points, num_distr=num_distr,
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
        ax.set_xlim(0.145, 0.175)
        ax.set_ylim(-16.5, -14.7)

    @staticmethod
    def set_axes_labels(ax):
        ax.set_xlabel('Saturation Density $n_0$ [fm$^{-3}$]')
        ax.set_ylabel('Saturation Energy $E_0/A$ [MeV]')
        ax.set_title("Empirical saturation box")

    @staticmethod
    def legend(ax):
        ax.legend(ncol=2, loc="upper center", prop={'size': 6})


class GenericDataSet(DataSet):
    def __init__(self, filenames):
        super().__init__(filenames=filenames)
        self.data_frame = super().read_csv_files(self.filenames)

    def __add__(self, other):
        return GenericDataSet(filenames=self.filenames + other.filenames)

    def sample(self, df=None, num_points=1, replace=True, exclude=None, **kwargs):
        ret = self.get_data_frame(exclude=exclude)
        if isinstance(num_points, int):
            ret = self.get_data_frame(exclude=exclude).sample(num_points, replace=replace)
        elif num_points is not None:
            raise ValueError(f"'num_points has to be int or 'None', got {num_points}")
        if df is not None:
            ret = pd.concat((df, ret))
        return ret

    def plot(self, ax=None, plot_scatter=True, plot_box_estimate=False, marker_size=8,
             place_legend=True, add_axis_labels=True, exclude=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        if plot_box_estimate:
            res = self.box_estimate(exclude=exclude)
            center_uncert = [[res["rho0"][i], res["E/A"][i]] for i in range(2)]
            plot_rectangle(center_uncert[0], center_uncert[1], ax=ax,
                           facecolor='lightgray', edgecolor='gray',
                           alpha=0.4, zorder=-5)

        if plot_scatter:
            dframe = self.get_data_frame(exclude=exclude)
            for clbls in dframe["class"].unique():
                idxs = np.where(dframe["class"] == clbls)
                ax.scatter(dframe.iloc[idxs]["rho0"], dframe.iloc[idxs]["E/A"],
                           #color="k",
                           s=marker_size, label=clbls)
        if add_axis_labels:
            super().set_axes_labels(ax)
        if place_legend:
            super().legend(ax)

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

        if use_filenames:
            files = filenames
        else:
            if set_specifier == "fsu_rmf":
                files = sorted(glob.glob("data/Piekarewicz/*/CovEllipse.com"))
            else:
                raise ValueError(f"unknown `set_specifier` '{set_specifier}'")

        data = []; labels = []
        for file in files:
            labels.append(re.search(r"(\w+)/Cov", file).group(1))
            data.append(open(file, 'r').readlines()[3].strip().split(","))
        data = pd.DataFrame(data, columns=("mean rho0", "mean E/A", "sigma rho0", "sigma E/A", "rho"), dtype=np.float64)
        data["label"] = labels
        data["class"] = set_specifier
        data["file"] = files
        return files, data

    @staticmethod
    def from_row_to_mean_cov(row):
        mean = row[["mean rho0", "mean E/A"]].to_numpy()
        offdiag = row["sigma rho0"] * row["sigma E/A"] * row["rho"]
        cov = np.array([[row["sigma rho0"]**2, offdiag], [offdiag, row["sigma E/A"]**2]])
        return mean, cov

    def sample(self, df=None, num_points=1, num_distr="all", num_pts_per_distr=None, replace=True, exclude=None, **kwargs):
        data = self.get_data_frame(exclude=exclude)
        if isinstance(num_distr, int):
            data = data.sample(num_distr, replace=replace)
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
            ret = ret.sample(n=num_points, replace=len(ret) < num_points)

        if df is not None:
            ret = pd.concat((df, ret))
        return ret

    def plot(self, ax=None, level=0.8647, marker_size=8, add_legend=True, set_axis_labels=True,
             add_axis_labels=True, exclude=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        for irow, row in self.get_data_frame(exclude=exclude).iterrows():
            mean, cov = NormDistDataSet.from_row_to_mean_cov(row)
            n_std = np.sqrt(-np.log(1.-level)*2)
            confidence_ellipse_mean_cov(mean=mean, cov=cov,
                                        ax=ax, n_std=n_std,
                                        facecolor=colorset[irow],
                                        label=f'{row["label"]} ({level*100:.0f}\%)')

        if set_axis_labels:
            super().set_axes_ranges(ax)
        if add_axis_labels:
            super().set_axes_labels(ax)
        if add_legend:
            super().legend(ax)


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
               replace=True, exclude=None, **kwargs):
        class_lbl = self.data_frame["class"].unique()
        if isinstance(num_distr, int):
            class_lbl = np.random.choice(class_lbl, num_distr, replace=replace)
        elif num_distr != "all":
            raise ValueError(f"'num_distr' should be int or 'all', got '{num_distr}'.")
        num_distr = len(class_lbl)

        if num_pts_per_distr is None:
            num_pts_per_distr = int(np.max([1, num_points/num_distr]))

        ret = pd.DataFrame()
        dframe_filtered = self.get_data_frame(exclude=exclude)
        for cls in class_lbl:
            tmp = dframe_filtered[dframe_filtered["class"] == cls]
            samples = tmp.sample(num_pts_per_distr, replace=replace)
            for lbl in ("class", "label"):
                samples[lbl] = cls
            ret = pd.concat((ret, samples))

        if num_points is not None:
            ret = ret.sample(n=num_points, replace=len(ret) < num_points)

        if df is not None:
            ret = pd.concat((df, ret))
        return ret

    def plot(self, ax=None, level=0.8647, num_distr=1, fill=True, plot_scatter=False, marker_size=8,
             add_legend=True, add_axis_labels=True, exclude=None, use_seaborn=False, **kwargs):
        if ax is None:
            ax = plt.gca()

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
                # TODO: sns.kdeplot() seems to have issues with displaying the handles in legends
            else:
                corner.hist2d(x=dframe_filtered[mask]["rho0"].to_numpy(),
                              y=dframe_filtered[mask]["E/A"].to_numpy(),
                              bins=20, range=None, weights=None,
                              levels=[level], smooth=None, ax=ax, color=colors[-icls-2],
                              quiet=False, plot_datapoints=plot_scatter, plot_density=True,
                              plot_contours=True, no_fill_contours=False, fill_contours=fill,
                              contour_kwargs=None, contourf_kwargs=None, data_kwargs=None,
                              pcolor_kwargs=None, new_fig=False)

            #if plot_scatter:
            #    ax.scatter(x=dframe_filtered[mask]["rho0"], y=dframe_filtered[mask]["E/A"], label="scatter")

        if add_axis_labels:
            super().set_axes_labels(ax)
        if add_legend:
            super().legend(ax)


@dataclass
class DataSetSampleConfig:
    data_set: DataSet
    sample_from_model: bool = False
    sample_kwargs: dict = field(default_factory=lambda: dict(exclude=None, num_points=None, num_pts_per_distr=1, num_distr="all"))
    sample_from_model_kwargs: dict = field(default_factory=lambda: dict(num_samples=1000, kind="predictive_y",
                                                                        based_on="posterior", validate=False))


@dataclass
class Scenario:
    label: str
    configs: list
#%%
