import sys; sys.path.append("./modules")
from abc import ABC, abstractmethod
from plot_helpers import plot_rectangle, confidence_ellipse_mean_cov, colors, colorset, flatui
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import glob
import pandas as pd
from scipy.stats import multivariate_normal


class DataSet(ABC):
    def __init__(self, filenames):
        self.filenames = [filenames] if isinstance(filenames, str) else filenames
        self.data_frame = None

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
    def sample(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    @staticmethod
    def set_axes_labels(ax):
        ax.set_xlim(0.145, 0.175)
        ax.set_ylim(-16.5, -14.7)
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

    def sample(self, df=None, num_points=1, replace=True):
        ret = self.data_frame.sample(num_points, replace=replace)
        if df is not None:
            ret = pd.concat((df, ret))
        return ret

    def plot(self, ax=None, plot_scatter=True, plot_box_estimate=False, marker_size=8):
        if ax is None:
            ax = plt.gca()

        if plot_box_estimate:
            res = self.box_estimate()
            center_uncert = [[res["rho0"][i], res["E/A"][i]] for i in range(2)]
            plot_rectangle(center_uncert[0], center_uncert[1], ax=ax,
                           facecolor='lightgray', edgecolor='gray',
                           alpha=0.4, zorder=9)

        if plot_scatter:
            for clbls in self.data_frame["class"].unique():
                idxs = np.where(self.data_frame["class"] == clbls)
                ax.scatter(self.data_frame.iloc[idxs]["rho0"], self.data_frame.iloc[idxs]["E/A"],
                           #color="k",
                           s=marker_size, label=clbls)

        super().set_axes_labels(ax)
        super().legend(ax)

    def box_estimate(self, print_result=False):
        result = dict()
        for dim in ("rho0", "E/A"):
            df = self.data_frame[dim]
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

    def sample(self, df=None, num_distr=1, num_pts_per_distr=1, sample_all=False, replace=True):
        data = self.data_frame if sample_all else self.data_frame.sample(num_distr, replace=replace)
        ret = pd.DataFrame()
        for irow, row in data.iterrows():
            mean, cov = NormDistDataSet.from_row_to_mean_cov(row)
            samples = multivariate_normal.rvs(mean=mean, cov=cov, size=num_pts_per_distr)
            samples = np.atleast_2d(samples)
            result_row = pd.DataFrame(data=samples, columns=("rho0", "E/A"))
            for lbl in ("class", "label", "file"):
                result_row[lbl] = row[lbl]
            ret = pd.concat((ret, result_row))

        if df is not None:
            ret = pd.concat((df, ret))
        return ret

    def plot(self, ax=None, n_std=3, marker_size=8):
        if ax is None:
            ax = plt.gca()
        for irow, row in self.data_frame.iterrows():
            mean, cov = NormDistDataSet.from_row_to_mean_cov(row)
            confidence_ellipse_mean_cov(mean=mean, cov=cov,
                                        ax=ax, n_std=n_std,
                                        facecolor=colorset[irow],
                                        label=f'{row["label"]} (${n_std} \sigma $)')

        super().set_axes_labels(ax)
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
                files = sorted(glob.glob("data/Schunck/samples?.csv"))
            else:
                raise ValueError(f"unknown `set_specifier` '{set_specifier}'")

        data = pd.DataFrame()
        for file in files:
            id = int(re.search(r"samples(\d)", file).group(1))
            data_read = pd.read_csv(file, comment="#", names=("rho0", "E/A"), skiprows=0)
            data_read["file"] = file
            data_read["class"] = f"{set_specifier}:{id}"
            data_read["label"] = data_read["class"]
            data = pd.concat([data, data_read])
        return files, data

    def sample(self, df=None, num_distr=1, num_pts_per_distr=1, sample_all=True, replace=True):
        max_num = 4 if sample_all else num_distr
        class_lbl = [f"{self.set_specifier}:{id}" for id in np.random.choice(np.arange(1, max_num), num_distr, replace=replace)]
        ret = pd.DataFrame()
        for cls in class_lbl:
            tmp = self.data_frame[self.data_frame["class"] == cls]
            samples = tmp.sample(num_pts_per_distr, replace=replace)
            for lbl in ("class", "label"):
                samples[lbl] = cls
            ret = pd.concat((ret, samples))
        if df is not None:
            ret = pd.concat((df, ret))
        return ret

    def plot(self, ax=None, levels=86, num_distr=1, fill=True, plot_scatter=False, marker_size=8):
        if ax is None:
            ax = plt.gca()

        levels = 1. - np.atleast_1d(levels)/100.
        if fill:
            levels = np.append(levels, 1.)
        num_sample = range(1, num_distr+1) if num_distr else range(1, 3+1)
        for isample in num_sample:
            mask = self.data_frame["class"] == f"{self.set_specifier}:{isample}"
            sns.kdeplot(ax=ax, x=self.data_frame[mask]["rho0"], y=self.data_frame[mask]["E/A"], fill=fill, levels=levels,
                        label=f"Schunck set {isample}" + f" ({(1-levels[0])*100:.0f}\%)",
                        legend=True,
                        color=colors[-isample])
            # TODO: sns.kdeplot() seems to have issues with displaying the handles in legends

            if plot_scatter:
                ax.scatter(x=self.data_frame[mask]["rho0"], y=self.data_frame[mask]["E/A"], label="test")

        super().set_axes_labels(ax)
        super().legend(ax)
#%%
