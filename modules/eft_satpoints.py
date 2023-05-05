import numpy as np
import pandas as pd


def plot_GPB_coester(ax, pos, lam=500, n_std=2, color="blue", annotate=True):
    if lam == 500:
        # Lambda 500 MeV
        mean = np.array([ 0.17014727, -14.2950795 ])
        cov = np.array([[ 2.66832950e-04, -1.51811388e-02], [-1.51811388e-02, 1.02588208e+00]])
    elif lam == 450:
        # Lambda 450 MeV
        mean = np.array([  0.17282904, -14.89276732])
        cov = np.array([[ 1.99079971e-04, -1.37165609e-02], [-1.37165609e-02,1.11058727e+00]])
    else:
        print("Error: cutoff unknown")
        return

    # plot ellipse
    from modules.plot_helpers import confidence_ellipse_mean_cov
    confidence_ellipse_mean_cov(mean, cov, ax, n_std=n_std, alpha=1,
                                facecolor=color, edgecolor="k", lw=0.8)

    # plot circumference
    confidence_ellipse_mean_cov(mean, cov, ax, n_std=n_std, facecolor='none',
                                zorder=4, edgecolor="k", lw=0.8)

    if annotate:
        bbox_dict = dict(boxstyle="round,pad=0.5", fc=color, ec="none", lw=1)
        ax.annotate(f"(GP-B {lam} ${n_std}\sigma$)",
                    xy=pos, textcoords='data',
                    arrowprops={'arrowstyle':'-','color':color,'relpos':(1., 0.5),
                                'connectionstyle':"angle3,angleA=0,angleB=75"},
                    horizontalalignment='left',verticalalignment='center', rotation=0,
                    size=5, zorder=8, bbox=bbox_dict
                    )


def plot_coester_band(ax, data_sat, color="lightgray", shift=1.6):
    pars = np.polyfit(data_sat["n0"], data_sat["En0"], 1)
    fit = np.poly1d( pars )
    dens_plt = np.linspace(0.12, 0.21, 5)
    ens_plt = fit( dens_plt )
    ax.fill_between(dens_plt, ens_plt-shift, ens_plt+shift,
                    color=color, alpha=1, zorder=0)


def make_coester_plot(fig, ax, emp_constraint=None, conf_level=None):
    # read Drischler et al. results
    data_sat = pd.read_csv("data/satpoints_predicted.csv", comment="#", skiprows=0, sep=',', header=0)

    # set rc params
    import matplotlib as mpl
    mpl.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r'\usepackage{amssymb}\usepackage{fdsymbol}')

    # plot empirical point
    from modules.plot_helpers import latex_markers, plot_rectangle, colors_alt2
    kwargs_edge = dict(facecolor='none', zorder=5) #, edgecolor='gray')
    kwargs_face = dict(facecolor='w', zorder=1) # , edgecolor='gray')
    if emp_constraint is None:
        emp_constraint = {'type': "box",
                          'rho0': (0.163655, 0.007345), 'E/A': (-15.86, 0.37)}
    if emp_constraint["type"] == "box":
        center = [emp_constraint[key][0] for key in ("rho0", "E/A")]
        uncertainty = [emp_constraint[key][1] for key in ("rho0", "E/A")]
        plot_rectangle(center=center, uncertainty=uncertainty, ax=ax, **kwargs_face)
        plot_rectangle(center=center, uncertainty=uncertainty, ax=ax, **kwargs_edge)
    elif emp_constraint["type"] == "t":
        from modules.plot_helpers import plot_confregion_bivariate_t
        conf_level = (0.5, 0.80, 0.95, 0.99) if conf_level is None else conf_level
        plot_confregion_bivariate_t(ax=ax, mu=emp_constraint["mu"], Sigma=emp_constraint["Psi"],
                                    nu=emp_constraint["nu"], plot_scatter=False, validate=False,
                                    alpha=conf_level, **kwargs_face)
        # plot_confregion_bivariate_t(ax=ax, mu=emp_constraint["mu"], Sigma=emp_constraint["Psi"],
        #                             nu=emp_constraint["nu"], plot_scatter=False, validate=False,
        #                             alpha=conf_level, **kwargs_edge)

        ax.legend(loc="lower center", ncol=len(conf_level), columnspacing=0.8,
                  frameon=False, framealpha=1, edgecolor="0.8",
                  prop={'size': 9}) #, bbox_to_anchor=(0.47,-0.25)) #,bbox_transform=fig.transFigure)

    # plot GP-B ellipses
    plot_GPB_coester(ax, (0.176, -12.6), lam=500, color=colors_alt2[0])
    plot_GPB_coester(ax, (0.176, -12.6-0.55), lam=450, color=colors_alt2[1])

    # plot MBPT results (Drischler et al. and Holt et al.)
    for index, row in data_sat.iterrows():
        # plot only Drischler et al.'s results at order 4
        if (row["mbpt_order"] != 4) and (row["method"] == "MBPT"):
            continue

        # determine color
        color = colors_alt2[row["set_id"]]
        marker = latex_markers[row["mbpt_order"]-1]

        kwargs = {"zorder": 5} if row["method"] == "CC" else {"zorder": 8,
                                                             "markeredgewidth": 0.5,
                                                             "markeredgecolor": '0.'}
        # plot result
        ax.plot(row['n0'], row['En0'],
                marker=marker,
                c=colors_alt2[row["set_id"]],
                alpha=1., ls="-", lw="1.",
                **kwargs
                )

        # annotate
        # compute off set of tags
        (ratio1,ratio2) = (np.diff([0.13, 0.199]),np.diff([-19.4, -11.8])/2)
        delta=0.0095
        bbox_dict = dict(boxstyle="round,pad=0.5", fc=color, ec="none", lw=1)
        if row["method"] != "CC":
            swap_label_side = (row["hamiltonian_brief"] == "sim 475") or (row['n0'] > 0.160 and row['n0'] < 0.174)
            (hat, vat) = ('left', 'bottom') if swap_label_side else ('right', 'top')
            deltat = delta if swap_label_side else -delta/3
            ax.annotate(f"\ ({row['hamiltonian_brief']})\ ",
                        xy=(row['n0']+deltat*ratio1, deltat*ratio2+row['En0']),
                        ha=hat, va=vat, rotation=45,
                        size=5, zorder=6, bbox=bbox_dict)
        elif row["method"] == "CC":
            xypos = (0.153, -15.3) if "sat" in row["hamiltonian_brief"] else (0.159, -17.55-(index-37)*0.55)
            ax.annotate(f"({row['hamiltonian_brief']})",
                        xy=(row['n0'], row['En0']),
                        xytext=xypos, textcoords='data',
                        arrowprops={'arrowstyle':'-','color':color,'relpos':(1., 0.5),
                                    'shrinkA':5, 'shrinkB':5, 'patchA':None, 'patchB':None,
                                    'connectionstyle':"angle3,angleA=30,angleB=75"},
                        horizontalalignment='right', verticalalignment='bottom',
                        rotation=0, size=5, zorder=6, bbox=bbox_dict)

    # plot Coester band
    plot_coester_band(ax, data_sat=data_sat)

    # set title
    # ax.set_title("Nuclear Saturation")

    # axes labels
    ax.set_xlabel("Sat. Density $n_0$ [fm$^{-3}$]")
    ax.set_ylabel("Sat. Energy $E_0$ [MeV]")

    ax.minorticks_on()
    ax.tick_params(which='both', direction='in',
                   bottom=True, top=True, left=True, right=True)

    # axes ranges
    ax.set_xlim(0.13, 0.199)
    ax.set_ylim(-20.2, -11.8)
