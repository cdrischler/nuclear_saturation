import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from modules.plot_helpers import purple, grey, flatui, colorset, darkblue, green, blue
from modules.plot_helpers import lighten_color, colors_alt, confidence_ellipse_mean_cov


annotate_fs = 7
avail_srcs = []


# Hebeler et al.
data_H = pd.read_csv("./data/Sv_L/H.csv", names=['Esym', 'L'], comment="#")
avail_srcs.append({
    "label" : "H", "facecolor" : purple, 'Esym': data_H['Esym'], 'L': data_H['L'],
    'label_x': 0.42, 'label_y': 0.34, 'ha': 'left', 'va': 'bottom', 'label_color': 'k',
    'use_spline': False, 'fontsize':annotate_fs,
    #     'reference': 'H: Hebeler, K., Lattimer, J. M., Pethick, C. J.,  Schwenk, A, PRL, 105, 161102'
    'reference': 'Hebeler \\textit{et al.},\nPRL \\textbf{105}, 161102 (2010)'
})

# Gandolfi et al.
data_G = pd.read_csv("./data/Sv_L/G.csv", names=['Esym', 'L'], comment="#")
avail_srcs.append({
    "label" : "G", "facecolor" : grey, 'Esym': data_G['Esym'], 'L': data_G['L'],
    'label_x': 0.55, 'label_y': 0.39, 'ha': 'left', 'va': 'bottom', 'label_color': 'k',
    'use_spline': False, 'fontsize':annotate_fs,
    #     'reference': 'G: Gandolfi, S., Carlson, J.,  Reddy, S., PRC, 85, 032801'
    'reference': 'Gandolfi \\textit{et al.},\nPRC \\textbf{85}, 032801 (2012)'
})

# Tews, Krueger et al.
data_TK = pd.read_csv("./data/Sv_L/TK.csv", names=['Esym', 'L'], comment="#")
avail_srcs.append({
    "label" : "TK", "facecolor" : blue, 'Esym': data_TK['Esym'], 'L': data_TK['L'],
    'label_x': 0.36, 'label_y': 0.46, 'ha': 'left', 'va': 'bottom', 'label_color': 'k',
    'use_spline': False, 'fontsize':annotate_fs, "alpha":0.6
    #'reference': 'Gandolfi \\textit{et al.},\nPRC \\textbf{85}, 032801 (2012)'
})

# Huth, Wellenhofer et al.
# data_HWS = pd.read_csv("./data/Sv_L/HWS.csv", names=['Esym', 'L'], comment="#")
# avail_srcs.append({
#     "label" : "HWS", "facecolor" : green, 'Esym': data_HWS['Esym'], 'L': data_HWS['L'],
#     'label_x': 0.48, 'label_y': 0.48, 'ha': 'left', 'va': 'bottom', 'label_color': 'k',
#     'use_spline': False, 'fontsize':annotate_fs, "alpha":0.6
#     #'reference': 'Gandolfi \\textit{et al.},\nPRC \\textbf{85}, 032801 (2012)'
# })


# Unitary gas limit
def getUgConstraint( ut, EUG0, Kn, Qnlower, Qnupper, E0 ):
    taylorDiff = ut-1.
    Qn = np.where(ut < 1, Qnlower, Qnupper)
    Esym = EUG0/(3.*np.cbrt(ut)) * (ut+2.) + Kn/18. * taylorDiff**2 + Qn/81. * taylorDiff**3 - E0
    L = 2.*EUG0/(np.cbrt(ut)) - Kn/3. * taylorDiff - Qn/18. * taylorDiff**2
    return Esym, L


def getUgAnalyticConstraint( L, EUG0, E0 ):
    return L/6. * ( 1. + 2. * (2.* EUG0 / L)**(3/2) ) - E0


def plot_source(
        ax, Esym, L=None, spline_lower=None, spline_upper=None, use_spline=True,
        label=None, label_x=0, label_y=0, ha=None, va=None, reference=None,
        label_color='k', rotation=None, bbox=None, fontsize=None,
        facecolor=None, edgecolor='k', ls='-', lw=0.5, hatch=None,
        zorder=None, **kwargs
):
    if not use_spline:
        ax.fill(
            Esym, L, edgecolor=edgecolor, ls=ls, lw=lw, facecolor=facecolor,
            hatch=hatch, zorder=zorder, **kwargs
        )
    else:
        old_facecolor = facecolor
        if hatch is not None:
            facecolor = "none"
        ax.plot(Esym, spline_lower(Esym), c=edgecolor, ls=ls, lw=lw, zorder=zorder)
        ax.plot(Esym, spline_upper(Esym), c=edgecolor, ls=ls, lw=lw, zorder=zorder)
        ax.fill_between(
            Esym, spline_lower(Esym), spline_upper(Esym),
            edgecolor=old_facecolor, ls=ls, lw=0, facecolor=facecolor,
            hatch=hatch, zorder=zorder, **kwargs
        )

    if label is not None:
        if bbox is True:
            bbox = dict(facecolor='w', boxstyle='round', alpha=1)
        ax.text(
            label_x, label_y, label, fontdict=dict(color=label_color),
            rotation=rotation, transform=ax.transAxes, bbox=bbox, zorder=100,
            fontsize=fontsize, ha=("left" if ha==None else ha), va=("bottom" if va==None else va)
        )
    return ax


def plot_holt_kaiser(ax, color='w', alpha=1, lw=1.1, zorder=10):
    #  Holt/Kaiser PRC 2017
    alf1=1.41508   ## correlation angle
    alf2=1.43502   ## correlation angle
    alf3=1.42708   ## correlation angle

    x1=31.3  ## central values of X, and Y
    y1=41.9  ##
    x2=32.3  ## central values of X, and Y
    y2=50.0  ##
    x3=31.5  ## central values of X, and Y
    y3=44.8  ##

    ## a2 and b2 are major axis and minor axis
    a1 = 22.2
    b1 = 0.64
    a2 = 36.0
    b2 = 0.86
    a3 = 44.5
    b3 = 1.10

    ## arrary for the plot

    imax = 201
    xp1 = np.zeros(imax)
    yp1 = np.zeros(imax)

    for i in range(0,imax):
        theta = 2.*np.pi*i/(imax-1)
        x = a1*np.cos(theta)
        y = b1*np.sin(theta)
        xp1[i] = x*np.cos(alf1) - y*np.sin(alf1) + x1
        yp1[i] = x*np.sin(alf1) + y*np.cos(alf1) + y1

    #ax.fill(xp1, yp1, facecolor=color, edgecolor='k', lw=lw, alpha=alpha)
    ax.plot(xp1, yp1, ls="-", lw=lw, color=color, alpha=alpha, zorder=zorder)

    i = 0
    bbox_dict = dict(boxstyle="round,pad=0.5", fc=color, alpha=alpha, ec="none", lw=lw)
    ax.annotate(f"HK",
                xy=(x1+0.5,y1),
                xytext=(33.5,42), textcoords='data',
                arrowprops={'arrowstyle':'-','color':color, 'alpha':alpha,'relpos':(0., 0.5),
                            'shrinkA':5, 'shrinkB':5, 'patchA':None, 'patchB':None,
                            'connectionstyle':"angle3,angleA=0,angleB=100"},
                horizontalalignment='left', verticalalignment='bottom',
                rotation=0, size=5, zorder=i+1, bbox=bbox_dict)

    image_type = 'pdf'


Esym_L_eft = [dict(), dict()]
# n=0.17 +/- 0.01, Lambda = 500
Esym_L_eft[0]["mean"] = np.array([31.69810539, 59.830485  ])
Esym_L_eft[0]["cov"] = np.array([
    [ 1.23651375,  3.27499281],
    [ 3.27499281, 16.95157735],
])

# n=0.17 +/- 0.01, Lambda = 450
Esym_L_eft[1]["mean"] = np.array([33.52341527, 67.79737685])
Esym_L_eft[1]["cov"] = np.array([
    [ 1.57488007,  3.05934762],
    [ 3.05934762, 15.97354339],
])


def plot_UG_constraint(ax, plot_analytic=False):
    # Add the unitary gas constraint boundaries
    ut = np.linspace(0.001, 2, 100)
    tews_zorder = 11
    n0 = 0.157 # fm**-3
    hbarc = 197.3269718 # MeV fm
    Mn = 939.565379 # MeV
    EUG0 = 3./(10.*Mn) * np.cbrt(3.*np.pi**2*n0)**2*hbarc**2*0.365 # about 12.64 MeV
    E0 = -15.5 # MeV

    TewsEtAlSetting = {
        "EUG0" : EUG0, "E0" : E0, "Kn" : 270,
        "Qnlower" : -750., "Qnupper" : 0.
    }
    Esym_tews, L_tews = getUgConstraint( ut, **TewsEtAlSetting)
    ax.plot(Esym_tews, L_tews, c='k', lw=1.0, zorder=tews_zorder)

    idx_arrow_tews = 60
    ax.arrow(Esym_tews[idx_arrow_tews], L_tews[idx_arrow_tews], 1, 0, head_length=0.3,
             head_width=2, zorder=tews_zorder, facecolor="k")
    ax.text(Esym_tews[idx_arrow_tews]+0.15, L_tews[idx_arrow_tews]+1.4,
            'UG', ha='left', va='bottom', zorder=tews_zorder, fontsize=annotate_fs)

    # now plot analytic constraint
    if plot_analytic:
        L_grid = np.linspace(0.001, 120, 100)
        Esym_tews_analytic = getUgAnalyticConstraint(L_grid, EUG0, E0)
        ax.plot(Esym_tews_analytic, L_grid, c='k', ls='--', lw=1.0, zorder=tews_zorder)

        idx_arrow_tews_a = 10
        ax.arrow(
            Esym_tews_analytic[idx_arrow_tews_a], L_grid[idx_arrow_tews_a], 1, 0,
            head_length=0.3, head_width=2, zorder=tews_zorder, facecolor="k")
        ax.text(Esym_tews_analytic[idx_arrow_tews_a]+0.15, L_grid[idx_arrow_tews_a]+1.4,
                'UG Analytic', ha='left', va='bottom', zorder=tews_zorder, fontsize=annotate_fs)

    # ug_ref = 'UG: Tews, I., Lattimer, J. M., Ohnishi, A. , Kolomeitsev, E. E., APJ 848, 105'


def make_sv_l_plot(ax):
    # set labels and title
    ax.set_xlabel(r"Symmetry Energy $S_v$ [MeV]")
    ax.set_ylabel(r"Slope Parameter $L$ [MeV]")
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    # set limits
    ax.set_xlim(27, 35)
    ax.set_ylim(0, 100)
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.tick_params(width=0.7, which='major')

    # external sources
    i = 1
    for src in avail_srcs:
        plot_source(ax, zorder=i, **src)
        i += 1

    # UG constraints
    plot_UG_constraint(ax)
    i += 1

    # GP-B constraints
    for ieft, Esym_L in enumerate(reversed(Esym_L_eft)):
        color =lighten_color(colors_alt[0], 0.5*(1+ieft))
        (alpha1s, alpha2s) = (0.6, 0.5)
        confidence_ellipse_mean_cov(
            Esym_L["mean"], Esym_L["cov"], ax=ax, n_std=2,
            facecolor=color, edgecolor='k', alpha=alpha2s, zorder=i+1
        )
        confidence_ellipse_mean_cov(
            Esym_L["mean"], Esym_L["cov"], ax=ax, n_std=1,
            facecolor=lighten_color(color, 0.6), edgecolor='k', alpha=alpha1s, zorder=i+1
        )

        bbox_dict = dict(boxstyle="round,pad=0.5", fc=color, alpha=alpha1s, ec="none", lw=1)
        ax.annotate(f"GP--B {[450,500][ieft]}",
                    xy=Esym_L["mean"],
                    xytext=(30., 85.5-ieft*6), textcoords='data',
                    arrowprops={'arrowstyle':'-','color':color, 'alpha':alpha1s,'relpos':(1., 0.5),
                                'shrinkA':5, 'shrinkB':5, 'patchA':None, 'patchB':None,
                                'connectionstyle':"angle3,angleA=-11,angleB=100"},
                    horizontalalignment='left', verticalalignment='bottom',
                    rotation=0, size=5, zorder=i, bbox=bbox_dict)

    # plot HK constraint
    i += 1
    plot_holt_kaiser(ax, color=lighten_color(colors_alt[6], 1.))

    # plot PREX-informed result
    mean = np.array([38.1, 106])
    cov = np.diag([4.7, 37])**2
    for n_std in (3, 2, 1):
        confidence_ellipse_mean_cov(
            mean, cov, ax=ax, n_std=n_std,
            facecolor=lighten_color("k", 1/n_std), edgecolor='k', alpha=0.66, zorder=0
        )
    ax.text(
        0.55, 0.2, "PREX--II informed ($3\sigma$)", fontdict=dict(color="k"),
        rotation=0, transform=ax.transAxes,
        fontsize=annotate_fs
    )

#%%
