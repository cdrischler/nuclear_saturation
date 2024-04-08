import numpy as np
from scipy.stats import multivariate_t
import matplotlib as mpl
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from collections.abc import Iterable
from scipy import optimize


cm=1./2.54

fontsize = 9
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['axes.titlesize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.title_fontsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['figure.titlesize'] = fontsize
mpl.rcParams['text.usetex'] = True
mpl.rcParams['figure.dpi'] = 130
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.direction'] = 'in'
plt.rcParams["xtick.minor.visible"] =  True
plt.rcParams["ytick.minor.visible"] =  True
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True
mpl.rc('savefig', transparent=False, bbox='tight', pad_inches=0.05, format='pdf')

color_68 = 'darkgrey'   # color for 1 sigma bands
color_95 = 'lightgrey'  # color for 2 sigma bands
mpt_default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']  # 8 colors
colorset = ['b', 'r', 'darkcyan', 'darkslateblue', 'orange', 'darkgrey', 'lime', 'aqua', 'g', 'magenta', 'k']  +  colors
markers = {'0': "*", '2': "o", '3': "s", '4': "D", '5': "p" }

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
[purple, blue, grey, red, darkblue, green] = flatui
orange = '#f39c12'
black = 'k'
yellow = 'yellow'

color_list = ['Oranges', 'Greens', 'Blues', 'Reds', 'Purples', 'Greys', 'plasma']
cmaps = [plt.get_cmap(name) for name in color_list]
colors_alt = [cmap(0.55 - 0.1 * (i == 0)) for i, cmap in enumerate(cmaps)]
colors_alt2 = plt.get_cmap('tab20').colors

latex_markers = ["$\medblackstar$",
                 "$\medblackdiamond$",
                 "$\medblacksquare$",
                 "$\medblackcircle$",
                 "$\medblacktriangledown$",
                 "$\medblacktriangleleft$",
                 "$\medblacktriangleright$",
                 "$\medblacktriangleup$"]


def highlight_saturation_density(ax, n0=0.164, n0_std=0.007, zorder=-1, alpha=0.5, color='0.6'):
    """
    highlights specified `n0 +/- n0_std` range on axis `ax` (vertical span). 
    Other parameters determine its appearence.
    """
    ax.axvspan(n0-n0_std, n0+n0_std, zorder=zorder, alpha=alpha, color=color)


class HandlerEllipse(HandlerPatch):
    """
    simple helper class for plotting confidence ellipses
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def confidence_ellipse_mean_cov(mean, cov, ax, n_std=3.0, facecolor='none', label="", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor, label=label,
                      **kwargs)

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', show_scatter=False, label="", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    mean = np.array([mean_x, mean_y])
    cov = np.cov(x, y)
    #print(mean)
    #print(cov)

    if show_scatter:
        scat_color = darken_color(facecolor, 0.5)
        ax.plot(x, y, ls='', marker='.', markersize=0.6, color=scat_color)

    return confidence_ellipse_mean_cov(mean, cov, ax, n_std=n_std, facecolor=facecolor, label=label, **kwargs)


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def darken_color(color, amount=0.5):
    """
    Darken the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> darken_color('g', 0.3)
    >> darken_color('#F034A3', 0.6)
    >> darken_color((.3,.55,.1), 0.5)
    """
    return lighten_color(color, 1./amount)


def plot_rectangle(center, uncertainty, ax=None, **kwargs):
    """
    plots a rectangle centered at `center` and with width & height specified by uncertainty

    :param center: center of the rectangle, 2d array
    :param uncertainty: symmetric uncertainty, i.e., length of rectangle = 2 uncertainty; 2d array
    :param ax: axis for plotting
    :param kwargs: additional keyword arguments
    :return: updated ax
    """
    if ax is None:
        ax = plt.gca()
    from matplotlib.patches import Rectangle
    left = center[0] - uncertainty[0]
    right = center[0] + uncertainty[0]
    rect = Rectangle(
        (left, center[1] - uncertainty[1]), width=right-left,
        height=2*uncertainty[1], **kwargs
    )
    ax.add_patch(rect)
    return ax


def plot_confregion_bivariate_t(mu, Sigma, nu, ax=None, alpha=None, alpha_unit="decimal", num_pts=10000000,
                                plot_scatter=True, validate=True,
                                sym_tol=1e-12, radius_tol=1e-3, ellipse_tol=1e-10, **kwargs):
    """
    Plots the confidence region of the bivariate t-distribution efficiently (without sampling)
    
    :param mu: mean vector (length-2)
    :param Sigma: sym., pos. def. scale matrix (will be verified)
    :param nu: degree of freedom, nu > 0
    :param ax: matplotlib axis used for plotting
    :param alpha: credibility levels [float or array of floats],
    either in decimal or in units of sigma (normal distribution) if alpha_unit="normal_std"

    :param alpha_unit: see alpha
    :param num_pts: number of points used for sampling (if scatter plot or validation is requested)
    :param plot_scatter: sample and plot `num_pts` points from distribution (bool)
    :param validate: validate the obtained confidence region by sampling `num_points` points (bool)
    :param sym_tol: tolerance for checking for symmetric matrix
    :param radius_tol: tolerance for checking radius in deformed coordinate system
    :param ellipse_tol: tolerance for checking confidence ellipse coordinate system

    :return: Nothing
    """
    # use default levels if levels are not specified
    if alpha is None and alpha_unit == "decimal":
        alpha = np.array((0.5, 0.8, 0.95, 0.99))

    if alpha is None and alpha_unit == "normal_std":
        alpha = np.array((range(1, 4)))

    alpha = np.sort(np.atleast_1d(alpha), False)[::-1]

    # handle limit nu-->inf: normal distribution
    if np.isinf(nu):
        nu = 99999  # might be a bit crude (than version commented out) but keeps this function self-contained
        # n_std = np.sqrt(-np.log(1.-alpha)*2)
        # for val in n_std:
        #     confidence_ellipse_mean_cov(mean=mu, cov=Sigma, ax=ax, n_std=val, **kwargs)
        # return

    # pick current axis if none is specified
    if ax is None:
        ax = plt.gca()
        
    # preparations and checks
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)
    if alpha_unit == "normal_std":
        alpha = 1-np.exp(-alpha**2/2)
    if np.any(alpha <= 0) or np.any(alpha > 1):
        raise ValueError(f"alpha has to be a (list of) confidence level(s), got {alpha}")

    if mu.shape != (2,) or Sigma.shape != (2,2):
        raise ValueError("requires a bivariate distribution function")
    if nu <=0:
        raise ValueError(f"`nu` must be positive, got {nu}")    
    
    # Sigma has to be symmetric...
    stat_sym = np.abs(Sigma[0,1]-Sigma[1,0]) < sym_tol
    if not stat_sym:
        raise ValueError("`Sigma` has to be symmetric.")
        
    # ... and positive definite
    lambda_, Q = np.linalg.eig(Sigma) 
    if (lambda_ <= 0).any():
        raise ValueError("`Sigma` is not positive definite.")
        
    # plot sampling points from distribution?
    samples = multivariate_t.rvs(mu, Sigma, df=nu, size=num_pts) if plot_scatter or validate else None
    if plot_scatter:
        ax.scatter(samples[:, 0], samples[:, 1], s=0.2)

    # determine radius in deformed coordinate system
    r0 = np.sqrt(nu/(1-alpha)**(2/nu) - nu)
    # print(f"r0 {r0} at level {alpha*100}")

    # create an ellipse
    angle = np.arctan2(Q[1, 0], Q[0, 0])  # compute the angle of the axis associated with the first eigenvalue and the x axis
    axes_lengths = 2*np.sqrt(lambda_) * r0[:, np.newaxis]

    for iellipse, axes_length in enumerate(axes_lengths):
        ell = Ellipse(xy=(mu[0], mu[1]),
                      width=axes_length[0],
                      height=axes_length[1],
                      angle=np.rad2deg(angle), **kwargs)
        if "facecolor" not in kwargs.keys():
            ell.set_facecolor('None')
        if "edgecolor" not in kwargs.keys():
            ell.set_edgecolor(colors[iellipse])
        ell.set_label(f"{alpha[iellipse]*100:.0f}\%")
        ax.add_patch(ell)
                
    # check that the confidence region is legit? (very simplistic check)
    if validate:
        invSigma = np.linalg.inv(Sigma)
        unit_circle = np.array([[np.cos(phi), np.sin(phi)] for phi in np.linspace(0, 2*np.pi, num_pts)]).T
        for ir0i, r0i in enumerate(r0):
            # get points on circle in the deformed coordinate system
            circle = r0i*unit_circle

            # gets points on final confidence ellipse and plot them
            conf_ellipse = ((Q @ np.sqrt(np.diag(lambda_))) @ circle).T + mu
            ax.plot(conf_ellipse[:, 0], conf_ellipse[:, 1], c="w", ls=":")  # c="g") #**kwargs)

            # check radius r0 and corresponding confidence interval
            X = samples-mu
            est_conf = np.sum(np.einsum('ij,jk,ik->i', X, invSigma, X, optimize=True) <= r0i**2)/len(samples)
            X = conf_ellipse-mu
            dev = np.linalg.norm((np.einsum('ij,jk,ik->i', X, invSigma, X, optimize=True) - r0i**2), ord=np.inf)

            err_radius = (np.abs(est_conf-alpha[ir0i]) > radius_tol)
            err_ellipse = (dev > ellipse_tol)
            if err_radius or err_ellipse:
                print(err_ellipse, err_radius)
                print(dev, ellipse_tol, np.abs(est_conf-alpha[ir0i]), radius_tol)
                raise ValueError(f"Obtained confidence region not consistent. Estimated alpha={est_conf:.4f} (expected: {alpha[ir0i]:.4f})")
            else:
                print(f"confidence ellipse at level '{alpha[ir0i]}' validated.")
    return ax


def plot_confregion_univariate_t(mu, Sigma, nu, ax=None, alpha=None, num_pts=100000000,
                                plot_hist=False, validate=True, orientation="horizontal",
                                atol=1e-3, plot_quantile="line", **kwargs):
    """
    plot confidence region of the univariate student t-distribution t_nu(mu, Sigma) 

    Parameters
    ----------
    mu, Sigma, nu : parameters of the distribution
    ax : The axes object to draw the ellipse into. (matplotlib.axes.Axes)
    alpha: transparency value
    num_pts: number of points used for validation (if enabled) via sampling
    plot_hist: specifies whether or not to plot a histogram after smapling
    validate: specifies whether to validate the determined confidence region using sampling
    orientation: orientation of the histogram plot
    atol: absolute tolerance used to determine whether validation passed
    plot_quantile: specifies whether quantiles are plotted as a "band" or "line"

    Returns
    -------
    confidence region as numpy array
    """
        
    # pick current axis if none is specified
    if ax is None:
        ax = plt.gca()

    if alpha is None:
        alpha = np.array((0.5, 0.8, 0.95, 0.99))
    alpha = np.sort(np.atleast_1d(alpha))[::-1]

    from scipy.stats import t

    def conf_region(x0, df, loc, scale, alpha):
        cdf_diff = t.cdf(loc+x0, df=df, loc=loc, scale=scale) - t.cdf(loc-x0, df=df, loc=loc, scale=scale)
        return cdf_diff - alpha

    samples = t.rvs(size=num_pts, df=nu, loc=mu, scale=Sigma) if plot_hist or validate else None
    if plot_hist:
        n, bins, patches = plt.hist(samples, 100, density=True, color='green', alpha=0.7, orientation=orientation)

    conf_intervals = []
    for ialph, alph in enumerate(alpha):
        # print(alph, Sigma, mu, nu)
        sol = optimize.newton(func=conf_region, x0=Sigma, args=(nu, mu, Sigma, alph))
        # print(sol)
        current_interval = {"alpha": alph, "mu": mu, "Delta(+/-)": sol, "left": mu-sol, "right": mu+sol}
        conf_intervals.append(current_interval)
        if validate:
            alpha_est = np.sum(np.abs(samples - mu) <= sol)/num_pts
            diff = np.abs(alph-alpha_est)
            if diff > atol:
                raise ValueError(f"requested tolerance not achieved; diff {diff}")
            else:
                print(f"confidence ellipse at level '{alph}' validated.")

        kwargs = dict(zorder=-1, alpha=1, color=colors[ialph])
        if plot_quantile == "band":
            if orientation == "vertical":
                ax.axvspan(current_interval["left"], current_interval["right"], **kwargs)
            else:
                ax.axhspan(current_interval["left"], current_interval["right"], **kwargs)
        elif plot_quantile == "line":
            kwargs = dict(zorder=-1, alpha=1, lw=0.8, ls="--", color=colors[ialph])
            for quant in ("left", "right"):
                if orientation == "vertical":
                    ax.axvline(current_interval[quant], **kwargs)
                else:
                    ax.axhline(current_interval[quant], **kwargs)

    return np.array(conf_intervals)


def test_plot_confregion_univariate_t():
    """
    simple unit test to make sure the limit nu --> infty of the studen t-distribution (Gaussian) is correct.
    Raises an error if not.
    """
    mu = 4
    Sigma = 2
    nu = 90000  # hence, approx. a normal distribution
    alpha = (0.6826894921370859, 0.9544997361036416, 0.9973002039367398)
    conf_intervals = plot_confregion_univariate_t(mu=mu, Sigma=Sigma, nu=nu, alpha=alpha, validate=True)
    assert conf_intervals


def fit_bivariate_t(data, alpha_fit=0.68, nu_limits=None, tol=1e-2, print_status=False, strategy="fit_marginals"):
    """
    optimizes a bivariate t-distribution to samples in `data` 

    Parameters
    ----------
    data : array with (x,y) samples
    alpha_fit : confidence level at which to optimize
    nu_limits : limits (lower, upper) of the dof nu
    tol: requested tolerance of the fit
    plot_hist: specifies whether or not to plot a histogram after smapling
    print_status: boolean as to print debugging information
    strategy: specifies how the fit is performed either by fitting the marginals ("fit_marginals")
    or by counting samples in a given confidence region (else)

    Returns
    -------
    dictionary specifying the parameters of the optimized bivariate t-distribution
    """
        
    if strategy == "fit_marginals":
        from scipy.stats import t
        nu_first, mean_first, std_first = t.fit(data=data[:,0])
        nu_second, mean_second, std_second = t.fit(data=data[:,1])
        nu_est = np.mean([nu_first, nu_second])
        mu_est = [mean_first, mean_second]
        Psi_est = np.cov(data[:,0],data[:,1]) * (nu_est-2)/nu_est
    else:  # count samples
        if nu_limits is None:
            nu_limits = (3, 40)

        # estimate mean value (simple!)
        mu_est = data.mean(axis=0)

        # estimate scale matrix (from covariance matrix)
        cov_est = np.cov(data[:, 0], data[:, 1])
        inv_cov_est = np.linalg.inv(cov_est)

        def metric(nu, alpha):
            if print_status:
                print(f"current nu: {nu}")
            rhs = ((nu - 2)/nu) * (nu/(1-alpha)**(2/nu) - nu)  # note that this includes the radius squared!
            X = data - mu_est
            alpha_est = np.sum(np.einsum('ij,jk,ik->i', X, inv_cov_est, X) <= rhs)/len(data)
            return alpha_est - alpha
        try:
            nu_est = optimize.bisect(metric, nu_limits[0], nu_limits[1], args=(alpha_fit,), xtol=tol)
            Psi_est = cov_est * (nu_est-2) / nu_est
        except ValueError as e:
            print(f"fit of a bivariate t distribution with finite dof in ({nu_limits}) failed: '{str(e)}'")
            print("assuming Normal distribution instead")
            nu_est = np.inf
            Psi_est = cov_est
    return {"mu": mu_est, "Psi": Psi_est, "nu": np.rint(nu_est)}


def test_fit_bivariate_t(df=7, M=None, mu=None, size=10000000, tol=1e-3, print_status=True):
    """
    simple test of `fit_bivariate_t()` using brute-force sampling `size` times. 
    `df`, `M`, `mu` specify the dof, shape matrix, and mean vector of the t-distribution used to generate
    mock data;

    raises an error if the test doesn't pass
    """
    nu_limits = (3, 8)
    if np.isfinite(df) and df > nu_limits[1]:
        print(f"requested (finite) df={df} too high")
        return

    if M is None:
        M = np.array([[1, 0], [0, 2]])
    if mu is None:
        mu = np.array([1, 2])
    if np.isfinite(df):
        is_normal_distr = False
        from scipy.stats import multivariate_t
        data = multivariate_t.rvs(df=df, loc=mu, shape=M, size=size)
    else:
        is_normal_distr = True
        from scipy.stats import multivariate_normal
        data = multivariate_normal.rvs(mean=mu, cov=M, size=size)

    fit = fit_bivariate_t(data, nu_limits=nu_limits, print_status=print_status)

    stat_mu = np.abs(mu-fit["mu"]) < tol
    stat_psi = np.abs(M-fit["Psi"]) < tol
    stat_nu = np.abs(df-fit["nu"]) < tol if not is_normal_distr else True
    assert np.any(stat_mu) and np.any(stat_psi) and stat_nu
# test_fit_bivariate_t()

#%%