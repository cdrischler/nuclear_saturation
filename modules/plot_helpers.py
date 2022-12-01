import numpy as np
from scipy.stats import multivariate_t
import matplotlib as mpl
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

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
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']  # 8 colors
colorset = ['b', 'r', 'darkcyan', 'darkslateblue', 'orange', 'darkgrey', 'lime', 'aqua', 'g', 'magenta', 'k']  +  colors
markers = {'0': "*", '2': "o", '3': "s", '4': "D", '5': "p" }

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
[purple, blue, grey, red, darkblue, green] = flatui
orange = '#f39c12'
black = 'k'
yellow = 'yellow'


def highlight_saturation_density(ax, n0 = 0.164, n0_std = 0.007, zorder=-1, alpha=0.5, color='0.6'):
    ax.axvspan(n0-n0_std, n0+n0_std, zorder=zorder, alpha=alpha, color=color)


class HandlerEllipse(HandlerPatch):
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
    # two-dimensionl dataset.
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


def plot_confregion_bivariate_t(mu, Sigma, nu, ax=None, alpha=3, alpha_unit="normal_std", num_pts=10000, 
                                plot_scatter=True, validate=True, 
                                sym_tol=1e-12, radius_tol=1e-3, ellipse_tol=1e-12, **kwargs):
    """
    Plots the confidence region of the bivariate t-distribution efficiently (without sampling)
    :param mu: mean vector (lenght-2)
    :param Sigma: sym., pos. def. scale matrix (will be verified)
    :param nu: degree of freedom, nu > 0
    :param ax: matplotlib axis used for plotting
    :param alpha: credibility level, either in decimal or in units of sigma (normal distribution) if alpha_unit="normal_std"
    :param alpha_unit: see alpha
    :param num_pts: number of points used for sampling (if scatter plot or validation is requested)
    :param plot_scatter: sample and plot `num_pts` points from distribution (bool)
    :param validate: validate the obtained confidence region by sampling `num_points` points (bool)
    :param sym_tol: tolerance for checking for symmetric matrix
    :param radius_tol: tolerance for checking radius in deformed coordinate system
    :param ellipse_tol: tolerance for checking confidence ellipse coordinate system
    :return: Nothing
    """
    if isinstance(alpha, list):
        for alph in np.sort(alpha)[::-1]:
            plot_confregion_bivariate_t(mu=mu, Sigma=Sigma, nu=nu, 
                                        ax=ax, alpha=alph, alpha_unit=alpha_unit, 
                                        num_pts=num_pts, plot_scatter=plot_scatter, 
                                        validate=validate, sym_tol=sym_tol, **kwargs)
            plot_scatter = False
        return
    
    if ax is None:
        ax = plt.gca()
        
    # preparations and checks
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)
    if alpha_unit == "normal_std":
        alpha = 1-np.exp(-alpha**2/2)
    if alpha <= 0 or alpha > 1:
        raise ValueError(f"alpha has to be a confidence level, got {alpha}")
        
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
        ax.scatter(samples[:,0], samples[:,1], s=0.2)

    # determine radius in deformed coordinate system
    r0 = np.sqrt(nu/(1-alpha)**(2/nu) - nu)
    
    # create an ellipse
    axes_length = 2*r0*np.sqrt(lambda_)  # note that x^T Sigma^-1 x
    angle = np.arctan2(Q[1, 0], Q[0, 0])  # compute the angle of the axis associated with the first eigenvalue and the x axis
    
    ell = Ellipse(xy=(mu[0], mu[1]),
                  width=axes_length[0], 
                  height=axes_length[1],
                  angle=np.rad2deg(angle), **kwargs)
    if "edgecolor" not in kwargs:
        ell.set_facecolor('None')
        ell.set_edgecolor('r')
    ax.add_artist(ell)  
                
    # check that the confidence region is legit? (very simplistic check)
    if validate:
        # get points on circle in the deformed coordinate system
        circle = r0*np.array([[np.cos(phi), np.sin(phi)] for phi in np.linspace(0, 2*np.pi,num_pts)]).T

        # gets points on final confidence ellipse and plot them
        conf_ellipse = ((Q @ np.sqrt(np.diag(lambda_))) @ circle).T + mu
        ax.plot(conf_ellipse[:,0], conf_ellipse[:,1], c="w", ls=":") # c="g") #**kwargs)
        
        # check radius r0 and corresponding confidence interval
        invSigma = np.linalg.inv(Sigma)
        est_conf = np.sum([np.einsum("i,ij,j->", (point-mu), invSigma, (point-mu)) <= r0**2 for point in samples])/len(samples)
        dev = np.linalg.norm(np.array([np.einsum("i,ij,j->", (point-mu), invSigma, (point-mu)) for point in conf_ellipse])-r0**2, 
                             ord=np.inf)
        err_radius = (np.abs(est_conf-alpha) > radius_tol)
        err_ellipse = (dev > ellipse_tol)
        if err_radius or err_ellipse:
            print(err_ellipse, err_radius)
            print(np.abs(est_conf-alpha), radius_tol)
            raise ValueError(f"Obtained confidence region not consistent. Estimated alpha={est_conf:.4f} (expected: {alpha:.4f})")

#%%