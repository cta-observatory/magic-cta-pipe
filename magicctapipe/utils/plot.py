import os
import numpy as np
import matplotlib.pylab as plt


def save_plt(n, rdir="", vect="pdf,eps"):
    """Save plot in the required formats

    Parameters
    ----------
    n : str
        plot name
    rdir : str, optional
        directory, by default ''
    vect : str, optional
        vectory formats, leave empty to save only png, by default "pdf,eps"
    """
    if os.path.exists(os.path.dirname(rdir)):
        for vect_ in vect.split(","):
            if vect_ == "pdf" or vect_ == "eps":
                plt.savefig(os.path.join(rdir, "%s.%s" % (n, vect_)))
        plt.savefig(os.path.join(rdir, "%s.png" % (n)), dpi=300)
    else:
        print("Figure NOT saved, directory doesn't exist")
    return


def load_default_plot_settings():
    """Load default plot settings"""
    params = {
        "figure.figsize": (10, 6),
        "savefig.bbox": "tight",
        "axes.grid": True,
        "errorbar.capsize": 3,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
    plt.rcParams.update(params)
