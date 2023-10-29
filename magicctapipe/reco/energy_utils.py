import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

__all__ = [
    "GetHist2D_energy",
    "evaluate_performance_energy",
    "plot_migmatrix",
]


def GetHist2D_energy(x, y, bins=30, range=None, weights=None):
    hs, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range, weights=weights)
    xloc = (xedges[1:] + xedges[:-1]) / 2
    yloc = (yedges[1:] + yedges[:-1]) / 2

    xxloc, yyloc = np.meshgrid(xloc, yloc, indexing="ij")

    hist = {}
    hist["Hist"] = hs
    hist["X"] = xloc
    hist["Y"] = yloc
    hist["XX"] = xxloc
    hist["YY"] = yyloc
    hist["XEdges"] = xedges
    hist["YEdges"] = yedges

    return hist


def evaluate_performance_energy(data, energy_name):
    valid_data = data.dropna(subset=[energy_name])
    migmatrix = GetHist2D_energy(
        np.lib.scimath.log10(valid_data["true_energy"]),
        np.lib.scimath.log10(valid_data[energy_name]),
        range=((-1.5, 1.5), (-1.5, 1.5)),
        bins=30,
    )

    matrix_norms = migmatrix["Hist"].sum(axis=1)
    for i in range(0, migmatrix["Hist"].shape[0]):
        if matrix_norms[i] > 0:
            migmatrix["Hist"][i, :] /= matrix_norms[i]

    true_energies = valid_data["true_energy"].values
    estimated_energies = valid_data[energy_name].values

    for confidence in (68, 95):
        name = "{:d}%".format(confidence)

        migmatrix[name] = dict()
        migmatrix[name]["upper"] = np.zeros_like(migmatrix["X"])
        migmatrix[name]["mean"] = np.zeros_like(migmatrix["X"])
        migmatrix[name]["lower"] = np.zeros_like(migmatrix["X"])
        migmatrix[name]["rms"] = np.zeros_like(migmatrix["X"])

        for i in range(0, len(migmatrix["X"])):
            true_energies_log = np.lib.scimath.log10(true_energies)
            wh = np.where(
                (true_energies_log >= migmatrix["XEdges"][i])
                & (true_energies_log < migmatrix["XEdges"][i + 1])
            )

            if len(wh[0]) > 0:
                rel_diff_ = estimated_energies[wh] - true_energies[wh]
                rel_diff = rel_diff_ / true_energies[wh]
                quantiles = np.percentile(
                    rel_diff, [50 - confidence / 2.0, 50, 50 + confidence / 2.0]
                )
                migmatrix[name]["upper"][i] = quantiles[2]
                migmatrix[name]["mean"][i] = quantiles[1]
                migmatrix[name]["lower"][i] = quantiles[0]
                migmatrix[name]["rms"][i] = rel_diff.std()
            else:
                migmatrix[name]["upper"][i] = 0
                migmatrix[name]["mean"][i] = 0
                migmatrix[name]["lower"][i] = 0
                migmatrix[name]["rms"][i] = 0

    return migmatrix


def plot_migmatrix(index, name, matrix, grid_shape):
    """Plot migration matrix

    Parameters
    ----------
    index : int
        plot index (different from tel_id)
    name : str
        telescope name (use short names)
    matrix : dict
        migration matrix to be plotted
    grid_shape : tuple
        grid shape
    """
    plt.subplot2grid(grid_shape, (0, index))
    plt.loglog()
    plt.title("%s estimation" % name)
    plt.xlabel("E$_{true}$, TeV")
    plt.ylabel("E$_{est}$, TeV")

    plt.pcolormesh(
        10 ** matrix["XEdges"],
        10 ** matrix["YEdges"],
        matrix["Hist"].transpose(),
        cmap="jet",
        norm=colors.LogNorm(vmin=1e-3, vmax=1),
    )
    plt.colorbar()

    plt.subplot2grid(grid_shape, (1, index))
    plt.semilogx()
    plt.title("%s estimation" % name)
    plt.xlabel("E$_{true}$, TeV")
    plt.ylim(-1, 1)

    plt.plot(
        10 ** matrix["X"],
        matrix["68%"]["mean"],
        linestyle="-",
        color="C0",
        label="Bias",
    )

    plt.plot(
        10 ** matrix["X"], matrix["68%"]["rms"], linestyle=":", color="red", label="RMS"
    )

    plt.plot(
        10 ** matrix["X"],
        matrix["68%"]["upper"],
        linestyle="--",
        color="C1",
        label="68% containment",
    )
    plt.plot(10 ** matrix["X"], matrix["68%"]["lower"], linestyle="--", color="C1")

    plt.plot(
        10 ** matrix["X"],
        matrix["95%"]["upper"],
        linestyle=":",
        color="C2",
        label="95% containment",
    )
    plt.plot(10 ** matrix["X"], matrix["95%"]["lower"], linestyle=":", color="C2")

    plt.grid(linestyle=":")
    plt.legend()
