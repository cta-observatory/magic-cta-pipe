# coding: utf-8

import argparse
import time
import pandas as pd
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

from magicctapipe.utils.utils import *
from magicctapipe.utils.filedir import *
from magicctapipe.utils.tels import *
from magicctapipe.utils.plot import *
from magicctapipe.train.utils import *
from magicctapipe.train.event_processing import EnergyEstimatorPandas

PARSER = argparse.ArgumentParser(
    description=("This tools fits the energy random forest regressor on "
                 "the specified events files. For stereo data."),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
PARSER.add_argument('-cfg', '--config_file', type=str, required=True,
                    help='Configuration file to steer the code execution')


def GetHist2D(x, y, bins=30, range=None, weights=None):
    hs, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=range, weights=weights)
    xloc = (xedges[1:] + xedges[:-1]) / 2
    yloc = (yedges[1:] + yedges[:-1]) / 2

    xxloc, yyloc = np.meshgrid(xloc, yloc, indexing='ij')

    hist = {}
    hist['Hist'] = hs
    hist['X'] = xloc
    hist['Y'] = yloc
    hist['XX'] = xxloc
    hist['YY'] = yyloc
    hist['XEdges'] = xedges
    hist['YEdges'] = yedges

    return hist


def evaluate_performance(data, energy_name):
    valid_data = data.dropna(subset=[energy_name])
    migmatrix = GetHist2D(np.lib.scimath.log10(valid_data['true_energy']),
                          np.lib.scimath.log10(valid_data[energy_name]),
                          range=((-1.5, 1.5), (-1.5, 1.5)), bins=30)

    matrix_norms = migmatrix['Hist'].sum(axis=1)
    for i in range(0, migmatrix['Hist'].shape[0]):
        if matrix_norms[i] > 0:
            migmatrix['Hist'][i, :] /= matrix_norms[i]

    true_energies = valid_data['true_energy'].values
    estimated_energies = valid_data[energy_name].values

    for confidence in (68, 95):
        name = '{:d}%'.format(confidence)

        migmatrix[name] = dict()
        migmatrix[name]['upper'] = np.zeros_like(migmatrix['X'])
        migmatrix[name]['mean'] = np.zeros_like(migmatrix['X'])
        migmatrix[name]['lower'] = np.zeros_like(migmatrix['X'])
        migmatrix[name]['rms'] = np.zeros_like(migmatrix['X'])

        for i in range(0, len(migmatrix['X'])):
            true_energies_log = np.lib.scimath.log10(true_energies)
            wh = np.where((true_energies_log >= migmatrix['XEdges'][i]) &
                          (true_energies_log < migmatrix['XEdges'][i + 1]))

            if len(wh[0]) > 0:
                rel_diff = \
                    (estimated_energies[wh] - true_energies[wh]) \
                    / true_energies[wh]
                quantiles = \
                    np.percentile(rel_diff,
                                  [50 - confidence / 2.0,
                                   50,
                                   50 + confidence / 2.0])
                migmatrix[name]['upper'][i] = quantiles[2]
                migmatrix[name]['mean'][i] = quantiles[1]
                migmatrix[name]['lower'][i] = quantiles[0]
                migmatrix[name]['rms'][i] = rel_diff.std()
            else:
                migmatrix[name]['upper'][i] = 0
                migmatrix[name]['mean'][i] = 0
                migmatrix[name]['lower'][i] = 0
                migmatrix[name]['rms'][i] = 0

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
    plt.title('%s estimation' % name)
    plt.xlabel('E$_{true}$, TeV')
    plt.ylabel('E$_{est}$, TeV')

    plt.pcolormesh(10**matrix['XEdges'], 10**matrix['YEdges'],
                   matrix['Hist'].transpose(), cmap='jet',
                   norm=colors.LogNorm(vmin=1e-3, vmax=1))
    plt.colorbar()

    plt.subplot2grid(grid_shape, (1, index))
    plt.semilogx()
    plt.title('%s estimation' % name)
    plt.xlabel('E$_{true}$, TeV')
    plt.ylim(-1, 1)

    plt.plot(10**matrix['X'], matrix['68%']['mean'],
             linestyle='-', color='C0', label='Bias')

    plt.plot(10**matrix['X'], matrix['68%']['rms'],
             linestyle=':', color='red', label='RMS')

    plt.plot(10**matrix['X'], matrix['68%']['upper'],
             linestyle='--', color='C1', label='68% containment')
    plt.plot(10**matrix['X'], matrix['68%']['lower'],
             linestyle='--', color='C1')

    plt.plot(10**matrix['X'], matrix['95%']['upper'],
             linestyle=':', color='C2', label='95% containment')
    plt.plot(10**matrix['X'], matrix['95%']['lower'],
             linestyle=':', color='C2')

    plt.grid(linestyle=':')
    plt.legend()


# =================
# === Main code ===
# =================
def train_energy_rf_stereo(config_file):
    # --- Reading the configuration file ---
    cfg = load_cfg_file_check(config_file=config_file, label='energy_rf')

    # --- Check output directory ---
    check_folder(cfg['classifier_rf']['save_dir'])

    # --- Train sample ---
    f_ = cfg['data_files']['mc']['train_sample']['hillas_h5']
    info_message("Loading train data...", prefix='EnergyRF')
    shower_data_train = load_dl1_data(f_)

    # Computing event weights
    info_message('Computing the train sample event weights...',
                 prefix='EnergyRF')
    sin_edges = np.linspace(0, 1, num=51)
    alt_edges = np.lib.scimath.arcsin(sin_edges)
    intensity_edges = np.logspace(1, 5, num=51)

    mc_weights = get_weights(shower_data_train, alt_edges, intensity_edges)

    shower_data_train = shower_data_train.join(mc_weights)

    # --- Test sample ---
    f_ = cfg['data_files']['mc']['test_sample']['hillas_h5']
    shower_data_test = load_dl1_data(f_)
    tel_ids = get_tel_ids_dl1(shower_data_test)

    # --- Data preparation ---
    l_ = ['obs_id', 'event_id']
    shower_data_train['multiplicity'] = \
        shower_data_train['intensity'].groupby(level=l_).count()
    shower_data_test['multiplicity'] = \
        shower_data_test['intensity'].groupby(level=l_).count()

    # Applying the cuts
    shower_data_train = shower_data_train.query(cfg['energy_rf']['cuts'])
    shower_data_test = shower_data_test.query(cfg['energy_rf']['cuts'])

    # --- Training the direction RF ---
    info_message('Training the RF\n', prefix='EnergyRF')

    energy_estimator = EnergyEstimatorPandas(
        cfg['energy_rf']['features'],
        **cfg['energy_rf']['settings']
    )
    energy_estimator.fit(shower_data_train)
    energy_estimator.save(os.path.join(cfg['energy_rf']['save_dir'],
                                       cfg['energy_rf']['joblib_name']))
    # energy_estimator.load(cfg['energy_rf']['save_name'])

    info_message('Parameter importances', prefix='EnergyRF')
    print('')
    r_ = energy_estimator.telescope_regressors
    for tel_id in r_:
        feature_importances = r_[tel_id].feature_importances_
        print(f'  tel_id: {tel_id}')
        z_ = zip(energy_estimator.feature_names, feature_importances)
        for feature, importance in z_:
            print(f"  {feature:.<15s}: {importance:.4f}")
        print('')

    info_message('Applying RF...', prefix='EnergyRF')
    energy_reco = energy_estimator.predict(shower_data_test)
    shower_data_test = shower_data_test.join(energy_reco)

    # Evaluating performance
    info_message('Evaluating performance...', prefix='EnergyRF')

    idx = pd.IndexSlice

    tel_migmatrix = {}
    for tel_id in tel_ids:
        tel_migmatrix[tel_id] = evaluate_performance(
            shower_data_test.loc[idx[:, :, tel_id],
                                 ['true_energy', 'energy_reco']],
            'energy_reco'
        )

    migmatrix = evaluate_performance(shower_data_test, 'energy_reco_mean')

    # ================
    # === Plotting ===
    # ================
    plt.figure(figsize=tuple(cfg['energy_rf']['fig_size']))

    grid_shape = (2, len(tel_ids)+1)
    # --- PLOT ---
    for index, tel_id in enumerate(tel_ids):
        for i, tel_label in enumerate(cfg['all_tels']['tel_n']):
            if(tel_id in cfg[tel_label]['tel_ids']):
                n_ = '%s%d' % (cfg['all_tels']['tel_n_short'][i],
                               tel_id-cfg[tel_label]['tel_ids'][0]+1)
        plot_migmatrix(index=index, name=n_, matrix=tel_migmatrix[tel_id],
                       grid_shape=grid_shape)
        index += 1
    # --- GLOBAL ---
    plot_migmatrix(index=index, name="All", matrix=migmatrix,
                   grid_shape=grid_shape)

    plt.tight_layout()
    # plt.show()
    save_plt(n=cfg['energy_rf']['fig_name'],
             rdir=cfg['energy_rf']['save_dir'],
             vect='')
    plt.close()


if __name__ == '__main__':
    args = PARSER.parse_args()
    kwargs = args.__dict__
    start_time = time.time()
    train_energy_rf_stereo(
        config_file=kwargs['config_file'],
    )
    print("Execution time: %.2f s" % (time.time() - start_time))
