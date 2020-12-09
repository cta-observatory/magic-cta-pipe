# coding: utf-8

import argparse
import yaml
import datetime

import pandas as pd
import numpy as np

import sklearn
import sklearn.ensemble

import ctapipe
from ctapipe.instrument import CameraGeometry
from ctapipe.instrument import TelescopeDescription
from ctapipe.instrument import OpticsDescription
from ctapipe.instrument import SubarrayDescription

from ctapipe.reco import HillasReconstructor
from event_processing import EnergyEstimatorPandas

from astropy import units as u

from matplotlib import pyplot, colors


def info_message(text, prefix='info'):
    """
    This function prints the specified text with the prefix of the current date

    Parameters
    ----------
    text: str

    Returns
    -------
    None

    """

    date_str = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"({prefix:s}) {date_str:s}: {text:s}")


def GetHist2D(x,y, bins=30, range=None, weights=None):
    hs, xedges, yedges = np.histogram2d(x,y, bins=bins, range=range, weights=weights)
    xloc = (xedges[1:] + xedges[:-1]) / 2
    yloc = (yedges[1:] + yedges[:-1]) / 2

    xxloc, yyloc = np.meshgrid( xloc, yloc, indexing='ij' )

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
            wh = np.where((np.lib.scimath.log10(true_energies) >= migmatrix['XEdges'][i]) &
                             (np.lib.scimath.log10(true_energies) < migmatrix['XEdges'][i + 1]))

            if len(wh[0]) > 0:
                rel_diff = (estimated_energies[wh] - true_energies[wh]) / true_energies[wh]
                quantiles = np.percentile(rel_diff, [50 - confidence / 2.0, 50, 50 + confidence / 2.0])

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


def load_data_sample(sample):
    shower_data = pd.DataFrame()

    for telescope in sample:
        info_message(f'Loading {telescope} data...', prefix='ClassifierRF')

        hillas_data = pd.read_hdf(sample[telescope]['hillas_output'], key='dl1/hillas_params')
        hillas_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

        shower_data = shower_data.append(hillas_data)

    shower_data.sort_index(inplace=True)

    return shower_data


def load_data_sample_stereo(input_file, is_mc):
    shower_data = pd.DataFrame()

    hillas_data = pd.read_hdf(input_file, key='dl1/hillas_params')
    stereo_data = pd.read_hdf(input_file, key='dl1/stereo_params')

    if ismc:
        dropped_keys = ['tel_alt','tel_az','n_islands', 'tel_id', 'true_alt', 'true_az', 'true_energy']
    else:
        dropped_keys = ['tel_alt','tel_az','n_islands', 'mjd', 'tel_id']

    stereo_data.drop(dropped_keys, axis=1, inplace=True)

    shower_data = hillas_data.merge(stereo_data, on=['obs_id', 'event_id'])
    shower_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    shower_data.sort_index(inplace=True)

    return shower_data

def get_weights(mc_data, alt_edges, intensity_edges):
    mc_hist, _, _ = np.histogram2d(mc_data['tel_alt'],
                                      mc_data['intensity'],
                                      bins=[alt_edges, intensity_edges])

    availability_hist = np.clip(mc_hist, 0, 1)

    # --- MC weights ---
    mc_alt_bins = np.digitize(mc_data['tel_alt'], alt_edges) - 1
    mc_intensity_bins = np.digitize(mc_data['intensity'], intensity_edges) - 1

    # Treating the out-of-range events
    mc_alt_bins[mc_alt_bins == len(alt_edges) - 1] = len(alt_edges) - 2
    mc_intensity_bins[mc_intensity_bins == len(intensity_edges) - 1] = len(intensity_edges) - 2

    mc_weights = 1 / mc_hist[mc_alt_bins, mc_intensity_bins]
    mc_weights *= availability_hist[mc_alt_bins, mc_intensity_bins]

    # --- Storing to a data frame ---
    mc_weight_df = pd.DataFrame(data={'event_weight': mc_weights},
                                index=mc_data.index)

    return mc_weight_df


# =================
# === Main code ===
# =================

# --------------------------
# Adding the argument parser
arg_parser = argparse.ArgumentParser(description="""
This tools fits the energy random forest regressor on the specified events files.
""")

arg_parser.add_argument("--config", default="config.yaml",
                        help='Configuration file to steer the code execution.')
arg_parser.add_argument("--stereo",
                        help='Use stereo DL1 files.',
                        action='store_true')

parsed_args = arg_parser.parse_args()
# --------------------------

# ------------------------------
# Reading the configuration file

file_not_found_message = """
Error: can not load the configuration file {:s}.
Please check that the file exists and is of YAML or JSON format.
Exiting.
"""

try:
    config = yaml.safe_load(open(parsed_args.config, "r"))
except IOError:
    print(file_not_found_message.format(parsed_args.config))
    exit()

if 'energy_rf' not in config:
    print('Error: the configuration file is missing the "energy_rf" section. Exiting.')
    exit()
# ------------------------------

if parsed_args.stereo:
    is_stereo = True
else:
    is_stereo = False

# --- Train sample ---
info_message('Loading MC train data...', prefix='ClassifierRF')
if is_stereo:
    shower_data_train = load_data_sample_stereo(config['data_files']['mc']['train_sample']['magic']['hillas_output'], True)
else:
    shower_data_train = load_data_sample(config['data_files']['mc']['train_sample'])

# Computing event weights
info_message('Computing the train sample event weights...', prefix='DirRF')
sin_edges = np.linspace(0, 1, num=51)
alt_edges = np.lib.scimath.arcsin(sin_edges)
intensity_edges = np.logspace(1, 5, num=51)

mc_weights = get_weights(shower_data_train, alt_edges, intensity_edges)

shower_data_train = shower_data_train.join(mc_weights)

# --- Test sample ---
info_message('Loading MC test data...', prefix='ClassifierRF')
if is_stereo:
    shower_data_test = load_data_sample_stereo(config['data_files']['mc']['test_sample']['magic']['hillas_output'], True)
else:
    shower_data_test = load_data_sample(config['data_files']['mc']['test_sample'])

info_message('Preprocessing...', prefix='EnergyRF')

# --- Data preparation ---
shower_data_train['multiplicity'] = shower_data_train['intensity'].groupby(level=['obs_id', 'event_id']).count()
shower_data_test['multiplicity'] = shower_data_test['intensity'].groupby(level=['obs_id', 'event_id']).count()

# Applying the cuts
shower_data_train = shower_data_train.query(config['energy_rf']['cuts'])
shower_data_test = shower_data_test.query(config['energy_rf']['cuts'])

# --- Training the direction RF ---
info_message('Training RF...', prefix='EnergyRF')

energy_estimator = EnergyEstimatorPandas(config['energy_rf']['features'],
                                         **config['energy_rf']['settings'])
energy_estimator.fit(shower_data_train)
energy_estimator.save(config['energy_rf']['save_name'])
#energy_estimator.load(config['energy_rf']['save_name'])

info_message('Parameter importances', prefix='EnergyRF')
print('')
for tel_id in energy_estimator.telescope_regressors:
    feature_importances = energy_estimator.telescope_regressors[tel_id].feature_importances_

    print(f'  tel_id: {tel_id}')
    for feature, importance in zip(energy_estimator.feature_names, feature_importances):
        print(f"  {feature:.<15s}: {importance:.4f}")
    print('')

info_message('Applying RF...', prefix='EnergyRF')
energy_reco = energy_estimator.predict(shower_data_test)
shower_data_test = shower_data_test.join(energy_reco)

# Evaluating performance
info_message('Evaluating performance...', prefix='EnergyRF')

idx = pd.IndexSlice

m1_migmatrix = evaluate_performance(shower_data_test.loc[idx[:, :, 1], ['true_energy', 'energy_reco']],
                                    'energy_reco')
m2_migmatrix = evaluate_performance(shower_data_test.loc[idx[:, :, 2], ['true_energy', 'energy_reco']],
                                    'energy_reco')

migmatrix = evaluate_performance(shower_data_test, 'energy_reco_mean')


# ================
# === Plotting ===
# ================

#pyplot.style.use('presentation')

pyplot.figure(figsize=(12, 6))

grid_shape = (2, 3)

pyplot.subplot2grid(grid_shape, (0, 0))
pyplot.loglog()
pyplot.title('M1 estimation')
pyplot.xlabel('E$_{true}$, TeV')
pyplot.ylabel('E$_{est}$, TeV')

pyplot.pcolormesh(10**m1_migmatrix['XEdges'], 10**m1_migmatrix['YEdges'], m1_migmatrix['Hist'].transpose(),
                  cmap='jet', norm=colors.LogNorm(vmin=1e-3, vmax=1))
pyplot.colorbar()

pyplot.subplot2grid(grid_shape, (1, 0))
pyplot.semilogx()
pyplot.title('M1 estimation')
pyplot.xlabel('E$_{true}$, TeV')
pyplot.ylim(-1, 1)

pyplot.plot(10**m1_migmatrix['X'], m1_migmatrix['68%']['mean'],
            linestyle='-', color='C0', label='Bias')

pyplot.plot(10**m1_migmatrix['X'], m1_migmatrix['68%']['rms'],
            linestyle=':', color='red', label='RMS')

pyplot.plot(10**m1_migmatrix['X'], m1_migmatrix['68%']['upper'],
            linestyle='--', color='C1', label='68% containment')
pyplot.plot(10**m1_migmatrix['X'], m1_migmatrix['68%']['lower'],
            linestyle='--', color='C1')

pyplot.plot(10**m1_migmatrix['X'], m1_migmatrix['95%']['upper'],
            linestyle=':', color='C2', label='95% containment')
pyplot.plot(10**m1_migmatrix['X'], m1_migmatrix['95%']['lower'],
            linestyle=':', color='C2')

pyplot.grid(linestyle=':')
pyplot.legend()

pyplot.subplot2grid(grid_shape, (0, 1))
pyplot.loglog()
pyplot.title('M2 estimation')
pyplot.xlabel('E$_{true}$, TeV')
pyplot.ylabel('E$_{est}$, TeV')

pyplot.pcolormesh(10**m2_migmatrix['XEdges'], 10**m2_migmatrix['YEdges'], m2_migmatrix['Hist'].transpose(),
                  cmap='jet', norm=colors.LogNorm(vmin=1e-3, vmax=1))
pyplot.colorbar()

pyplot.subplot2grid(grid_shape, (1, 1))
pyplot.semilogx()
pyplot.title('M2 estimation')
pyplot.xlabel('E$_{true}$, TeV')
pyplot.ylim(-1, 1)

pyplot.plot(10**m2_migmatrix['X'], m2_migmatrix['68%']['mean'],
            linestyle='-', color='C0', label='Bias')

pyplot.plot(10**m2_migmatrix['X'], m2_migmatrix['68%']['rms'],
            linestyle=':', color='red', label='RMS')

pyplot.plot(10**m2_migmatrix['X'], m2_migmatrix['68%']['upper'],
            linestyle='--', color='C1', label='68% containment')
pyplot.plot(10**m2_migmatrix['X'], m2_migmatrix['68%']['lower'],
            linestyle='--', color='C1')

pyplot.plot(10**m2_migmatrix['X'], m2_migmatrix['95%']['upper'],
            linestyle=':', color='C2', label='95% containment')
pyplot.plot(10**m2_migmatrix['X'], m2_migmatrix['95%']['lower'],
            linestyle=':', color='C2')

pyplot.grid(linestyle=':')
pyplot.legend()

pyplot.subplot2grid(grid_shape, (0, 2))
pyplot.title('M1+M2 estimation')
pyplot.loglog()
pyplot.xlabel('E$_{true}$, TeV')
pyplot.ylabel('E$_{est}$, TeV')

pyplot.pcolormesh(10**migmatrix['XEdges'], 10**migmatrix['YEdges'], migmatrix['Hist'].transpose(),
                  cmap='jet', norm=colors.LogNorm(vmin=1e-3, vmax=1))
pyplot.colorbar()

pyplot.subplot2grid(grid_shape, (1, 2))
pyplot.semilogx()
pyplot.title('M1+M2 estimation')
pyplot.xlabel('E$_{true}$, TeV')
pyplot.ylim(-1, 1)

pyplot.plot(10**migmatrix['X'], migmatrix['68%']['mean'],
            linestyle='-', color='C0', label='Bias')

pyplot.plot(10**migmatrix['X'], migmatrix['68%']['rms'],
            linestyle=':', color='red', label='RMS')

pyplot.plot(10**migmatrix['X'], migmatrix['68%']['upper'],
            linestyle='--', color='C1', label='68% containment')
pyplot.plot(10**migmatrix['X'], migmatrix['68%']['lower'],
            linestyle='--', color='C1')

pyplot.plot(10**migmatrix['X'], migmatrix['95%']['upper'],
            linestyle=':', color='C2', label='95% containment')
pyplot.plot(10**migmatrix['X'], migmatrix['95%']['lower'],
            linestyle=':', color='C2')

pyplot.grid(linestyle=':')
pyplot.legend()

pyplot.tight_layout()

#pyplot.show()
pyplot.savefig('Energy_RF_migmatrix.png')
pyplot.close()
