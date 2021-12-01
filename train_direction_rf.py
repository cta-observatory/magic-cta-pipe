# coding: utf-8

import datetime
import yaml
import argparse
import pandas as pd

import numpy as np

import sklearn
import sklearn.ensemble

import ctapipe
from ctapipe.instrument import CameraDescription
from ctapipe.instrument import TelescopeDescription
from ctapipe.instrument import OpticsDescription
from ctapipe.instrument import SubarrayDescription

from event_processing import DirectionEstimatorPandas

from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz
from astropy.coordinates.angle_utilities import angular_separation, position_angle

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


def compute_separation_angle(shower_data):
    separation = dict()

    for tel_id in [1, 2]:
        event_coord_true = SkyCoord(shower_data_test.loc[(slice(None), slice(None), tel_id), 'true_az'].values * u.rad,
                                    shower_data_test.loc[(slice(None), slice(None), tel_id), 'true_alt'].values * u.rad,
                                    frame=AltAz())

        event_coord_reco = SkyCoord(shower_data_test.loc[(slice(None), slice(None), tel_id), 'az_reco'].values * u.rad,
                                    shower_data_test.loc[(slice(None), slice(None), tel_id), 'alt_reco'].values * u.rad,
                                    frame=AltAz())

        separation[tel_id] = event_coord_true.separation(event_coord_reco)

    event_coord_true = SkyCoord(shower_data_test['true_az'].values * u.rad,
                            shower_data_test['true_alt'].values * u.rad,
                            frame=AltAz())

    event_coord_reco = SkyCoord(shower_data_test['az_reco_mean'].values * u.rad,
                                shower_data_test['alt_reco_mean'].values * u.rad,
                                frame=AltAz())

    separation[0] = event_coord_true.separation(event_coord_reco)

    # Converting to a data frame
    separation_df = pd.DataFrame(data={'sep_0': separation[0]}, index=shower_data_test.index)
    for tel_id in separation_df.index.levels[2]:
        df = pd.DataFrame(data={f'sep_{tel_id:d}': separation[tel_id]},
                        index=shower_data_test.loc[(slice(None), slice(None), tel_id), 'true_az'].index)
        separation_df = separation_df.join(df)

    separation_df = separation_df.join(shower_data)

    for tel_id in [0, 1, 2]:
        print(f"  Tel {tel_id} scatter: {separation[tel_id].to(u.deg).std():.2f}")

    return separation_df


def load_data_sample(sample):
    """
    This function loads Hillas data from the corresponding sample, merging the data
    from the different telescopes found.

    Parameters
    ----------
    sample: str
        Sample from which data will be read

    Returns
    -------
    pandas.Dataframe
        Pandas dataframe with data from all the telescopes in sample
    """

    shower_data = pd.DataFrame()

    for telescope in sample:
        info_message(f'Loading {telescope} data...', prefix='DirectionRF')

        hillas_data = pd.read_hdf(sample[telescope]['hillas_output'], key='dl1/hillas_params')
        hillas_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)

        shower_data = shower_data.append(hillas_data)

    shower_data.sort_index(inplace=True)

    return shower_data


def load_data_sample_stereo(input_file, is_mc):
    """
    This function loads Hillas and stereo data from input_file.

    Parameters
    ----------
    input_file: str
        Input HDF5 file
    is_mc: bool
        Flag to denote if data is MC or real

    Returns
    -------
    pandas.Dataframe
        Pandas dataframe with Hillas and stereo data
    """

    shower_data = pd.DataFrame()

    hillas_data = pd.read_hdf(input_file, key='dl1/hillas_params')
    stereo_data = pd.read_hdf(input_file, key='dl1/stereo_params')

    if is_mc:
        dropped_keys = ['tel_alt','tel_az','n_islands', 'tel_id', 'true_alt', 'true_az', 'true_energy', 'true_core_x', 'true_core_y']
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
This tools fits the direction random forest regressor on the specified events files.
""")

arg_parser.add_argument("--config", default="config.yaml",
                        help='Configuration file to steer the code execution.')
arg_parser.add_argument("--stereo",
                        help='Use stereo DL1 files.',
                        action='store_true')

parsed_args = arg_parser.parse_args()
# ------------------------------
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

if 'direction_rf' not in config:
    print('Error: the configuration file is missing the "direction_rf" section. Exiting.')
    exit()
# ------------------------------

if parsed_args.stereo:
    is_stereo = True
else:
    is_stereo = False

# MAGIC telescope positions in m wrt. to the center of CTA simulations
magic_tel_positions = {
    1: [-27.24, -146.66, 50.00] * u.m,
    2: [-96.44, -96.77, 51.00] * u.m
}

# MAGIC telescope description
magic_optics = OpticsDescription.from_name('MAGIC')
magic_cam = CameraDescription.from_name('MAGICCam')
magic_tel_description = TelescopeDescription(name='MAGIC',
                                             tel_type='MAGIC',
                                             optics=magic_optics,
                                             camera=magic_cam)
magic_tel_descriptions = {1: magic_tel_description,
                          2: magic_tel_description}
# MAGIC sub-array
magic_subarray = SubarrayDescription('MAGIC',
                                     magic_tel_positions,
                                     magic_tel_descriptions)

# --- Train sample ---
info_message('Loading MC train data...', prefix='DirectionRF')
if is_stereo:
    shower_data_train = load_data_sample_stereo(config['data_files']['mc']['train_sample']['magic']['hillas_output'], True)
else:
    shower_data_train = load_data_sample(config['data_files']['mc']['train_sample'])

# Computing event weights
info_message('Computing the train sample event weights...', prefix='DirectionRF')
sin_edges = np.linspace(0, 1, num=51)
alt_edges = np.lib.scimath.arcsin(sin_edges)
intensity_edges = np.logspace(1, 5, num=51)

mc_weights = get_weights(shower_data_train, alt_edges, intensity_edges)

shower_data_train = shower_data_train.join(mc_weights)

# --- Test sample ---
info_message('Loading MC test data...', prefix='DirectionRF')
if is_stereo:
    shower_data_test = load_data_sample_stereo(config['data_files']['mc']['test_sample']['magic']['hillas_output'], True)
else:
    shower_data_test = load_data_sample(config['data_files']['mc']['test_sample'])

# --- Data preparation ---

shower_data_train['multiplicity'] = shower_data_train['intensity'].groupby(level=['obs_id', 'event_id']).count()
shower_data_test['multiplicity'] = shower_data_test['intensity'].groupby(level=['obs_id', 'event_id']).count()

# Applying the cuts
shower_data_train = shower_data_train.query(config['direction_rf']['cuts'])
shower_data_test = shower_data_test.query(config['direction_rf']['cuts'])

# --- Training the direction RF ---
info_message('Training the RF\n', prefix='DirectionRF')

direction_estimator = DirectionEstimatorPandas(config['direction_rf']['features'],
                                               magic_tel_descriptions,
                                               **config['direction_rf']['settings'])
direction_estimator.fit(shower_data_train)
direction_estimator.save(config['direction_rf']['save_name'])
#direction_estimator.load(config['direction_rf']['save_name'])

# Printing the parameter "importances"
for kind in direction_estimator.telescope_rfs:
    for tel_id in direction_estimator.telescope_rfs[kind]:
        feature_importances = direction_estimator.telescope_rfs[kind][tel_id].feature_importances_

        print(f'  Kind: {kind}, tel_id: {tel_id}')
        for feature, importance in zip(config['direction_rf']['features'][kind], feature_importances):
            print(f"  {feature:.<15s}: {importance:.4f}")
        print('')

# --- Applying RF to the "test" sample ---
info_message('Applying RF to the "test" sample', prefix='DirectionRF')
coords_reco = direction_estimator.predict(shower_data_test)
shower_data_test = shower_data_test.join(coords_reco)

# --- Evaluating the performance ---
info_message('Evaluating the performance', prefix='DirectionRF')
separation_df = compute_separation_angle(shower_data_test)

# Energy-dependent resolution
info_message('Estimating the energy-dependent resolution', prefix='DirectionRF')
energy_edges = np.logspace(-1, 1.3, num=20)
energy = (energy_edges[1:] * energy_edges[:-1])**0.5

energy_psf = dict()
for i in range(3):
    energy_psf[i] = np.zeros_like(energy)

for ei in range(len(energy_edges) - 1):
    cuts = f'(true_energy>= {energy_edges[ei]:.2e}) & (true_energy < {energy_edges[ei+1]:.2e})'
    #cuts += ' & (intensity > 100)'
    #cuts += ' & (length > 0.05)'
    cuts += ' & (multiplicity > 1)'
    query = separation_df.query(cuts)

    for pi in range(3):
        if pi > 0:
            tel_id = pi
        else:
            tel_id = 1
        selection = query.loc[(slice(None), slice(None), tel_id), f'sep_{pi}'].dropna()
        energy_psf[pi][ei] = np.percentile(selection, 68)

# Offset-dependent resolution
info_message('Estimating the offset-dependent resolution', prefix='DirectionRF')
offset = angular_separation(separation_df['tel_az'], separation_df['tel_alt'],
                            separation_df['true_az'], separation_df['true_alt'])

separation_df['offset'] = np.degrees(offset)

offset_edges = np.linspace(0, 1.3, num=10)
offset = (offset_edges[1:] * offset_edges[:-1])**0.5

offset_psf = dict()
for i in range(3):
    offset_psf[i] = np.zeros_like(offset)

for oi in range(len(offset_edges) - 1):
    cuts = f'(offset >= {offset_edges[oi]:.2f}) & (offset < {offset_edges[oi+1]:.2f})'
    #cuts += ' & (intensity > 100)'
    #cuts += ' & (length > 0.05)'
    cuts += ' & (multiplicity > 1)'
    query = separation_df.query(cuts)

    for pi in range(3):
        if pi > 0:
            tel_id = pi
        else:
            tel_id = 1
        selection = query.loc[(slice(None), slice(None), tel_id), [f'sep_{pi}']].dropna()
        offset_psf[pi][oi] = np.percentile(selection[f'sep_{pi}'], 68)

# ================
# === Plotting ===
# ================

pyplot.figure(figsize=(12, 12))
#pyplot.style.use('presentation')

pyplot.xlabel(r'$\theta^2$, deg$^2$')

for tel_id in [0, 1, 2]:
    pyplot.subplot2grid((3, 2), (tel_id, 0))
    pyplot.title(f'Tel {tel_id}')
    pyplot.xlabel(r'$\theta^2$, deg$^2$')
    #pyplot.semilogy()
    pyplot.hist(separation_df[f'sep_{tel_id}']**2, bins=100, range=(0, 0.5), density=True, alpha=0.1, color='C0');
    pyplot.hist(separation_df[f'sep_{tel_id}']**2, bins=100, range=(0, 0.5), density=True, histtype='step', color='C0');
    pyplot.grid(linestyle=':')

    pyplot.subplot2grid((3, 2), (tel_id, 1))
    pyplot.xlabel(r'$\theta$, deg')
    pyplot.xlim(0, 2.0)
    pyplot.hist(separation_df[f'sep_{tel_id}'], bins=400, range=(0, 5), cumulative=True, density=True, alpha=0.1, color='C0');
    pyplot.hist(separation_df[f'sep_{tel_id}'], bins=400, range=(0, 5), cumulative=True, density=True, histtype='step', color='C0');
    pyplot.grid(linestyle=':')

pyplot.tight_layout()
pyplot.savefig('Direction_RF_theta2.png')
pyplot.close()

pyplot.clf()
pyplot.semilogx()
pyplot.xlabel('Energy [TeV]')
pyplot.ylabel(r'$\sigma_{68}$ [deg]')
pyplot.ylim(0, 1.0)
pyplot.plot(energy, energy_psf[0], linewidth=4, label='Total')
pyplot.plot(energy, energy_psf[1], label='M1')
pyplot.plot(energy, energy_psf[2], label='M2')
pyplot.grid(linestyle=':')
pyplot.legend()
pyplot.savefig('Direction_RF_PSF_energy.png')
pyplot.close()

pyplot.clf()
pyplot.xlabel('Offset [deg]')
pyplot.ylabel(r'$\sigma_{68}$ [deg]')
pyplot.ylim(0, 0.5)
pyplot.plot(offset, offset_psf[0], linewidth=4, label='Total')
pyplot.plot(offset, offset_psf[1], label='M1')
pyplot.plot(offset, offset_psf[2], label='M2')
pyplot.grid(linestyle=':')
pyplot.legend()
pyplot.savefig('Direction_RF_PSF_offset.png')
pyplot.close()
