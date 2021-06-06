# coding: utf-8
import datetime
import yaml
import argparse
import pandas as pd
import numpy as np
import scipy

import sklearn
import sklearn.ensemble

import ctapipe
from ctapipe.instrument import CameraDescription
from ctapipe.instrument import TelescopeDescription
from ctapipe.instrument import OpticsDescription
from ctapipe.instrument import SubarrayDescription

from event_processing import EnergyEstimatorPandas, DirectionEstimatorPandas, EventClassifierPandas
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation,SkyCoord, AltAz
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

def compute_theta2_real(shower_data_test):
    observatory_location = EarthLocation.of_site("Roque de los Muchachos")
    event_times = Time(shower_data_test["mjd"],
                    format='mjd',
                    location=observatory_location)
    alt_az_frame = AltAz(obstime=event_times, location=observatory_location)

    #reconstructed coordinates
    event_coord_reco = SkyCoord(alt=scipy.degrees(shower_data_test['alt_reco_mean']),
                                    az=scipy.degrees(shower_data_test['az_reco_mean']),
                                    frame=alt_az_frame,
                                    unit='deg')

    event_ra_reco = event_coord_reco.fk5.ra
    event_dec_reco = event_coord_reco.fk5.dec
    
    #Crab coordinates
    event_ra_true = ra_dec_source[0]
    event_dec_true = ra_dec_source[1]

    #separation
    c1 = SkyCoord(ra=event_ra_true,dec=event_dec_true,unit='deg')
    c2 = SkyCoord(ra=event_ra_reco.to(u.deg).value,dec=event_dec_reco.to(u.deg).value,unit='deg')
    sep = c1.separation(c2)
    
    #theta2
    theta2=(sep.to(u.deg).value)**2

    return theta2


def compute_theta2_mc(shower_data_test):
    #reconstructed coordinates
    event_coord_reco = SkyCoord(alt=scipy.degrees(shower_data_test['alt_reco_mean']),
                                    az=scipy.degrees(shower_data_test['az_reco_mean']),
                                    frame=AltAz(),
                                    unit='deg')
    #true coordinates
    event_coord_true = SkyCoord(alt=scipy.degrees(shower_data_test['true_alt']),
                                    az=scipy.degrees(shower_data_test['true_az']),
                                    frame=AltAz(),
                                    unit='deg')
    #separation
    sep = event_coord_true.separation(event_coord_reco)
    #theta2
    theta2=(sep.to(u.deg).value)**2

    return theta2


# =================
# === Main code ===
# =================

# --------------------------
# Adding the argument parser
arg_parser = argparse.ArgumentParser(description="""
This tools applies the trained random forests regressor on the "test" event files.
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

if 'direction_rf' not in config:
    print('Error: the configuration file is missing the "direction_rf" section. Exiting.')
    exit()
# ------------------------------

if parsed_args.stereo:
    is_stereo = True
else:
    is_stereo = False

# -----------------
# MAGIC definitions
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
# -----------------

## RF classes to be used for recostruction
#estimator_classes = {
    #'direction_rf': DirectionEstimatorPandas,
    #'energy_rf': EnergyEstimatorPandas,
#}

ra_dec_source=(config['source']['coordinates']['ra_dec'])
print(f"Source coordinates: RA={ra_dec_source[0]} deg; DEC={ra_dec_source[1]} deg")

# Looping over MC / data etc
for data_type in config['data_files']:
    # Using only the "test" sample
    for sample in ['test_sample']:
        shower_data = pd.DataFrame()
        original_mc_data = pd.DataFrame()
        mc_header_data   = pd.DataFrame()

        if is_stereo:

            info_message(f'Loading "{data_type}", sample "{sample}"', prefix='ApplyRF')

            hillas_data = pd.read_hdf(config['data_files'][data_type][sample]['magic']['hillas_output'], key='dl1/hillas_params')
            stereo_data = pd.read_hdf(config['data_files'][data_type][sample]['magic']['hillas_output'], key='dl1/stereo_params')

            if data_type == 'mc':
                orig_mc   = pd.read_hdf(config['data_files'][data_type][sample]['magic']['hillas_output'], key='dl1/original_mc')
                mc_header = pd.read_hdf(config['data_files'][data_type][sample]['magic']['hillas_output'], key='dl1/mc_header')
                dropped_keys = ['tel_alt','tel_az','n_islands', 'tel_id', 'true_alt', 'true_az', 'true_energy', 'true_core_x', 'true_core_y']
            else:
                dropped_keys = ['tel_alt','tel_az','n_islands', 'mjd', 'tel_id']

            stereo_data.drop(dropped_keys, axis=1, inplace=True)
            shower_data = hillas_data.merge(stereo_data, on=['obs_id', 'event_id'])
            if data_type == 'mc':
                original_mc_data = original_mc_data.append(orig_mc)
                mc_header_data   = mc_header_data.append(mc_header)

        else:

            # Reading data of all available telescopes and join them together
            for telescope in config['data_files'][data_type][sample]:

                info_message(f'Loading "{data_type}", sample "{sample}", telescope "{telescope}"',
                    prefix='ApplyRF')

                tel_data = pd.read_hdf(config['data_files'][data_type][sample][telescope]['hillas_output'],
                    key='dl1/hillas_params')

                if data_type == 'mc':
                    orig_mc = pd.read_hdf(config['data_files'][data_type][sample][telescope]['hillas_output'],
                        key='dl1/original_mc')
                    mc_header = pd.read_hdf(config['data_files'][data_type][sample][telescope]['hillas_output'], key='dl1/mc_header')

                shower_data = shower_data.append(tel_data)
                if data_type == 'mc':
                    original_mc_data = original_mc_data.append(orig_mc)
                    mc_header_data   = mc_header_data.append(mc_header)

        # Sorting the data frame for convenience
        shower_data = shower_data.reset_index()
        shower_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
        shower_data.sort_index(inplace=True)

        # Dropping data with the wrong altitude
        shower_data = shower_data.query('tel_alt < 1.5707963267948966')

        if data_type == 'mc':
            original_mc_data = original_mc_data.reset_index()
            original_mc_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
            original_mc_data.sort_index(inplace=True)

        # Computing the event "multiplicity"
        shower_data['multiplicity'] = shower_data['intensity'].groupby(level=['obs_id', 'event_id']).count()
        if data_type == 'mc':
            original_mc_data['multiplicity'] = original_mc_data['true_energy'].groupby(level=['obs_id', 'event_id']).count()

        #Added by Lea Heckmann 2020-05-15 for the moment to delete duplicate events
        info_message(f'Removing duplicate events', prefix='ApplyRF')
        shower_data = shower_data[~shower_data.index.duplicated()]

        # Applying RFs of every kind
        for rf_kind in ['direction_rf', 'energy_rf', 'classifier_rf']:
            info_message(f'Loading RF: {rf_kind}', prefix='ApplyRF')

            if rf_kind == 'direction_rf':
                estimator = DirectionEstimatorPandas(config[rf_kind]['features'],
                                                     magic_tel_descriptions,
                                                     **config[rf_kind]['settings'])
            elif rf_kind == 'energy_rf':
                estimator = EnergyEstimatorPandas(config[rf_kind]['features'],
                                                  **config[rf_kind]['settings'])

            elif rf_kind == 'classifier_rf':
                estimator = EventClassifierPandas(config[rf_kind]['features'],
                                                  **config[rf_kind]['settings'])

            estimator.load(config[rf_kind]['save_name'])

            # --- Applying RF ---
            info_message(f'Applying RF: {rf_kind}', prefix='ApplyRF')
            reco = estimator.predict(shower_data)

            # Appeding the result to the main data frame
            shower_data = shower_data.join(reco)

        # Storing the reconstructed values for the given data sample
        info_message('Saving the reconstructed data', prefix='ApplyRF')

        if is_stereo:
            if data_type == 'mc':
                shower_data['theta2']=compute_theta2_mc(shower_data)
                original_mc_data.to_hdf(config['data_files'][data_type][sample]['magic']['reco_output'],key='dl3/original_mc')
                mc_header_data.to_hdf(config['data_files'][data_type][sample]['magic']['reco_output'],key='dl3/mc_header')
            else:
                shower_data['theta2']=compute_theta2_real(shower_data)
            shower_data.to_hdf(config['data_files'][data_type][sample]['magic']['reco_output'],key='dl3/reco')

        else:
            for telescope in config['data_files'][data_type][sample]:
                shower_data.to_hdf(config['data_files'][data_type][sample][telescope]['reco_output'],key='dl3/reco')
                if data_type == 'mc':
                    original_mc_data.to_hdf(config['data_files'][data_type][sample][telescope]['reco_output'],key='dl3/original_mc')
                    mc_header_data.to_hdf(config['data_files'][data_type][sample][telescope]['reco_output'],key='dl3/mc_header')
