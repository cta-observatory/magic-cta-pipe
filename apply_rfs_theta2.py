# coding: utf-8
from astropy.time import Time
import datetime
import yaml
import argparse
import pandas as pd
import numpy as np
import scipy

import sklearn
import sklearn.ensemble

import ctapipe
from ctapipe.instrument import CameraGeometry
from ctapipe.instrument import TelescopeDescription
from ctapipe.instrument import OpticsDescription
from ctapipe.instrument import SubarrayDescription

from event_processing import EnergyEstimatorPandas, DirectionEstimatorPandas, EventClassifierPandas
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
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

def compute_separation_angle_real(shower_data_test):
    print('-----------REAL-----------------')
    separation = dict()

    observatory_location = EarthLocation.of_site("Roque de los Muchachos")

    event_times = Time(shower_data_test["mjd"],
                    format='mjd',
                    location=observatory_location)

    alt_az_frame = AltAz(obstime=event_times, location=observatory_location)


    #coordinate ricostruite
    
    event_coord_reco = SkyCoord(alt=scipy.degrees(shower_data_test['alt_reco_mean']),
                                    az=scipy.degrees(shower_data_test['az_reco_mean']),
                                    frame=alt_az_frame,
                                    unit='deg')

    event_ra_reco = event_coord_reco.fk5.ra
    event_dec_reco = event_coord_reco.fk5.dec
    
    event_ra_true = ra_dec_Crab[0]
    event_dec_true = ra_dec_Crab[1]

    c1 = SkyCoord(ra=event_ra_true,dec=event_dec_true,unit='deg')
    c2 = SkyCoord(ra=event_ra_reco.to(u.deg).value,dec=event_dec_reco.to(u.deg).value,unit='deg')
    sep = c1.separation(c2)

    return sep.to(u.deg).value


def compute_separation_angle_mc(shower_data_test):
    print('-----------MONTECARLO-----------')
    #coordinate ricostruite                                                                                      
    event_coord_reco = SkyCoord(alt=scipy.degrees(shower_data_test['alt_reco_mean']),
                                    az=scipy.degrees(shower_data_test['az_reco_mean']),
                                    frame=AltAz(),
                                    unit='deg')
    #coordinate "vere"                                                                                                
    event_coord_true = SkyCoord(alt=scipy.degrees(shower_data_test['true_alt']),
                                    az=scipy.degrees(shower_data_test['true_az']),
                                    frame=AltAz(),
                                    unit='deg')

    sep = event_coord_true.separation(event_coord_reco)
    return sep.to(u.deg).value


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
magic_cam = CameraGeometry.from_name('MAGICCam')
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

# Looping over MC / data etc (SIA MC CHE REAL)
for data_type in config['data_files']:
    #if(data_type=='mc'):
    #    continue
    # Using only the "test" sample
    for sample in ['test_sample']:
        shower_data = pd.DataFrame()
        original_mc_data = pd.DataFrame()

        if is_stereo:

            info_message(f'Loading "{data_type}", sample "{sample}"', prefix='ApplyRF')

            #prendo i parametri hillas/stero
            hillas_data = pd.read_hdf(config['data_files'][data_type][sample]['magic']['hillas_output'], key='dl1/hillas_params')
            stereo_data = pd.read_hdf(config['data_files'][data_type][sample]['magic']['hillas_output'], key='dl1/stereo_params')

            ra_dec_Crab=(config['coordinates']['ra_dec'])
            print(ra_dec_Crab)
            
            if data_type == 'mc':
                #se è MC 
                orig_mc = pd.read_hdf(config['data_files'][data_type][sample]['magic']['hillas_output'], key='dl1/original_mc')
                dropped_keys = ['tel_alt','tel_az','n_islands', 'tel_id', 'true_alt', 'true_az', 'true_energy', 'true_core_x', 'true_core_y']
            else:
                #se è real
                dropped_keys = ['tel_alt','tel_az','n_islands', 'mjd', 'tel_id']

            #stereo data = parametri stereo
            stereo_data.drop(dropped_keys, axis=1, inplace=True)
            
            #hillas data+stereo data
            shower_data = hillas_data.merge(stereo_data, on=['obs_id', 'event_id'])
            #shower_data=shower_data[:10] #qui seleziono alcuni-------------------------------
            if data_type == 'mc':
                original_mc_data = original_mc_data.append(orig_mc)
                #original_mc_data=original_mc_data[:10]   #QUI SELEZIONO ALCUNI---------------------------

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

                shower_data = shower_data.append(tel_data)
                if data_type == 'mc':
                    original_mc_data = original_mc_data.append(orig_mc)


        
        # Sorting the data frame for convenience
        shower_data = shower_data.reset_index()
        shower_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
        shower_data.sort_index(inplace=True)
        #shower_data=shower_data[:10]

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

        """if is_stereo:
            if data_type == 'mc':
                separation_df=compute_separation_angle(shower_data)
                #print(np.degrees(separation_df))
            else:
                separation_df=compute_separation_angle_(shower_data)
                #print(np.degrees(separation_df))"""
        
        if is_stereo:
            #shower_data['theta']=compute_separation_angle(shower_data) 
            #shower_data.to_hdf(config['data_files'][data_type][sample]['magic']['reco_output'],key='dl3/reco')
            if data_type == 'mc':
                shower_data['theta']=compute_separation_angle_mc(shower_data)
                original_mc_data.to_hdf(config['data_files'][data_type][sample]['magic']['reco_output'],key='dl3/original_mc')
            else:
                shower_data['theta']=compute_separation_angle_real(shower_data)
                print(shower_data['theta'])
            shower_data.to_hdf(config['data_files'][data_type][sample]['magic']['reco_output'],key='dl3/reco') 
            #print(shower_data.keys())
            #print(shower_data['dl3/original_mc'].keys())
            #print(shower_data.columns)
            #print(shower_data['theta'])
        else:
            for telescope in config['data_files'][data_type][sample]:
                #shower_data['theta']=compute_separation_angle(shower_data)
                shower_data.to_hdf(config['data_files'][data_type][sample][telescope]['reco_output'],key='dl3/reco')
                if data_type == 'mc':
                    original_mc_data.to_hdf(config['data_files'][data_type][sample][telescope]['reco_output'],key='dl3/original_mc')
