# coding: utf-8

import datetime
import yaml
import argparse
import pandas as pd

import scipy

import sklearn
import sklearn.ensemble

import ctapipe
from ctapipe.instrument import CameraGeometry
from ctapipe.instrument import TelescopeDescription
from ctapipe.instrument import OpticsDescription
from ctapipe.instrument import SubarrayDescription

from ctapipe.reco.event_processing import EnergyEstimatorPandas, DirectionEstimatorPandas, EventClassifierPandas

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
    config = yaml.load(open(parsed_args.config, "r"))
except IOError:
    print(file_not_found_message.format(parsed_args.config))
    exit()

if 'direction_rf' not in config:
    print('Error: the configuration file is missing the "direction_rf" section. Exiting.')
    exit()
# ------------------------------

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

# Looping over MC / data etc
for data_type in config['data_files']:
    # Using only the "test" sample
    for sample in ['test_sample']:
        shower_data = pd.DataFrame()
        original_mc_data = pd.DataFrame()
        
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
        
        for telescope in config['data_files'][data_type][sample]:
            shower_data.to_hdf(config['data_files'][data_type][sample][telescope]['reco_output'], 
                            key='dl3/reco')
            if data_type == 'mc':
                original_mc_data.to_hdf(config['data_files'][data_type][sample][telescope]['reco_output'], 
                                    key='dl3/original_mc')
