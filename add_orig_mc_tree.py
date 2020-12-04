#!/usr/bin/env python

import glob
import re
import argparse
import yaml
import datetime

import uproot
import pandas as pd
import numpy as np

from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.coordinates.angle_utilities import angular_separation

import ctapipe
from ctapipe.instrument import CameraGeometry
from ctapipe.instrument import TelescopeDescription
from ctapipe.instrument import OpticsDescription
from ctapipe.instrument import SubarrayDescription
from ctapipe.coordinates import CameraFrame, TelescopeFrame


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


def read_original_mc_tree(file_mask):
    shower_data = pd.DataFrame()

    file_list = glob.glob(file_mask)

    for file_name in file_list:
        telelscope = int(re.findall(r'.*_M(\d)_.*', file_name)[0])
        run_number = int(re.findall(r'.*_M\d_za\d+to\d+_\d_(\d+)_Y_.*', file_name)[0])
        
        with uproot.open(file_name) as input_data:       
            true_energy = input_data['OriginalMC']['MMcEvtBasic.fEnergy'].array()
            tel_az = input_data['OriginalMC']['MMcEvtBasic.fTelescopePhi'].array()
            tel_zd = input_data['OriginalMC']['MMcEvtBasic.fTelescopeTheta'].array()
            
            true_energy /= 1e3  # GeV -> TeV
            tel_alt = np.pi/2 - tel_zd
            
            # # Transformation from Monte Carlo to usual azimuth
            # tel_az = -1 * (tel_az - np.pi + np.radians(7))

            cam_x = input_data['OriginalMC']['MSrcPosCam.fX'].array()
            cam_y = input_data['OriginalMC']['MSrcPosCam.fY'].array()

            tel_pointing = AltAz(alt=tel_alt * u.rad,
                                az=tel_az * u.rad)

            optics = magic_tel_descriptions[telelscope].optics
            camera = magic_tel_descriptions[telelscope].camera
        
            camera_frame = CameraFrame(focal_length=optics.equivalent_focal_length,
                                    rotation=camera.cam_rotation)

            telescope_frame = TelescopeFrame(telescope_pointing=tel_pointing)

            camera_coord = SkyCoord(-cam_y * u.mm,
                                    cam_x * u.mm,
                                    frame=camera_frame)
            shower_coord_in_telescope = camera_coord.transform_to(telescope_frame)

            true_az = shower_coord_in_telescope.altaz.az.to(u.rad)
            true_alt = shower_coord_in_telescope.altaz.alt.to(u.rad)
            
            offcenter = angular_separation(0 * u.deg, 0 * u.deg,
                                        shower_coord_in_telescope.fov_lon,
                                        shower_coord_in_telescope.fov_lat)
            offcenter = offcenter.to(u.deg)
            
            evt_id = np.arange(len(tel_az))
            obs_id = np.repeat(run_number, len(tel_az))
            tel_id = np.repeat(telelscope, len(tel_az))
            
            data_ = {
                'obs_id': obs_id, 
                'tel_id': tel_id,
                'event_id': evt_id,
                'tel_az': tel_az,
                'tel_alt': tel_alt,
                'true_az': true_az,
                'true_alt': true_alt,
                'true_energy': true_energy
            }
            
            df_ = pd.DataFrame(data=data_)
            
            shower_data = shower_data.append(df_)

    shower_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    
    return shower_data


# =================
# === Main code ===
# =================

# --------------------------
# Adding the argument parser
arg_parser = argparse.ArgumentParser(description="""
This tools adds the "original MC" tree info to the MC events tree processed earlier.
""")

arg_parser.add_argument("--config", default="config.yaml",
                        help='Configuration file to steer the code execution.')
arg_parser.add_argument("--usetest",
                        help='Process only test files.',
                        action='store_true')
arg_parser.add_argument("--usetrain",
                        help='Process only train files.',
                        action='store_true')
arg_parser.add_argument("--usem1",
                        help='Process only M1 files.',
                        action='store_true')
arg_parser.add_argument("--usem2",
                        help='Process only M2 files.',
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
# ------------------------------


# ------------------------------
# Magic telescopes configuration

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
magic_subarray = SubarrayDescription('MAGIC', 
                                     magic_tel_positions, 
                                     magic_tel_descriptions)

# ------------------------------

if parsed_args.usetrain and parsed_args.usetest:
    data_sample_to_process = ['train_sample', 'test_sample']
elif parsed_args.usetrain:
    data_sample_to_process = ['train_sample']
elif parsed_args.usetest:
    data_sample_to_process = ['test_sample']
else:
    data_sample_to_process = ['train_sample', 'test_sample']

if parsed_args.usem1 and parsed_args.usem2:
    telescope_to_process = ['magic1', 'magic2']
elif parsed_args.usem1:
    telescope_to_process = ['magic1']
elif parsed_args.usem2:
    telescope_to_process = ['magic2']
else:
    telescope_to_process = ['magic1', 'magic2']

for data_type in config['data_files']:
    for sample in data_sample_to_process:
        for telescope in telescope_to_process:
            is_mc = data_type.lower() == "mc"

            if is_mc:
                info_message(f'Processing "{data_type}", sample "{sample}", telescope "{telescope}"',
                            prefix='OriginalMC')

                try:
                    telescope_type = re.findall('(.*)[_\d]+', telescope)[0]
                except:
                    ValueError(f'Can not recognize the telescope type from name "{telescope}"')

                shower_data = read_original_mc_tree(config['data_files'][data_type][sample][telescope]['input_mask'])

                shower_data.to_hdf(config['data_files'][data_type][sample][telescope]['hillas_output'],
                                   key='dl1/original_mc',
                                   mode='a')
