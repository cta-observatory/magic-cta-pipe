#!/usr/bin/env python
# coding: utf-8

import re
import os
import sys
import glob
import copy
import time
import scipy
import argparse
import warnings 

import pandas as pd
import numpy as np

from scipy.sparse.csgraph import connected_components
from astropy import units as u
from ctapipe_io_magic import MAGICEventSource
from ctapipe.io import HDF5TableWriter
from ctapipe.core.container import Container, Field
from ctapipe.image import hillas_parameters, leakage
from ctapipe.image.timing import timing_parameters
from ctapipe.image.cleaning import tailcuts_clean     

from utils import MAGIC_Badpixels
from utils import MAGIC_Cleaning

warnings.simplefilter('ignore') 

def get_num_islands(camera, clean_mask, event_image):
    """Get the number of connected islands in a shower image.

    Parameters
    ----------
    camera : CameraGeometry
        Description
    clean_mask : np.array
        Cleaning mask
    event_image : np.array
        Event image

    Returns
    -------
    int
        Number of islands
    """

    neighbors = camera.neighbor_matrix_sparse
    clean_neighbors = neighbors[clean_mask][:, clean_mask]
    num_islands, labels = connected_components(clean_neighbors, directed=False)

    return num_islands

class InfoContainer(Container):
    obs_id = Field(-1, "Observation ID")
    event_id = Field(-1, "Event ID")
    tel_id = Field(-1, "Telescope ID")
    mjd = Field(-1, "Event mjd")
    millisec = Field(-1, "Event millisec")
    nanosec = Field(-1, "Event nanosec")
    tel_alt = Field(-1, "MC telescope altitude", unit=u.rad)
    tel_az = Field(-1, "MC telescope azimuth", unit=u.rad)
    n_islands = Field(-1, "Number of image islands")

cleaning_config = dict(
    picture_thresh = 6,
    boundary_thresh = 3.5,
    max_time_off = 4.5 * 1.64,
    max_time_diff = 1.5 * 1.64,
    usetime = True,
    usesum = True,
    findhotpixels=True,
)

bad_pixels_config = dict(
    pedestalLevel = 400,
    pedestalLevelVariance = 4.5,
    pedestalType = 'FromExtractorRndm'
)

# ========================
# === Get the argument ===
# ========================

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--input-dir', '-i', dest='input_dir', type=str, default='./data_calibrated', 
                        help='Path to the directory that contains the input files')
arg_parser.add_argument('--output-dir', '-o', dest='output_dir', type=str, default='./data_dl1', 
                        help='Path to the direcotry that the output files are stored')

args = arg_parser.parse_args()

# ============
# === Main ===
# ============

start_time = time.time()

input_mask = args.input_dir + "/*_Y_*.root"

data_paths = glob.glob(input_mask)
data_paths.sort()

if data_paths == []:
    print('Not accessible to the input files. Please check the path to the files. Exiting.')
    sys.exit()

re_parser = re.findall('(\d+)_M(\d)_(\d+)\.(\d+)_(\D)_.*-W0.40.*.root', data_paths[0])[0]

tel_id = int(re_parser[1])
obs_id = str(re_parser[2])

os.makedirs(args.output_dir, exist_ok=True)
output_name = args.output_dir + f'/dl1_magic{tel_id}_run{obs_id}.h5'

# Now let's loop over the events and perform:
#  - image cleaning;
#  - hillas parameter calculation;
#  - timing parameter calculation;
#  - leakage parameter calculation
#  
# We'll write the result to the HDF5 file that can be used for further processing.

previous_event_id = 0

with HDF5TableWriter(filename=output_name, group_name='events', overwrite=True) as writer:

    print('\nLoading the input files...')
    
    source = MAGICEventSource(input_url=input_mask)

    camera = source.subarray.tel[tel_id].camera
    magic_clean = MAGIC_Cleaning.magic_clean(camera,cleaning_config)
    badpixel_calculator = MAGIC_Badpixels.MAGICBadPixelsCalc(config=bad_pixels_config)

    print('\nProcessing the events...')

    for i_ev, event in enumerate(source._mono_event_generator(telescope=f'M{tel_id}')):  
        
        if i_ev%1000 == 0:  
            print(f'{i_ev} events')

        #Exclude pedestal runs?? 
        if previous_event_id == event.index.event_id:
            continue

        previous_event_id = copy.copy(event.index.event_id)

        tels_with_data = event.r1.tels_with_data

        computed_hillas_params = dict()
        pointing_alt = dict()
        pointing_az = dict()

        for tel_id in tels_with_data:
            event_image = event.dl1.tel[tel_id].image
            event_pulse_time = event.dl1.tel[tel_id].peak_time

            badrmspixel_mask = badpixel_calculator.get_badrmspixel_mask(event)
            deadpixel_mask = badpixel_calculator.get_deadpixel_mask(event)
            unsuitable_mask = np.logical_or(badrmspixel_mask[tel_id-1], deadpixel_mask[tel_id-1])

            clean_mask, event_image, event_pulse_time = magic_clean.clean_image(event_image, event_pulse_time,unsuitable_mask=unsuitable_mask)

            num_islands = get_num_islands(camera, clean_mask, event_image)

            event_image_cleaned = event_image.copy()
            event_image_cleaned[~clean_mask] = 0

            event_pulse_time_cleaned = event_pulse_time.copy()
            event_pulse_time_cleaned[~clean_mask] = 0

            if np.any(event_image_cleaned):
                try:
                    # If event has survived the cleaning, computing the Hillas parameters
                    hillas_params = hillas_parameters(camera, event_image_cleaned)
                    image_mask = event_image_cleaned > 0
                    timing_params = timing_parameters(camera,
                                                    event_image_cleaned,
                                                    event_pulse_time_cleaned,
                                                    hillas_params,
                                                    image_mask)
                    leakage_params = leakage(camera, event_image, clean_mask)

                    event_info = InfoContainer(
                        obs_id=event.index.obs_id,
                        event_id=scipy.int32(event.index.event_id),
                        tel_id=tel_id,
                        mjd=event.trigger.mjd,
                        millisec=event.trigger.millisec,
                        nanosec=event.trigger.nanosec,
                        tel_alt=event.pointing.tel[tel_id].altitude.to(u.rad),
                        tel_az=event.pointing.tel[tel_id].azimuth.to(u.rad),
                        n_islands=num_islands
                    )

                    writer.write("params", (event_info, hillas_params, leakage_params, timing_params))

                except ValueError:
                    print(f'-->  Event ID {event.index.event_id}: parameter calculation failed. Skipping.')
                    continue
                    
            else:
                # Event did not pass cleaning. Skipping.
                continue
                

end_time = time.time()
print(f'elapsed time = {end_time - start_time} [sec]')


