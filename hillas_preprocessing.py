#!/usr/bin/env python
# coding: utf-8

import datetime
import argparse
import glob
import re
import yaml

import pandas as pd

import numpy as np
import scipy
from scipy.sparse.csgraph import connected_components

import traitlets

import ctapipe

from ctapipe_io_magic import MAGICEventSource, MAGICEventSourceMC

from ctapipe.io import HDF5TableWriter
from ctapipe.core.container import Container, Field
from ctapipe.calib import CameraCalibrator
from ctapipe.reco import HillasReconstructor
from ctapipe.image import hillas_parameters, leakage
from ctapipe.image.timing_parameters import timing_parameters
from ctapipe.image.cleaning import tailcuts_clean     # apply_time_delta_cleaning

from astropy import units as u


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

# Added on 06/07/2019
def magic_clean_step1(geom, charge_map, core_thresh):
    mask = charge_map <= core_thresh
    return ~mask

# Added on 06/07/2019
def magic_clean_step2(geom, mask, charge_map, arrival_times, 
                      max_time_off, core_thresh, 
                      usetime=True):
    
    pixels_to_remove = []

    neighbors = geom.neighbor_matrix_sparse
    clean_neighbors = neighbors[mask][:, mask]
    num_islands, labels = connected_components(clean_neighbors, directed=False)

    island_ids = scipy.zeros(geom.n_pixels)
    island_ids[mask] = labels + 1

    # Finding the islands "sizes" (total charges)
    island_sizes = scipy.zeros(num_islands)
    for i in range(num_islands):
        island_sizes[i] = charge_map[mask][labels == i].sum()
      
    # Disabling pixels for all islands save the brightest one
    brightest_id = island_sizes.argmax() + 1

    if usetime:
        brightest_pixel_times = arrival_times[mask & (island_ids == brightest_id)]
        brightest_pixel_charges = charge_map[mask & (island_ids == brightest_id)]

        brightest_time = np.sum(brightest_pixel_times * brightest_pixel_charges**2) / np.sum(brightest_pixel_charges**2)

        time_diff = np.abs(arrival_times - brightest_time)

        mask[(charge_map > 2*core_thresh) & (time_diff > 2*max_time_off)] = False
        mask[(charge_map < 2*core_thresh) & (time_diff > max_time_off)] = False

    mask = single_island(geom,mask,charge_map)
    
    return mask

# Added on 06/07/2019
def magic_clean_step3(geom, mask, event_image, arrival_times, 
                      max_time_diff, boundary_thresh, 
                      usetime=True):

    thing = []
    core_mask = mask.copy()
    
    pixels_with_picture_neighbors_matrix = geom.neighbor_matrix_sparse

    for pixel in np.where(event_image)[0]:
        
        if pixel in np.where(core_mask)[0]:
            continue

        if event_image[pixel] <= boundary_thresh:
            continue
        
        hasNeighbor = False
        if usetime:
            
            neighbors = geom.neighbor_matrix_sparse[pixel].indices
            
            for neighbor in neighbors:
                if neighbor not in np.where(core_mask)[0]:
                    continue
                time_diff = np.abs(arrival_times[neighbor] - arrival_times[pixel])
                if time_diff < max_time_diff:
                    hasNeighbor = True
                    break
            if not hasNeighbor:
                continue
            
        if not pixels_with_picture_neighbors_matrix.dot(core_mask)[pixel]:
            continue
        
        thing.append(pixel)
        
    mask[thing] = True

    return mask

# Added on 02/07/2019
def apply_time_delta_cleaning(geom, mask, core_mask, arrival_times,
                              min_number_neighbors, time_limit):
    """ Remove all pixels from selection that have less than N
    neighbors that arrived within a given timeframe.
    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    mask: array, boolean
        boolean mask of *clean* pixels before time_delta_cleaning
    arrival_times: array
        pixel timing information
    min_number_neighbors: int
        Threshold to determine if a pixel survives cleaning steps.
        These steps include checks of neighbor arrival time and value
    time_limit: int or float
        arrival time limit for neighboring pixels
    Returns
    -------
    A boolean mask of *clean* pixels.  To get a zero-suppressed image and pixel
    list, use `image[mask], geom.pix_id[mask]`, or to keep the same
    image size and just set unclean pixels to 0 or similar, use
    `image[~mask] = 0`
    """
    pixels_to_remove = []
    for pixel in np.where(mask)[0]:
        if pixel in np.where(core_mask)[0]:
            continue
        neighbors = geom.neighbor_matrix_sparse[pixel].indices
        time_diff = np.abs(arrival_times[neighbors] - arrival_times[pixel])
        if sum(time_diff < time_limit) < min_number_neighbors:
            pixels_to_remove.append(pixel)
    mask[pixels_to_remove] = False
    return mask

# Added on 02/07/2019
def single_island(camera, mask, image):
    pixels_to_remove = []
    neighbors = camera.neighbor_matrix
    for pix_id in np.where(mask)[0]:
        if len(set(np.where(neighbors[pix_id] & mask)[0])) == 0:
            pixels_to_remove.append(pix_id)
    mask[pixels_to_remove] = False
    return mask

    
def apply_magic_time_off_cleaning(camera, clean_mask, charge_map, arrival_times, max_time_off, picture_thresh):
    # Identifying connected islands
    neighbors = camera.neighbor_matrix_sparse
    clean_neighbors = neighbors[clean_mask][:, clean_mask]
    num_islands, labels = connected_components(clean_neighbors, directed=False)

    # Marking pixels according to their islands
    island_ids = scipy.zeros(camera.n_pixels)
    island_ids[clean_mask] = labels + 1

    # Finding the islands "sizes" (total charges)
    island_sizes = scipy.zeros(num_islands)
    for i in range(num_islands):
        island_sizes[i] = charge_map[clean_mask][labels == i].sum()
        
    # Disabling pixels for all islands save the brightest one
    brightest_id = island_sizes.argmax() + 1

    core_pixel_times = arrival_times[clean_mask & (island_ids == brightest_id)]
    core_pixel_charges = charge_map[clean_mask & (island_ids == brightest_id)]
    
    core_pixel_times = core_pixel_times[core_pixel_charges > 6]
    core_pixel_charges = core_pixel_charges[core_pixel_charges > 6]
    
    core_time = np.sum(core_pixel_times * core_pixel_charges**2) / np.sum(core_pixel_charges**2)
    #core_time = core_pixel_times[core_pixel_charges.argmax()]
    
    time_diff = np.abs(arrival_times - core_time)
    
    clean_mask[(charge_map >= 2*picture_thresh) & (time_diff > 2*max_time_off)] = False
    clean_mask[(charge_map < 2*picture_thresh) & (time_diff > max_time_off)] = False
    
    return clean_mask


def filter_brightest_island(camera, clean_mask, event_image):
    # Identifying connected islands
    neighbors = camera.neighbor_matrix_sparse
    clean_neighbors = neighbors[clean_mask][:, clean_mask]
    num_islands, labels = connected_components(clean_neighbors, directed=False)

    # Marking pixels according to their islands
    island_ids = scipy.zeros(camera.n_pixels)
    island_ids[clean_mask] = labels + 1

    # Finding the islands "sizes" (total charges)
    island_sizes = scipy.zeros(num_islands)
    for i in range(num_islands):
        island_sizes[i] = event_image[clean_mask][labels == i].sum()
        
    # Disabling pixels for all islands save the brightest one
    brightest_id = island_sizes.argmax() + 1
    clean_mask[island_ids != brightest_id] = False

    return clean_mask


def get_num_islands(camera, clean_mask, event_image):
    # Identifying connected islands
    neighbors = camera.neighbor_matrix_sparse
    clean_neighbors = neighbors[clean_mask][:, clean_mask]
    num_islands, labels = connected_components(clean_neighbors, directed=False)

    return num_islands


def process_dataset_mc(input_mask, output_name, image_cleaning_settings):
    # Create event metadata container to hold event / observation / telescope IDs 
    # and MC true values for the event energy and direction. We will need it to add 
    # this information to the event Hillas parameters when dumping the results to disk.

    class InfoContainer(Container):
        obs_id = Field(-1, "Observation ID")
        event_id = Field(-1, "Event ID")
        tel_id = Field(-1, "Telescope ID")
        true_energy = Field(-1, "MC event energy", unit=u.TeV)
        true_alt = Field(-1, "MC event altitude", unit=u.rad)
        true_az = Field(-1, "MC event azimuth", unit=u.rad)
        tel_alt = Field(-1, "MC telescope altitude", unit=u.rad)
        tel_az = Field(-1, "MC telescope azimuth", unit=u.rad)
        n_islands = Field(-1, "Number of image islands")

    # Setting up the calibrator class.

    config = traitlets.config.Config()
    integrator_name = 'LocalPeakWindowSum'
    config[integrator_name]['window_width'] = 5
    config[integrator_name]['window_shift'] = 2

    calibrator = CameraCalibrator(image_extractor=integrator_name, config=config)

    # Finding available MC files
    input_files = glob.glob(input_mask)
    input_files.sort()

    # Now let's loop over the events and perform:
    #  - image cleaning;
    #  - hillas parameter calculation;
    #  - time gradient calculation.
    #  
    # We'll write the result to the HDF5 file that can be used for further processing.

    hillas_reconstructor = HillasReconstructor()

    charge_thresholds = image_cleaning_settings['charge_thresholds']
    time_thresholds = image_cleaning_settings['time_thresholds']
    #core_charge_thresholds = charge_thresholds.copy()
    #core_charge_thresholds['boundary_thresh'] = core_charge_thresholds['picture_thresh']

    # Opening the output file
    with HDF5TableWriter(filename=output_name, group_name='dl1', overwrite=True) as writer:
        # Creating an input source

        for input_file in input_files:
            file_name = input_file.split('/')[-1]
            print("")
            print(f"-- Working on {file_name:s} --")
            print("")
            # Event source
            source = MAGICEventSourceMC(input_url=input_file)
            
            # Looping over the events
            for event in source:
                tels_with_data = event.r1.tels_with_data

                # Calibrating an event
                # calibrator.calibrate(event)

                computed_hillas_params = dict()
                pointing_alt = dict()
                pointing_az = dict()
                
                # Looping over the triggered telescopes
                for tel_id in tels_with_data:
                    # Obtained image
                    event_image = event.dl1.tel[tel_id].image
                    # Pixel arrival time map
                    event_pulse_time = event.dl1.tel[tel_id].pulse_time
                    # Camera geometry
                    camera = source.subarray.tel[tel_id].camera

                    # Added on 06/07/2019
                    clean_mask = magic_clean_step1(camera,event_image,core_thresh=charge_thresholds['picture_thresh'])

                    if event_image[clean_mask].sum() == 0:                                                                                                           
                        # Event did not survive image cleaining                                                                                                      
                        continue    
                    
                    clean_mask = magic_clean_step2(camera, clean_mask, event_image, event_pulse_time, 
                               max_time_off=time_thresholds['max_time_off'], 
                               core_thresh=charge_thresholds['picture_thresh'])
                               #usetime=usetime)

                    if event_image[clean_mask].sum() == 0:
                        # Event did not survive image cleaining                                                                                                     
                        continue
                               
                    clean_mask = magic_clean_step3(camera, clean_mask, event_image, event_pulse_time, 
                               max_time_diff=time_thresholds['max_time_diff'], 
                               boundary_thresh=charge_thresholds['boundary_thresh'])
                               #usetime=usetime)

                    if event_image[clean_mask].sum() == 0:
                        # Event did not survive image cleaining                                                                                                   
                        continue

                               
                    # ---------------------------
                    # Computing the cleaning mask
                    #clean_mask = tailcuts_clean(camera, event_image, 
                    #                             **charge_thresholds)
                    #clean_mask_core = tailcuts_clean(camera, event_image,
                    #                                  **core_charge_thresholds)
                    #                         
                    #if event_image[clean_mask].sum() == 0:
                        # Event did not survive image cleaining
                    #    continue

                    #clean_mask = apply_time_delta_cleaning(camera, clean_mask, clean_mask_core, 
                    #                                       event_pulse_time,
                    #                                       time_limit=time_thresholds['time_limit'], 
                    #                                       min_number_neighbors=time_thresholds['min_number_neighbors'])

                    #if event_image[clean_mask].sum() == 0:
                        # Event did not survive image cleaining
                    #    continue

                    #clean_mask = apply_magic_time_off_cleaning(camera, clean_mask, 
                    #                                           event_image, event_pulse_time, 
                    #                                           max_time_off=time_thresholds['max_time_off'],
                    #                                           picture_thresh=charge_thresholds['picture_thresh'])

                    #if event_image[clean_mask].sum() == 0:
                        # Event did not survive image cleaining
                    #    continue

                    ### Added on 02/07/2019
                    #clean_mask = single_island(camera,clean_mask,event_image)

                    #if event_image[clean_mask].sum() == 0:
                        # Event did not survive image cleaining
                    #    continue
                    
                    num_islands = get_num_islands(camera, clean_mask, event_image)
                    #clean_mask = filter_brightest_island(camera, clean_mask, event_image)
                    # ---------------------------
                    
                    event_image_cleaned = event_image.copy()
                    event_image_cleaned[~clean_mask] = 0
                    
                    event_pulse_time_cleaned = event_pulse_time.copy()
                    event_pulse_time_cleaned[~clean_mask] = 0
                    
                    # if event_image_cleaned.sum() > 0:
                    if len(event_image[clean_mask]) > 3:
                        # If event has survived the cleaning, computing the Hillas parameters
                        hillas_params = hillas_parameters(camera, event_image_cleaned)
                        timing_params = timing_parameters(camera, 
                                                          event_image_cleaned, 
                                                          event_pulse_time_cleaned, 
                                                          hillas_params)
                        leakage_params = leakage(camera, event_image, clean_mask)
                        
                        #computed_hillas_params[tel_id] = hillas_params
                        
                        #pointing_alt[tel_id] = event.pointing[tel_id].altitude.to(u.rad)
                        #pointing_az[tel_id] = event.pointing[tel_id].azimuth.to(u.rad)
                        
                        # Preparing metadata
                        event_info = InfoContainer(obs_id=event.dl0.obs_id, 
                                                   event_id=scipy.int32(event.dl0.event_id),
                                                   tel_id=tel_id,
                                                   true_energy=event.mc.energy,
                                                   true_alt=event.mc.alt.to(u.rad),
                                                   true_az=event.mc.az.to(u.rad),
                                                   tel_alt=event.pointing[tel_id].altitude.to(u.rad),
                                                   tel_az=event.pointing[tel_id].azimuth.to(u.rad),
                                                   n_islands=num_islands)

                        # Storing the result
                        writer.write("hillas_params", (event_info, hillas_params, leakage_params, timing_params))
                        
                #if len(pointing_alt.keys()) > 1:
                    #stereo_params = hillas_reconstructor.predict(computed_hillas_params, event.inst, pointing_alt, pointing_az)
                    #event_info.tel_id = -1
                    ## Storing the result
                    #writer.write("stereo_params", (event_info, stereo_params))


def process_dataset_data(input_mask, tel_id, output_name, image_cleaning_settings):
    # Create event metadata container to hold event / observation / telescope IDs 
    # and MC true values for the event energy and direction. We will need it to add 
    # this information to the event Hillas parameters when dumping the results to disk.

    class InfoContainer(Container):
        obs_id = Field(-1, "Observation ID")
        event_id = Field(-1, "Event ID")
        tel_id = Field(-1, "Telescope ID")
        mjd = Field(-1, "Event MJD")
        tel_alt = Field(-1, "MC telescope altitude", unit=u.rad)
        tel_az = Field(-1, "MC telescope azimuth", unit=u.rad)
        n_islands = Field(-1, "Number of image islands")

    # Setting up the calibrator class.

    config = traitlets.config.Config()
    integrator_name = 'LocalPeakWindowSum'
    config[integrator_name]['window_width'] = 5
    config[integrator_name]['window_shift'] = 2

    calibrator = CameraCalibrator(image_extractor=integrator_name, config=config)

    # Now let's loop over the events and perform:
    #  - image cleaning;
    #  - hillas parameter calculation;
    #  - time gradient calculation.
    #  
    # We'll write the result to the HDF5 file that can be used for further processing.

    hillas_reconstructor = HillasReconstructor()

    charge_thresholds = image_cleaning_settings['charge_thresholds']
    time_thresholds = image_cleaning_settings['time_thresholds']
    #core_charge_thresholds = charge_thresholds.copy()
    #core_charge_thresholds['boundary_thresh'] = core_charge_thresholds['picture_thresh']

    # Opening the output file
    with HDF5TableWriter(filename=output_name, group_name='dl1', overwrite=True) as writer:
        # Creating an input source
        source = MAGICEventSource(input_url=input_mask)
        
        # Looping over the events
        for event in source._mono_event_generator(telescope=f'M{tel_id}'):
            tels_with_data = event.r1.tels_with_data

            # Calibrating an event
            # calibrator.calibrate(event)

            computed_hillas_params = dict()
            pointing_alt = dict()
            pointing_az = dict()
            
            # Looping over the triggered telescopes
            for tel_id in tels_with_data:
                # Obtained image
                event_image = event.dl1.tel[tel_id].image
                # Pixel arrival time map
                event_pulse_time = event.dl1.tel[tel_id].pulse_time
                # Camera geometry
                camera = source.subarray.tel[tel_id].camera

                # ---------------------------
                # Computing the cleaning mask
                #clean_mask = tailcuts_clean(camera, event_image, 
                #                            **charge_thresholds)
                #clean_mask_core = tailcuts_clean(camera, event_image,
                #                                  **core_charge_thresholds)
                
                clean_mask = magic_clean_step1(camera,event_image,core_thresh=charge_thresholds['picture_thresh'])

                if event_image[clean_mask].sum() == 0:
                    # Event did not survive image cleaining
                    continue

                #clean_mask = apply_time_delta_cleaning(camera, clean_mask, clean_mask_core,
                #                                       event_pulse_time,
                #                                       time_limit=time_thresholds['time_limit'], 
                #                                       min_number_neighbors=time_thresholds['min_number_neighbors'])

                clean_mask = magic_clean_step2(camera, clean_mask, event_image, event_pulse_time,
                               max_time_off=time_thresholds['max_time_off'],
                               core_thresh=charge_thresholds['picture_thresh'])

                if event_image[clean_mask].sum() == 0:
                    # Event did not survive image cleaining
                    continue

                #clean_mask = apply_magic_time_off_cleaning(camera, clean_mask,
                #                                           event_image, event_pulse_time, 
                #                                           max_time_off=time_thresholds['max_time_off'],
                #                                           picture_thresh=charge_thresholds['picture_thresh'])

                clean_mask = magic_clean_step3(camera, clean_mask, event_image, event_pulse_time,
                               max_time_diff=time_thresholds['max_time_diff'],
                               boundary_thresh=charge_thresholds['boundary_thresh'])

                if event_image[clean_mask].sum() == 0:
                    # Event did not survive image cleaining
                    continue

                ### Added on 02/07/2019                                                                                                                      
                #clean_mask = single_island(camera,clean_mask,event_image)

                #if event_image[clean_mask].sum() == 0:
                    # Event did not survive image cleaining
                #    continue
                
                num_islands = get_num_islands(camera, clean_mask, event_image)
                #clean_mask = filter_brightest_island(camera, clean_mask, event_image)
                # ---------------------------
                
                event_image_cleaned = event_image.copy()
                event_image_cleaned[~clean_mask] = 0
                
                event_pulse_time_cleaned = event_pulse_time.copy()
                event_pulse_time_cleaned[~clean_mask] = 0
                
                # if event_image_cleaned.sum() > 0:
                if len(event_image[clean_mask]) > 3:
                    # If event has survived the cleaning, computing the Hillas parameters
                    hillas_params = hillas_parameters(camera, event_image_cleaned)
                    timing_params = timing_parameters(camera, 
                                                      event_image_cleaned, 
                                                      event_pulse_time_cleaned, 
                                                      hillas_params)
                    leakage_params = leakage(camera, event_image, clean_mask)
                    
                    #computed_hillas_params[tel_id] = hillas_params
                    
                    #pointing_alt[tel_id] = event.pointing[tel_id].altitude.to(u.rad)
                    #pointing_az[tel_id] = event.pointing[tel_id].azimuth.to(u.rad)
                    
                    # Preparing metadata
                    event_info = InfoContainer(obs_id=event.dl0.obs_id, 
                                               event_id=scipy.int32(event.dl0.event_id),
                                               tel_id=tel_id,
                                               mjd=event.trig.gps_time.mjd,
                                               tel_alt=event.pointing[tel_id].altitude.to(u.rad),
                                               tel_az=event.pointing[tel_id].azimuth.to(u.rad),
                                               n_islands=num_islands)

                    # Storing the result
                    writer.write("hillas_params", (event_info, hillas_params, leakage_params, timing_params))
                    
            #if len(pointing_alt.keys()) > 1:
                #stereo_params = hillas_reconstructor.predict(computed_hillas_params, event.inst, pointing_alt, pointing_az)
                #event_info.tel_id = -1
                ## Storing the result
                #writer.write("stereo_params", (event_info, stereo_params))


# =================
# === Main code ===
# =================

# --------------------------
# Adding the argument parser
arg_parser = argparse.ArgumentParser(description="""
This tools computes the Hillas parameters for the specified data sets.
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

if 'data_files' not in config:
    print('Error: the configuration file is missing the "data_files" section. Exiting.')
    exit()
    
if 'image_cleanining' not in config:
    print('Error: the configuration file is missing the "image_cleanining" section. Exiting.')
    exit()
# ------------------------------

for data_type in config['data_files']:
    for sample in config['data_files'][data_type]:
        for telescope in config['data_files'][data_type][sample]:
            
            info_message(f'Data "{data_type}", sample "{sample}", telescope "{telescope}"',
                         prefix='Hillas')
            
            try:
                telescope_type = re.findall('(.*)[_\d]+', telescope)[0]
            except:
                ValueError(f'Can not recognize the telescope type from name "{telescope}"')
                
            if telescope_type not in config['image_cleanining']:
                raise ValueError(f'Guessed telescope type "{telescope_type}" does not have image cleaning settings')

            is_mc = data_type.lower() == "mc"

            if is_mc:
                process_dataset_mc(input_mask=config['data_files'][data_type][sample][telescope]['input_mask'],
                                   output_name=config['data_files'][data_type][sample][telescope]['hillas_output'],
                                   image_cleaning_settings=config['image_cleanining'][telescope_type])
            else:
                tel_id = re.findall('.*([_\d]+)', telescope)[0]
                tel_id = int(tel_id)
                process_dataset_data(input_mask=config['data_files'][data_type][sample][telescope]['input_mask'],
                                     tel_id=tel_id,
                                     output_name=config['data_files'][data_type][sample][telescope]['hillas_output'],
                                     image_cleaning_settings=config['image_cleanining'][telescope_type])
