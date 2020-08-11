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

from ctapipe_io_magic import MAGICEventSource

from ctapipe.io import HDF5TableWriter
from ctapipe.core.container import Container, Field
from ctapipe.reco import HillasReconstructor
from ctapipe.image import hillas_parameters, leakage
from ctapipe.image.timing import timing_parameters
from ctapipe.image.cleaning import tailcuts_clean     # apply_time_delta_cleaning

from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz


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

    island_ids = np.zeros(geom.n_pixels)
    island_ids[mask] = labels + 1

    # Finding the islands "sizes" (total charges)
    island_sizes = np.zeros(num_islands)
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
    island_ids = np.zeros(camera.n_pixels)
    island_ids[clean_mask] = labels + 1

    # Finding the islands "sizes" (total charges)
    island_sizes = np.zeros(num_islands)
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
    island_ids = np.zeros(camera.n_pixels)
    island_ids[clean_mask] = labels + 1

    # Finding the islands "sizes" (total charges)
    island_sizes = np.zeros(num_islands)
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

    # Now let's loop over the events and perform:
    #  - image cleaning;
    #  - hillas parameter calculation;
    #  - time gradient calculation.
    #  
    # We'll write the result to the HDF5 file that can be used for further processing.

    hillas_reconstructor = HillasReconstructor()

    charge_thresholds = image_cleaning_settings['charge_thresholds']
    time_thresholds = image_cleaning_settings['time_thresholds']

    horizon_frame = AltAz()

    # Opening the output file
    with HDF5TableWriter(filename=output_name, group_name='dl1', overwrite=True) as writer:
        # Creating an input source

        # Event source
        source = MAGICEventSource(input_url=input_mask)

        # Looping over the events
        for event in source:
            tels_with_data = event.r1.tels_with_data

            array_pointing = SkyCoord(
                alt=event.mc.alt,
                az=event.mc.az,
                frame=horizon_frame,
            )

            computed_hillas_params = dict()
            telescope_pointings = dict()

            # Looping over the triggered telescopes
            for tel_id in tels_with_data:
                # Obtained image
                event_image = event.dl1.tel[tel_id].image
                # Pixel arrival time map
                event_pulse_time = event.dl1.tel[tel_id].peak_time
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

                    computed_hillas_params[tel_id] = hillas_params

                    telescope_pointings[tel_id] = SkyCoord(
                        alt=event.pointing.tel[tel_id].altitude,
                        az=event.pointing.tel[tel_id].azimuth,
                        frame=horizon_frame,
                    )

                    # Preparing metadata
                    event_info = InfoContainer(
                        obs_id=event.index.obs_id,
                        event_id=scipy.int32(event.index.event_id),
                        tel_id=tel_id,
                        true_energy=event.mc.energy,
                        true_alt=event.mc.alt.to(u.rad),
                        true_az=event.mc.az.to(u.rad),
                        tel_alt=event.pointing.tel[tel_id].altitude.to(u.rad),
                        tel_az=event.pointing.tel[tel_id].azimuth.to(u.rad),
                        n_islands=num_islands
                    )

                    # Storing the result
                    writer.write("hillas_params", (event_info, hillas_params, leakage_params, timing_params))
                else:
                    print(f"Event ID {event.index.event_id} (obs ID: {event.index.obs_id}; "
                        "telescope ID: {tel_id}) did not pass cleaning.")

            if len(computed_hillas_params.keys()) > 1:
                stereo_params = hillas_reconstructor.predict(computed_hillas_params, source.subarray, array_pointing, telescope_pointings)
                event_info.tel_id = -1
                # Storing the result
                writer.write("stereo_params", (event_info, stereo_params))

def process_dataset_data(input_mask, output_name, image_cleaning_settings):
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

    # Now let's loop over the events and perform:
    #  - image cleaning;
    #  - hillas parameter calculation;
    #  - time gradient calculation.
    #  
    # We'll write the result to the HDF5 file that can be used for further processing.

    hillas_reconstructor = HillasReconstructor()

    charge_thresholds = image_cleaning_settings['charge_thresholds']
    time_thresholds = image_cleaning_settings['time_thresholds']

    horizon_frame = AltAz()

    # Opening the output file
    with HDF5TableWriter(filename=output_name, group_name='dl1', overwrite=True) as writer:
        # Creating an input source
        source = MAGICEventSource(input_url=input_mask)

        # Looping over the events
        for event in source:
            tels_with_data = event.r1.tels_with_data

            array_pointing = SkyCoord(
                alt=event.pointing.tel[0].altitude,
                az=event.pointing.tel[0].azimuth,
                frame=horizon_frame,
            )

            computed_hillas_params = dict()
            telescope_pointings = dict()

            # Looping over the triggered telescopes
            for tel_id in tels_with_data:
                # Obtained image
                event_image = event.dl1.tel[tel_id].image
                # Pixel arrival time map
                event_pulse_time = event.dl1.tel[tel_id].peak_time
                # Camera geometry
                camera = source.subarray.tel[tel_id].camera

                clean_mask = magic_clean_step1(camera,event_image,core_thresh=charge_thresholds['picture_thresh'])

                if event_image[clean_mask].sum() == 0:
                    # Event did not survive image cleaining
                    continue

                clean_mask = magic_clean_step2(camera, clean_mask, event_image, event_pulse_time,
                               max_time_off=time_thresholds['max_time_off'],
                               core_thresh=charge_thresholds['picture_thresh'])

                if event_image[clean_mask].sum() == 0:
                    # Event did not survive image cleaining
                    continue

                clean_mask = magic_clean_step3(camera, clean_mask, event_image, event_pulse_time,
                               max_time_diff=time_thresholds['max_time_diff'],
                               boundary_thresh=charge_thresholds['boundary_thresh'])

                if event_image[clean_mask].sum() == 0:
                    # Event did not survive image cleaining
                    continue

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

                    computed_hillas_params[tel_id] = hillas_params

                    telescope_pointings[tel_id] = SkyCoord(
                        alt=event.pointing.tel[tel_id].altitude,
                        az=event.pointing.tel[tel_id].azimuth,
                        frame=horizon_frame,
                    )

                    # Preparing metadata
                    event_info = InfoContainer(
                        obs_id=event.index.obs_id,
                        event_id=scipy.int32(event.index.event_id),
                        tel_id=tel_id,
                        mjd=event.trigger.time.mjd,
                        tel_alt=event.pointing.tel[tel_id].altitude.to(u.rad),
                        tel_az=event.pointing.tel[tel_id].azimuth.to(u.rad),
                        n_islands=num_islands
                    )

                    # Storing the result
                    writer.write("hillas_params", (event_info, hillas_params, leakage_params, timing_params))
                else:
                    print(f"Event ID {event.index.event_id} (obs ID: {event.index.obs_id}; "
                        "telescope ID: {tel_id}) did not pass cleaning.")

            if len(computed_hillas_params.keys()) > 1:
                stereo_params = hillas_reconstructor.predict(computed_hillas_params, source.subarray, array_pointing, telescope_pointings)
                event_info.tel_id = -1
                # Storing the result
                writer.write("stereo_params", (event_info, stereo_params))


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
arg_parser.add_argument("--usereal",
                        help='Process only real data files.',
                        action='store_true')
arg_parser.add_argument("--usemc",
                        help='Process only simulated data files.',
                        action='store_true')
arg_parser.add_argument("--usetest",
                        help='Process only test files.',
                        action='store_true')
arg_parser.add_argument("--usetrain",
                        help='Process only train files.',
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

if 'data_files' not in config:
    print('Error: the configuration file is missing the "data_files" section. Exiting.')
    exit()
    
if 'image_cleaning' not in config:
    print('Error: the configuration file is missing the "image_cleaning" section. Exiting.')
    exit()
# ------------------------------

if parsed_args.usereal and parsed_args.usemc:
    data_type_to_process = config['data_files']
elif parsed_args.usereal:
    data_type_to_process = ['data']
elif parsed_args.usemc:
    data_type_to_process = ['mc']
else:
    data_type_to_process = config['data_files']

if parsed_args.usetrain and parsed_args.usetest:
    data_sample_to_process = ['train_sample', 'test_sample']
elif parsed_args.usetrain:
    data_sample_to_process = ['train_sample']
elif parsed_args.usetest:
    data_sample_to_process = ['test_sample']
else:
    data_sample_to_process = ['train_sample', 'test_sample']

telescope_to_process = ['magic1', 'magic2']

for data_type in data_type_to_process:
    for sample in data_sample_to_process:
        try:
            telescope_type = re.findall('(.*)[_\d]+', telescope_to_process[0])[0]
        except:
            ValueError(f'Can not recognize the telescope type from name "{telescope_to_process}"')

        info_message(f'Data "{data_type}", sample "{sample}", telescope "{telescope_type}"',
                    prefix='Hillas')

        if telescope_type not in config['image_cleaning']:
            raise ValueError(f'Guessed telescope type "{telescope_type}" does not have image cleaning settings')

        is_mc = data_type.lower() == "mc"

        if is_mc:
            process_dataset_mc(input_mask=config['data_files'][data_type][sample]['magic1']['input_mask'],
                               output_name=config['data_files'][data_type][sample]['magic1']['hillas_output'],
                               image_cleaning_settings=config['image_cleaning'][telescope_type])
        else:
            process_dataset_data(input_mask=config['data_files'][data_type][sample]['magic1']['input_mask'],
                                output_name=config['data_files'][data_type][sample]['magic1']['hillas_output'],
                                image_cleaning_settings=config['image_cleaning'][telescope_type])
