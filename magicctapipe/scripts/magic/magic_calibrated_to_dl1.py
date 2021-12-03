#!/usr/bin/env python
# coding: utf-8

import datetime
import argparse
import glob
import re
import yaml
import copy
from pathlib import Path

import pandas as pd
import numpy as np
import scipy
from scipy.sparse.csgraph import connected_components

import traitlets

import ctapipe

from ctapipe_io_magic import MAGICEventSource

from ctapipe.io import DataWriter
from ctapipe.core.container import Container, Field
from ctapipe.instrument import CameraGeometry
from ctapipe.containers import (
    IntensityStatisticsContainer,
    ImageParametersContainer,
    TimingParametersContainer,
    PeakTimeStatisticsContainer,
)
from ctapipe.containers import LeakageContainer
from ctapipe.image import (
    concentration_parameters,
    hillas_parameters,
    morphology_parameters,
    timing_parameters,
)

from astropy import units as u

from magicctapipe.utils import MAGIC_Badpixels
# from utils import bad_pixel_treatment
from magicctapipe.utils import MAGIC_Cleaning
from magicctapipe.utils.utils import info_message

DEFAULT_IMAGE_PARAMETERS = ImageParametersContainer()
DEFAULT_TRUE_IMAGE_PARAMETERS = ImageParametersContainer()
DEFAULT_TRUE_IMAGE_PARAMETERS.intensity_statistics = IntensityStatisticsContainer(
    max=np.int32(-1),
    min=np.int32(-1),
    mean=np.float64(np.nan),
    std=np.float64(np.nan),
    skewness=np.float64(np.nan),
    kurtosis=np.float64(np.nan),
)
DEFAULT_TIMING_PARAMETERS = TimingParametersContainer()
DEFAULT_PEAKTIME_STATISTICS = PeakTimeStatisticsContainer()

def get_leakage(camera, event_image, clean_mask):
    """Calculate the leakage with pixels on the border of the image included #IS THIS TRUE?????

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
    LeakageContainer
    """

    neighbors = camera.neighbor_matrix_sparse

    # find pixels in the outermost ring
    outermostring = []
    for pix in range(camera.n_pixels):
        if neighbors[pix].getnnz() < 5:
            outermostring.append(pix)

    # find pixels in the second outermost ring
    outerring = []
    for pix in range(camera.n_pixels):
        if pix in outermostring:
            continue
        for neigh in np.where(neighbors[pix][0,:].toarray() == True)[1]:
            if neigh in outermostring:
                outerring.append(pix)

    # needed because outerring has some pixels appearing more than once
    outerring = np.unique(outerring).tolist()
    outermostring_mask = np.zeros(camera.n_pixels, dtype=bool)
    outermostring_mask[outermostring] = True
    outerring_mask = np.zeros(camera.n_pixels, dtype=bool)
    outerring_mask[outerring] = True
    # intersection between 1st outermost ring and cleaning mask
    mask1 = np.array(outermostring_mask) & clean_mask
    # intersection between 2nd outermost ring and cleaning mask
    mask2 = np.array(outerring_mask) & clean_mask

    leakage_pixel1 = np.count_nonzero(mask1)
    leakage_pixel2 = np.count_nonzero(mask2)

    leakage_intensity1 = np.sum(event_image[mask1])
    leakage_intensity2 = np.sum(event_image[mask2])

    size = np.sum(event_image[clean_mask])

    return LeakageContainer(
        pixels_width_1=leakage_pixel1 / camera.n_pixels,
        pixels_width_2=leakage_pixel2 / camera.n_pixels,
        intensity_width_1=leakage_intensity1 / size,
        intensity_width_2=leakage_intensity2 / size,
    )

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

def scale_camera_geometry(camera_geom, factor):
    """Scale given camera geometry of a given (constant) factor
    
    Parameters
    ----------
    camera : CameraGeometry
        Camera geometry
    factor : float
        Scale factor
    
    Returns
    -------
    CameraGeometry
        Scaled camera geometry
    """
    pix_x_scaled = factor*camera_geom.pix_x
    pix_y_scaled = factor*camera_geom.pix_y
    pix_area_scaled = camera_geom.guess_pixel_area(pix_x_scaled, pix_y_scaled, camera_geom.pix_type)

    return CameraGeometry(
        camera_name='MAGICCam',
        pix_id=camera_geom.pix_id,
        pix_x=pix_x_scaled,
        pix_y=pix_y_scaled,
        pix_area=pix_area_scaled,
        pix_type=camera_geom.pix_type,
        pix_rotation=camera_geom.pix_rotation,
        cam_rotation=camera_geom.cam_rotation
    )

def reflected_camera_geometry(camera_geom):
    """Reflect camera geometry (x->-y, y->-x)

    Parameters
    ----------
    camera_geom : CameraGeometry
        Camera geometry

    Returns
    -------
    CameraGeometry
        Reflected camera geometry
    """

    return CameraGeometry(
        camera_name='MAGICCam',
        pix_id=camera_geom.pix_id,
        pix_x=-1.*camera_geom.pix_y,
        pix_y=-1.*camera_geom.pix_x,
        pix_area=camera_geom.guess_pixel_area(camera_geom.pix_x, camera_geom.pix_y, camera_geom.pix_type),
        pix_type=camera_geom.pix_type,
        pix_rotation=camera_geom.pix_rotation,
        cam_rotation=camera_geom.cam_rotation
    )

def process_dataset_mc(input_mask, tel_id, cleaning_config):
    """Create event metadata container to hold event / observation / telescope
    IDs and MC true values for the event energy and direction. We will need it
    to add this information to the event Hillas parameters when dumping the
    results to disk.

    Parameters
    ----------
    input_mask : str
        Mask for MC input files. Reading of files is managed
        by the MAGICEventSource class.
    tel_id : int
        Telescope ID
    output_name : str
        Name of the HDF5 output file.
    cleaning_config: dict
        Dictionary for cleaning settings

    Returns
    -------
    None
    """

    cleaning_config["findhotpixels"] = False

    aberration_factor = 1./1.0713

    # Finding available MC files
    input_files = glob.glob(input_mask)
    input_files.sort()

    # Now let's loop over the events and perform:
    #  - image cleaning;
    #  - hillas parameter calculation;
    #  - time gradient calculation.
    #  
    # We'll write the result to the HDF5 file that can be used for further processing.

    for input_file in input_files:
        file_name = Path(input_file).name
        output_name = file_name.replace(".root", ".h5")
        print("")
        print(f"-- Working on {file_name:s} --")
        print("")
        # Event source
        source = MAGICEventSource(input_url=input_file)

        camera_old = source.subarray.tel[tel_id].camera.geometry
        camera_refl = reflected_camera_geometry(camera_old)
        geometry = scale_camera_geometry(camera_refl, aberration_factor)
        magic_clean = MAGIC_Cleaning.magic_clean(geometry,cleaning_config)

        info_message("Cleaning configuration", prefix='Hillas')
        for item in vars(magic_clean).items():
            print(f"{item[0]}: {item[1]}")
        if magic_clean.findhotpixels:
            for item in vars(magic_clean.pixel_treatment).items():
                print(f"{item[0]}: {item[1]}")

        with DataWriter(event_source=source, output_path=output_name, overwrite=True) as write_data:

            # Looping over the events
            for event in source:
                tels_with_data = data.trigger.tels_with_trigger

                # Looping over the triggered telescopes
                for tel_id in tels_with_data:
                    # Obtained image
                    event_image = event.dl1.tel[tel_id].image
                    # Pixel arrival time map
                    peak_time = event.dl1.tel[tel_id].peak_time

                    clean_mask, event_image, peak_time = magic_clean.clean_image(event_image, peak_time)

                    event.dl1.tel[tel_id].image_mask = clean_mask

                    num_islands = get_num_islands(geometry, clean_mask, event_image)

                    event_image_cleaned = event_image.copy()
                    event_image_cleaned[~clean_mask] = 0

                    event_pulse_time_cleaned = event_pulse_time.copy()
                    event_pulse_time_cleaned[~clean_mask] = 0

                    geom_selected  = geometry[clean_mask]
                    image_selected = event_image[clean_mask]

                    if np.any(event_image_cleaned):
                        try:
                            # If event has survived the cleaning, computing the Hillas parameters
                            hillas = hillas_parameters(geom=geom_selected, image=image_selected)
                            leakage_params = get_leakage(geometry, event_image, clean_mask)
                            concentration = concentration_parameters(
                                geom=geom_selected, image=image_selected, hillas_parameters=hillas
                            )
                            morphology = morphology_parameters(geom=geometry, image_mask=clean_mask)
                            intensity_statistics = descriptive_statistics(
                                image_selected, container_class=IntensityStatisticsContainer
                            )
                            if peak_time is not None:
                                timing = timing_parameters(
                                    geom=geom_selected,
                                    image=image_selected,
                                    peak_time=peak_time[clean_mask],
                                    hillas_parameters=hillas,
                                )
                                peak_time_statistics = descriptive_statistics(
                                    peak_time[clean_mask],
                                    container_class=PeakTimeStatisticsContainer,
                                )
                            else:
                                timing = DEFAULT_TIMING_PARAMETERS
                                peak_time_statistics = DEFAULT_PEAKTIME_STATISTICS

                            event.dl1.tel[tel_id].parameters = ImageParametersContainer(
                                hillas=hillas,
                                timing=timing,
                                leakage=leakage,
                                morphology=morphology,
                                concentration=concentration,
                                intensity_statistics=intensity_statistics,
                                peak_time_statistics=peak_time_statistics,
                            )

                        except ValueError:
                            print(f"Event ID {event.index.event_id} (obs ID: {event.index.obs_id}; "
                                f"telescope ID: {tel_id}): Hillas calculation failed.")
                    else:
                        print(f"Event ID {event.index.event_id} (obs ID: {event.index.obs_id}; "
                        f"telescope ID: {tel_id}) did not pass cleaning.")

                write_data(event)


def process_dataset_data(input_mask, tel_id, cleaning_config, bad_pixels_config):
    """Create event metadata container to hold event / observation / telescope
    IDs and MC true values for the event energy and direction. We will need it
    to add this information to the event Hillas parameters when dumping the
    results to disk.

    Parameters
    ----------
    input_mask : str
        Mask for MC input files. Reading of files is managed
        by the MAGICEventSource class.
    tel_id : int
        Telescope ID
    output_name : str
        Name of the HDF5 output file.
    cleaning_config: dict
        Dictionary for cleaning settings
    bad_pixels_config: dict
        Dictionary for bad pixels settings

    Returns
    -------
    None
    """

    # Now let's loop over the events and perform:
    #  - image cleaning;
    #  - hillas parameter calculation;
    #  - time gradient calculation.
    #  
    # We'll write the result to the HDF5 file that can be used for further processing.

    previous_event_id = 0

    aberration_factor = 1./1.0713

    input_files = glob.glob(input_mask)
    input_files.sort()

    # Now let's loop over the events and perform:
    #  - image cleaning;
    #  - hillas parameter calculation;
    #  - time gradient calculation.
    #  
    # We'll write the result to the HDF5 file that can be used for further processing.

    for input_file in input_files:
        file_name = Path(input_file).name
        output_name = file_name.replace(".root", ".h5")
        print("")
        print(f"-- Working on {file_name:s} --")
        print("")
        # Event source
        source = MAGICEventSource(input_url=input_file)

        camera_old = source.subarray.tel[tel_id].camera.geometry
        camera_refl = reflected_camera_geometry(camera_old)
        geometry = scale_camera_geometry(camera_refl, aberration_factor)
        magic_clean = MAGIC_Cleaning.magic_clean(geometry,cleaning_config)
        badpixel_calculator = MAGIC_Badpixels.MAGICBadPixelsCalc(config=bad_pixels_config)

        info_message("Cleaning configuration", prefix='Hillas')
        for item in vars(magic_clean).items():
            print(f"{item[0]}: {item[1]}")
        if magic_clean.findhotpixels:
            for item in vars(magic_clean.pixel_treatment).items():
                print(f"{item[0]}: {item[1]}")

        with DataWriter(event_source=source, output_path=output_name, overwrite=True) as write_data:

            # Looping over the events
            for event in source:
                tels_with_data = data.trigger.tels_with_trigger

                # Looping over the triggered telescopes
                for tel_id in tels_with_data:
                    # Obtained image
                    event_image = event.dl1.tel[tel_id].image
                    # Pixel arrival time map
                    peak_time = event.dl1.tel[tel_id].peak_time

                    badrmspixel_mask = badpixel_calculator.get_badrmspixel_mask(event)
                    deadpixel_mask = badpixel_calculator.get_deadpixel_mask(event)
                    unsuitable_mask = np.logical_or(badrmspixel_mask[tel_id-1], deadpixel_mask[tel_id-1])

                    clean_mask, event_image, peak_time = magic_clean.clean_image(event_image, peak_time, unsuitable_mask=unsuitable_mask)

                    event.dl1.tel[tel_id].image_mask = clean_mask

                    num_islands = get_num_islands(geometry, clean_mask, event_image)

                    event_image_cleaned = event_image.copy()
                    event_image_cleaned[~clean_mask] = 0

                    event_pulse_time_cleaned = event_pulse_time.copy()
                    event_pulse_time_cleaned[~clean_mask] = 0

                    geom_selected  = geometry[clean_mask]
                    image_selected = event_image[clean_mask]

                    if np.any(event_image_cleaned):
                        try:
                            # If event has survived the cleaning, computing the Hillas parameters
                            hillas = hillas_parameters(geom=geom_selected, image=image_selected)
                            leakage_params = get_leakage(geometry, event_image, clean_mask)
                            concentration = concentration_parameters(
                                geom=geom_selected, image=image_selected, hillas_parameters=hillas
                            )
                            morphology = morphology_parameters(geom=geometry, image_mask=clean_mask)
                            intensity_statistics = descriptive_statistics(
                                image_selected, container_class=IntensityStatisticsContainer
                            )
                            if peak_time is not None:
                                timing = timing_parameters(
                                    geom=geom_selected,
                                    image=image_selected,
                                    peak_time=peak_time[clean_mask],
                                    hillas_parameters=hillas,
                                )
                                peak_time_statistics = descriptive_statistics(
                                    peak_time[clean_mask],
                                    container_class=PeakTimeStatisticsContainer,
                                )
                            else:
                                timing = DEFAULT_TIMING_PARAMETERS
                                peak_time_statistics = DEFAULT_PEAKTIME_STATISTICS

                            event.dl1.tel[tel_id].parameters = ImageParametersContainer(
                                hillas=hillas,
                                timing=timing,
                                leakage=leakage,
                                morphology=morphology,
                                concentration=concentration,
                                intensity_statistics=intensity_statistics,
                                peak_time_statistics=peak_time_statistics,
                            )

                        except ValueError:
                            print(f"Event ID {event.index.event_id} (obs ID: {event.index.obs_id}; "
                                f"telescope ID: {tel_id}): Hillas calculation failed.")
                    else:
                        print(f"Event ID {event.index.event_id} (obs ID: {event.index.obs_id}; "
                        f"telescope ID: {tel_id}) did not pass cleaning.")

                write_data(event)

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

if parsed_args.usem1 and parsed_args.usem2:
    telescope_to_process = ['magic1', 'magic2']
elif parsed_args.usem1:
    telescope_to_process = ['magic1']
elif parsed_args.usem2:
    telescope_to_process = ['magic2']
else:
    telescope_to_process = ['magic1', 'magic2']

for data_type in data_type_to_process:
    for sample in data_sample_to_process:
        for telescope in telescope_to_process:

            info_message(f'Data "{data_type}", sample "{sample}", telescope "{telescope}"',
                         prefix='Hillas')

            try:
                telescope_type = re.findall('(.*)[_\d]+', telescope)[0]
            except:
                ValueError(f'Can not recognize the telescope type from name "{telescope}"')

            if telescope_type not in config['image_cleaning']:
                raise ValueError(f'Guessed telescope type "{telescope_type}" does not have image cleaning settings')

            is_mc = data_type.lower() == "mc"

            tel_id = re.findall('.*([_\d]+)', telescope)[0]
            tel_id = int(tel_id)

            cleaning_config = config['image_cleaning'][telescope_type]
            bad_pixels_config = config['bad_pixels'][telescope_type]

            if is_mc:
                process_dataset_mc(
                    input_mask=config['data_files'][data_type][sample][telescope]['input_mask'],
                    tel_id=tel_id,
                    cleaning_config=cleaning_config
                )
            else:
                process_dataset_data(
                    input_mask=config['data_files'][data_type][sample][telescope]['input_mask'],
                    tel_id=tel_id,
                    cleaning_config=cleaning_config,
                    bad_pixels_config=bad_pixels_config
                )
