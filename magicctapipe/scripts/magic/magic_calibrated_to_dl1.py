#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import re
import yaml
import copy
from pathlib import Path

import numpy as np

from ctapipe_io_magic import MAGICEventSource

from ctapipe.containers import (
    IntensityStatisticsContainer,
    ImageParametersContainer,
    TimingParametersContainer,
    PeakTimeStatisticsContainer,
)

from ctapipe.coordinates import TelescopeFrame

from ctapipe.io import DataWriter

from ctapipe.image import (
    concentration_parameters,
    descriptive_statistics,
    hillas_parameters,
    morphology_parameters,
    timing_parameters,
)

from magicctapipe.utils import (
    info_message,
)

from magicctapipe.image import (
    MAGICClean,
    get_leakage,
)

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


def magic_calibrated_to_dl1(input_mask, cleaning_config, bad_pixels_config, write_images):
    """Process MAGIC calibrated files, both real and simulation, to get
    DL1 files in the standard CTA format.
    Custom parts:
    - MAGIC cleaning
    - treatment of hot/bad pixels

    Parameters
    ----------
    input_mask : str
        Mask for MC input files. Reading of files is managed
        by the MAGICEventSource class.
    cleaning_config: dict
        Dictionary for cleaning settings
    bad_pixels_config: dict
        Dictionary for bad pixels settings

    Returns
    -------
    None
    """

    aberration_factor = 1./1.0713

    input_files = glob.glob(input_mask)
    input_files.sort()

    clean_config = copy.deepcopy(cleaning_config)

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
        is_simulation = source.is_mc

        geometry_camera_frame = source.subarray.tel[source.telescope].camera.geometry
        geometry = geometry_camera_frame.transform_to(TelescopeFrame())
        #camera_old = source.subarray.tel[source.telescope].camera.geometry
        #camera_refl = reflected_camera_geometry(camera_old)
        #geometry = scale_camera_geometry(camera_refl, aberration_factor)
        if is_simulation:
            clean_config["find_hotpixels"] = False

        magic_clean = MAGICClean(geometry, clean_config)

        info_message("Cleaning configuration", prefix='Hillas')
        for item in vars(magic_clean).items():
            print(f"{item[0]}: {item[1]}")
        if magic_clean.find_hotpixels:
            for item in vars(magic_clean.pixel_treatment).items():
                print(f"{item[0]}: {item[1]}")

        with DataWriter(
            event_source=source,
            output_path=output_name,
            overwrite=True,
            write_images=write_images
        ) as write_data:

            # Looping over the events
            for event in source:
                tels_with_data = event.trigger.tels_with_trigger

                # Looping over the triggered telescopes
                for tel_id in tels_with_data:
                    if is_simulation:
                        event.dl1.tel[tel_id].parameters = DEFAULT_TRUE_IMAGE_PARAMETERS
                    else:
                        event.dl1.tel[tel_id].parameters = DEFAULT_IMAGE_PARAMETERS
                    # Obtained image
                    event_image = event.dl1.tel[tel_id].image
                    # Pixel arrival time map
                    peak_time = event.dl1.tel[tel_id].peak_time

                    if is_simulation:
                        clean_mask, event_image, peak_time = magic_clean.clean_image(event_image, peak_time)
                    else:
                        dead_pixels = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels
                        badrms_pixels = event.mon.tel[tel_id].pixel_status.pedestal_failing_pixels[2]
                        unsuitable_mask = np.logical_or(dead_pixels, badrms_pixels)
                        clean_mask, event_image, peak_time = magic_clean.clean_image(event_image, peak_time, unsuitable_mask=unsuitable_mask)

                    event.dl1.tel[tel_id].image_mask = clean_mask

                    event_image_cleaned = event_image.copy()
                    event_image_cleaned[~clean_mask] = 0

                    event_pulse_time_cleaned = peak_time.copy()
                    event_pulse_time_cleaned[~clean_mask] = 0

                    geom_selected = geometry[clean_mask]
                    image_selected = event_image[clean_mask]

                    if np.any(event_image_cleaned):
                        try:
                            # If event has survived the cleaning, computing the Hillas parameters
                            hillas = hillas_parameters(geom=geom_selected, image=image_selected)
                            leakage = get_leakage(geometry, event_image, clean_mask)
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
arg_parser.add_argument("--write_images",
                        help='Write images in the output file.',
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

            cleaning_config = config['image_cleaning'][telescope_type]
            bad_pixels_config = config['bad_pixels'][telescope_type]

            magic_calibrated_to_dl1(
                input_mask=config['data_files'][data_type][sample][telescope]['input_mask'],
                cleaning_config=cleaning_config,
                bad_pixels_config=bad_pixels_config,
                write_images=parsed_args.write_images,
            )
