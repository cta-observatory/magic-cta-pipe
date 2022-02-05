#!/usr/bin/env python
# coding: utf-8

import yaml
import argparse
import sys
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from ctapipe_io_magic import MAGICEventSource
from ctapipe.io import EventSeeker
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from ctapipe.image import hillas_parameters
from ctapipe.image.timing import timing_parameters
from magicctapipe.utils import MAGIC_Badpixels, MAGIC_Cleaning

from ctapipe.image.morphology import number_of_islands
from magicctapipe.scripts import read_images


# define camera geometry
def new_camera_geometry(camera_geom):
    return CameraGeometry(
        camera_name="MAGICCam",
        pix_id=camera_geom.pix_id,
        pix_x=-1.0 * camera_geom.pix_y,
        pix_y=-1.0 * camera_geom.pix_x,
        pix_area=camera_geom.guess_pixel_area(
            camera_geom.pix_x, camera_geom.pix_y, camera_geom.pix_type
        ),
        pix_type=camera_geom.pix_type,
        pix_rotation=camera_geom.pix_rotation,
        cam_rotation=camera_geom.cam_rotation,
    )


def parse_args(args):
    """
    Parse command line options and arguments.
    """

    parser = argparse.ArgumentParser(description="", prefix_chars='-')
    parser.add_argument("-c", "--config", nargs='?', help="Path to configuration file.")
    parser.add_argument(
        "-m",
        "--mode",
        nargs='?',
        choices=["useall", "use_ids_config"],
        default="use_ids_config",
        help='Mode of comparison: "all" to compare all events, "use_ids_config" to compare specific event ids.'
        )

    return parser.parse_args(args)


def image_comparison(config_file="config.yaml", mode="use_ids_config"):
    """
    This tool compares the camera images of events processed by MARS and the magic-cta-pipeline.
    The output is a png file with the camera images and a hdf5 file that contains the pixel information.
    ---
    config_file: Configuration file
    mode:
        use_all: use all events in the given input file
        use_ids_config: use events, whose ids are given in the config file
    """
    config = yaml.safe_load(open(config_file, "r"))

    # errors?

    ids_to_compare = []

    magic_calibrated_file = config["input_files"]["mars_calibrated"]

    if mode == "useall":
        with uproot.open(magic_calibrated_file) as mcp_file:
            trigger_pattern = mcp_file["Events"]['MTriggerPattern.fPrescaled'].array(library="np")
            ids_to_compare = mcp_file["Events"][
                    "MRawEvtHeader.fStereoEvtNumber"
                ].array(library="np")
            ids_to_compare = ids_to_compare[np.where(trigger_pattern == 128)].tolist()
            # the keys depend on the file, there may be a case where they have to be changed

    elif mode == "use_ids_config":
        ids_to_compare = config["event_list"]

    if len(ids_to_compare) == 0:
        print("List with event ids to compare is empty. Exiting.")
        sys.exit(1)

    print(len(ids_to_compare), "events will be compared")

    # get information from the config file
    telescope_id = config["information"]["tel_id"]
    run_num = config["information"]["run_number"]
    out_path = config["output_files"]["file_path"]

    mars_input = config["input_files"]["mars_images"]

    image_mars = []  # pixel charges
    events_mars = []  # event ids
    telescope = []  # telescope id
    runnum_mars = []  # run number

    # get data from HDF5 containing calibrated and cleaned images

    for image_container in read_images(mars_input, read_calibrated=True, allowed_tels=[telescope_id]):
        events_mars.append(image_container.event_id)
        telescope.append(image_container.tel_id)
        runnum_mars.append(image_container.obs_id)
        image_mars.append(image_container.image_cleaned)

    image_mars = np.array(image_mars)
    stereo_id_mars = np.array(events_mars)

    # prepare event source from calibrated file

    source = MAGICEventSource(
        input_url=magic_calibrated_file,
        process_run=False,
    )
    if source.is_simulation:
        find_hot_pixels = False
    else:
        find_hot_pixels = True
    # check for correct values!!
    cleaning_config = dict(
        picture_thresh=6,
        boundary_thresh=3.5,
        max_time_off=4.5 * 1.64,
        max_time_diff=1.5 * 1.64,
        usetime=True,
        usesum=True,
        findhotpixels=find_hot_pixels,
    )

    bad_pixels_config = dict(
        pedestalLevel=400,
        pedestalLevelVariance=4.5,
        pedestalType="FromExtractorRndm",
    )

    tel_id = source.telescope
    geometry_old = source.subarray.tel[tel_id].camera.geometry
    geometry_mars = new_camera_geometry(geometry_old)
    magic_clean = MAGIC_Cleaning.magic_clean(
        geometry_mars, cleaning_config
    )
    badpixel_calculator = MAGIC_Badpixels.MAGICBadPixelsCalc(
        config=bad_pixels_config, is_simulation=source.is_simulation
    )

    seeker = EventSeeker(event_source=source)

    for id_event in ids_to_compare:
        print("Processing Event ID:", id_event)
        # ------------------
        #      MARS
        # -----------------

        # get image from MARS, if present (if it did not pass MARS
        # cleaning, it will be in the calibrated files but not
        # in the HDF5 file with the images)

        id_index_list = np.where(stereo_id_mars == id_event)[0]
        if len(id_index_list) == 0:
            print(f"Skipping Event ID {id_event}, not present in MARS image file")
            continue

        id_index = id_index_list[0]  # stereo id is supposed to be unique in file

        event_image_mars = np.array(image_mars[id_index][:1039])
        clean_mask_mars = event_image_mars != 0
        print(f"Image cleaned MARS: {event_image_mars[clean_mask_mars]}")

        num_islands_mars, island_labels_mars = number_of_islands(
            geometry_mars, (np.array(event_image_mars[:1039]) > 0)
        )
        # print(num_islands_mars)

        # -----------------------------
        #       magic-cta-pipe
        # -----------------------------
        #
        # now get the corresponding event from the calibrated file with
        # EventSeeker, and apply the cleaning

        event = seeker.get_event_id(id_event)

        if event is None:
            print(f"Skipping Event ID {id_event}, not present in calibrated file.")
            continue

        event_image = event.dl1.tel[tel_id].image
        calibrated_data_image = event_image.copy()
        event_pulse_time = event.dl1.tel[tel_id].peak_time

        badrmspixel_mask = badpixel_calculator.get_badrmspixel_mask(
            event
        )

        deadpixel_mask = badpixel_calculator.get_deadpixel_mask(event)
        unsuitable_mask = np.logical_or(
            badrmspixel_mask[tel_id - 1], deadpixel_mask[tel_id - 1]
        )

        try:
            bad_pixel_indices = np.where(badrmspixel_mask[telescope_id - 1] is True)[0][0]
        except IndexError:
            bad_pixel_indices = np.array([])
        print("badpixel_indices:", bad_pixel_indices)
        try:
            dead_pixel_indices = np.where(deadpixel_mask[telescope_id - 1] is True)[0][0]
        except IndexError:
            dead_pixel_indices = np.array([])
        print("deadpixel_indices:", dead_pixel_indices)
        bad_not_dead_pixels = np.setdiff1d(bad_pixel_indices, dead_pixel_indices)
        print("bad but not dead pixels:", bad_not_dead_pixels)

        (
            clean_mask,
            event_image,
            event_pulse_time,
        ) = magic_clean.clean_image(
            event_image,
            event_pulse_time,
            unsuitable_mask=unsuitable_mask,
        )

        event_image_cleaned = event_image.copy()
        event_image_cleaned[~clean_mask] = 0
        # print(event_image_cleaned[clean_mask])

        event_pulse_time_cleaned = event_pulse_time.copy()
        event_pulse_time_cleaned[~clean_mask] = 0

        if np.any(event_image_cleaned):
            hillas_params = hillas_parameters(
                geometry_mars, event_image_cleaned
            )
            image_mask = event_image_cleaned > 0
            timing_params = timing_parameters(
                geometry_mars,
                event_image_cleaned,
                event_pulse_time_cleaned,
                hillas_params,
                image_mask,
            )
        else:
            print("Cleaning failed.")
            continue

        # compare number of islands
        num_islands_mcp, island_labels_mcp = number_of_islands(
            geometry_mars, (np.array(event_image_cleaned[:1039]) > 0)
        )

        mcp_max = np.amax(event_image_cleaned[clean_mask])
        mars_max = np.amax(event_image_mars[clean_mask_mars])
        if mcp_max >= mars_max:
            vmax = mcp_max
        else:
            vmax = mars_max

        vmin = 0

        # ---------------------------------------------
        # find pixel differences and write h5 output file
        # -------------------------------------------------

        pix_diff = np.absolute(event_image_mars - event_image_cleaned)

        comparison = []
        errors = False
        # threshold for differences allowed
        tolerance = 0.01
        if np.any(np.where(pix_diff > tolerance)[0]):
            print(f"There are pixels wich differ more than {tolerance}")
            errors = True

        # if errors == True:
        if errors:
            comparison.append(event.index.event_id)

        # if you want negative values instead of the absolutes
        # pix_diff = pix_mars - pix_mcp
        # pix_diff = abs(pix_mars - pix_mcp[0])
        pix_image = pix_diff
        clean_mask_pixels = pix_image != 0

        # differences between outputs from MARS and mcp and input calibrated data
        pix_diff_mars = np.absolute(event_image_mars - calibrated_data_image)
        pix_diff_mcp = np.absolute(event_image_cleaned - calibrated_data_image)
        print(pix_diff_mars)
        # print(calibrated_data_image)

        data_value = []

        for i in range(1039):
            data_value.append(calibrated_data_image[i])

        df_pixel = pd.DataFrame()
        data = np.array([event_image_mars, event_image_cleaned])
        data_real = np.transpose(data)

        df_pixel = pd.DataFrame(
            data_real, columns=["MARS charge", "mcp charge"]
        )
        df_pixel["calibrated data"] = np.transpose(
            data_value
        )  # (calibrated_data_image[:1039])
        df_pixel["difference MARS mcp"] = np.transpose(pix_diff)
        df_pix_diff = df_pixel.loc[df_pixel["difference MARS mcp"] > 0]
        pix_diff_ids = df_pix_diff.index.tolist()
        print(df_pix_diff)

        # saving the output
        if config["save_only_when_differences"] is True:
            # the file only gets saved if there are differences between the images
            if errors:
                print("Differences found. Saving files!")
                df_pixel.to_hdf(
                    f"{out_path}/{run_num}_{id_event}_M{telescope_id}_pixel_info.h5",
                    "/pixel_information",
                    "w",
                )
                df_pix_diff.to_hdf(
                    f"{out_path}/{run_num}_{id_event}_M{telescope_id}_pixel_diff.h5",
                    "/pixel_differences",
                    "w",
                )
                print(pix_diff_ids)

            else:
                print("No differences found. No files will be saved!")
                continue

        elif config["save_only_when_differences"] is False:
            # the file gets saved in any case
            df_pixel.to_hdf(
                f"{out_path}/{run_num}_{id_event}_M{telescope_id}_pixel_info.h5",
                "/pixel_information",
                "w",
            )
            df_pix_diff.to_hdf(
                f"{out_path}/{run_num}_{id_event}_M{telescope_id}_pixel_diff.h5",
                "/pixel_differences",
                "w",
            )
            print(pix_diff_ids)
        else:
            print("No criteria for saving data specified. Exiting.")
            exit()

        # --------------------------------
        #  creating output image
        # ----------------------------
        fig = plt.figure(figsize=(20, 10))
        grid_shape = (2, 3)

        # original data
        plt.subplot2grid(grid_shape, (0, 0))

        disp = CameraDisplay(geometry_mars, calibrated_data_image)
        # pixels whose original value is negative
        negative_mask = calibrated_data_image < 0
        disp.highlight_pixels(
            negative_mask, color="white", alpha=1, linewidth=1
        )
        disp.add_colorbar(label="pixel charge")
        disp.set_limits_minmax(vmin, vmax)

        plt.title("original data")

        # mars_data
        plt.subplot2grid(grid_shape, (0, 1))

        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        disp_mars = CameraDisplay(geometry_mars, event_image_mars[:1039])
        # disp_mars.highlight_pixels(clean_mask, color='white', alpha=0.5, linewidth=1)
        disp_mars.set_limits_minmax(vmin, vmax)
        disp_mars.add_colorbar(label="pixel charge")
        disp_mars.highlight_pixels(
            clean_mask_pixels, color="red", alpha=1, linewidth=1
        )

        plt.title("MARS data")

        # mcp_data
        plt.subplot2grid(grid_shape, (0, 2))

        disp_mcp = CameraDisplay(geometry_mars, image=event_image_cleaned)
        # disp_mcp.highlight_pixels(clean_mask, color='white', alpha=0.5, linewidth=1)
        disp_mcp.add_colorbar(label="pixel charge")
        disp_mcp.set_limits_minmax(vmin, vmax)
        disp_mcp.highlight_pixels(
            clean_mask_pixels, color="red", alpha=1, linewidth=1
        )

        plt.title("magic_cta_pipe data")

        # differences between MARS and mcp
        plt.subplot2grid(grid_shape, (1, 0))

        disp = CameraDisplay(geometry_mars, pix_image[:1039])
        disp.add_colorbar(label="pixel charge")
        disp.highlight_pixels(
            clean_mask_pixels, color="red", alpha=1, linewidth=1
        )

        # alternative image: used pixels after the cleaning and differences mask
        """for id in range(len(image)):
           if stereo_id[id] not in ids:
                continue
            event_image = np.array(image[id][:1039])
            clean_mask = event_image != 0
            #print(event_image[clean_mask])
            norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
            disp = CameraDisplay(geometry_mars, image=event_image)
            disp.highlight_pixels(clean_mask_pixels, color='red', alpha=0.5, linewidth=2)
            disp.set_limits_minmax(vmin, 1)"""

        plt.title("differences MARS-mcp")

        # differences between MARS and the calibrated data
        plt.subplot2grid(grid_shape, (1, 1))

        pix_diff_mars_copy = np.array(pix_diff_mars).copy()
        pix_diff_mars_copy[np.array(event_image_mars[:1039]) == 0] = 0

        disp = CameraDisplay(geometry_mars, pix_diff_mars_copy > 0)
        # disp.highlight_pixels(clean_mask_pixels, color='red', alpha=1, linewidth=1)
        disp.highlight_pixels(
            np.array(event_image_mars[:1039]) != 0,
            color="white",
            alpha=1,
            linewidth=3,
        )
        # disp.set_limits_minmax(vmin, vmax)
        # disp.add_colorbar(label='pixel charge')

        plt.title("differences MARS-original data")

        # mcp-orig-data-diff
        plt.subplot2grid(grid_shape, (1, 2))

        pix_diff_mcp_copy = np.array(pix_diff_mcp).copy()
        pix_diff_mcp_copy[np.array(event_image_cleaned) == 0] = 0

        disp = CameraDisplay(geometry_mars, pix_diff_mcp_copy > 0)
        disp.highlight_pixels(
            np.array(event_image_cleaned) != 0,
            color="white",
            alpha=1,
            linewidth=3,
        )
        # disp.set_limits_minmax(vmin, vmax)
        # disp.add_colorbar(label='pixel charge')
        plt.title("differences mcp-original data")

        fig.suptitle(
            f"Comparison_MARS_magic-cta-pipe: Event ID {id_event}, {run_num}, M{tel_id}",
            fontsize=16,
        )
        fig.savefig(
            f"{out_path}/data-comparison-{run_num}_{id_event}_M{tel_id}.png"
        )
        # fig.savefig(f"{out_path}/data-comparison-{run_num}_{id_event}_M{tel_id}.pdf")

        # ids.clear()
    # array das einem anzeigt ob pixel differences bestehen!
    return comparison


def main(*args):
    flags = parse_args(args)

    config = flags.config
    mode = flags.mode

    image_comparison(
        config_file=config,
        mode=mode
    )


if __name__ == '__main__':
    main(*sys.argv[1:])
