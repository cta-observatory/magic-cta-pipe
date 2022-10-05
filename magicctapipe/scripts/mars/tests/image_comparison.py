#!/usr/bin/env python
# coding: utf-8

import yaml
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from ctapipe_io_magic import MAGICEventSource
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry, CameraDescription
from ctapipe.io.eventseeker import EventSeeker
from magicctapipe.image import MAGICClean
from magicctapipe.scripts.mars import read_images


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


# set configurations for the cleaning
cleaning_config = dict(
    picture_thresh=6.0,
    boundary_thresh=3.5,
    max_time_off=4.5,
    max_time_diff=1.5,
    use_time=True,
    use_sum=True,
    find_hotpixels=True,
)

bad_pixels_config = dict(
    pedestalLevel=400, pedestalLevelVariance=4.5, pedestalType="FromExtractorRndm"
)


# define the image comparison function
def image_comparison(
    config_file="config.yaml", mode="use_ids_config", tel_id=1, max_events=None
):
    """
    This tool compares the camera images of events processed by MARS and the magic-cta-pipeline.
    The output is a png file with the camera images and a hdf5 file that contains the pixel information.
    The function returns a list with the Event IDs for events, where there's a difference between the images.
    ---
    config_file: Configuration file
    mode:
        use_all: use all events in the given input file
        use_ids_config: use events, whose ids are given in the config file
    tel_id: Telescope of which the data will be compared. use "1" for M1 and "2" for M2
    """

    # load config file--------------------------------------------------------------------------------------------
    config = yaml.safe_load(open(config_file, "r"))
    out_path = config["output_files"]["file_path"]
    Path(out_path).mkdir(exist_ok=True, parents=True)
    comparison = []

    # get id to compare from config file---------------------------------------------------------------------------
    if mode == "use_all":
        with uproot.open(
            config["input_files"]["magic_cta_pipe"][f"M{tel_id}"]
        ) as mcp_file:
            ids_to_compare = mcp_file["Events"]["MRawEvtHeader.fStereoEvtNumber"].array(
                library="np"
            )
            ids_to_compare = np.delete(ids_to_compare, np.where(ids_to_compare == 0)[0])
            ids_to_compare = ids_to_compare.tolist()
    elif mode == "use_ids_config":
        ids_to_compare = config["event_list"]

    if max_events is not None:
        ids_to_compare = ids_to_compare[:max_events]

    print(len(ids_to_compare), "events will be compared", ids_to_compare)

    # we will now load the data files, and afterwards select the corresponding data for our events
    # get mars data ------------------------------------------------------------------------------------------------------
    mars_input = config["input_files"]["mars"]
    image = []  # pixel charges
    events = []  # event ids
    telescope = []  # telescope id
    obs_id = []  # run number

    for image_container in read_images(mars_input, read_calibrated=True):
        events.append(image_container.event_id)
        telescope.append(image_container.tel_id)
        obs_id.append(image_container.obs_id)
        image.append(image_container.image_cleaned)

    mars_data = pd.DataFrame(
        list(zip(events, telescope, obs_id, image)),
        columns=["event_id", "tel_id", "obs_id", "image"],
    )

    # get other data----------------------------------------------------------------------------------------------------
    # we loop through the events, and only compare those that are in ids_to_compare

    source = MAGICEventSource(
        input_url=config["input_files"]["magic_cta_pipe"][f"M{tel_id}"],
        process_run=False,
    )

    run_num = source.obs_ids[0]

    seeker = EventSeeker(event_source=source)

    MAGICCAM = CameraDescription.from_name("MAGICCam")
    GEOM = MAGICCAM.geometry
    geometry_mars = new_camera_geometry(GEOM)

    geometry_old = source.subarray.tel[tel_id].camera.geometry
    geometry_mcp = new_camera_geometry(geometry_old)
    geom = CameraGeometry.from_name("MAGICCamMars")

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    fig.set_figheight(20)
    fig.set_figwidth(40)

    disp1 = CameraDisplay(geom, ax=ax1)
    disp_mars = CameraDisplay(geometry_mars, ax=ax2)
    disp_mcp = CameraDisplay(geometry_mcp, ax=ax3)
    disp2 = CameraDisplay(geom, ax=ax4)
    disp3 = CameraDisplay(geom, ax=ax5)
    disp4 = CameraDisplay(geom, ax=ax6)
    disp1.add_colorbar(label="pixel charge")
    disp2.add_colorbar(label="pixel charge")
    disp3.add_colorbar(label="pixel charge")
    disp4.add_colorbar(label="pixel charge")
    disp_mars.add_colorbar(label="pixel charge")
    disp_mcp.add_colorbar(label="pixel charge")

    if source.is_simulation:
        cleaning_config.update({"find_hotpixels": False})
    else:
        cleaning_config.update({"find_hotpixels": True})

    for k, v in cleaning_config.items():
        print(f"{k} : {v}")

    for event_id in ids_to_compare:
        try:
            event = seeker.get_event_id(event_id)
        except IndexError:
            print(f"Event with ID {event_id} not found in calibrated file. Skipping...")
            continue

        print("Event ID:", event.index.event_id, "- Telecope ID:", tel_id)

        # mars data -----------------------------------------------------------------------------
        mars_event = mars_data.loc[
            (mars_data["event_id"] == event.index.event_id)
            & (mars_data["tel_id"] == tel_id)
        ]
        if mars_event.empty:
            print(f"Event with ID {event_id} not found in MARS file. Skipping...")
            continue
        idx = mars_event["image"].index[0]
        event_image_mars = np.array(mars_event["image"][idx][:1039])
        clean_mask_mars = event_image_mars != 0

        # get mcp data------------------------------------------------------------------------------

        magic_clean = MAGICClean(geometry_mcp, cleaning_config)
        original_data_images = event.dl1.tel[tel_id].image
        original_data_images_copy = original_data_images.copy()
        event_pulse_time = event.dl1.tel[tel_id].peak_time

        neighbors_array = geometry_mcp.neighbor_matrix

        for pix in np.where(neighbors_array[138] is True)[0]:
            print(
                f"{pix} pixel charge before cleaning: {original_data_images_copy[pix]}"
            )

        if not source.is_simulation:
            badrmspixel_mask = event.mon.tel[
                tel_id
            ].pixel_status.pedestal_failing_pixels[2]
            deadpixel_mask = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[
                0
            ]
            unsuitable_mask = np.logical_or(badrmspixel_mask, deadpixel_mask)

            (
                clean_mask,
                calibrated_data_images,
                event_pulse_time,
            ) = magic_clean.clean_image(
                original_data_images_copy,
                event_pulse_time,
                unsuitable_mask=unsuitable_mask,
            )
        else:
            (
                clean_mask,
                calibrated_data_images,
                event_pulse_time,
            ) = magic_clean.clean_image(original_data_images_copy, event_pulse_time)

        # print(f"Pixels selected after 1st step ({len(np.where(magic_clean.mask_step1 == True)[0])} pixels): {np.where(magic_clean.mask_step1 == True)[0]}")
        # print(f"Pixels selected after 2nd step ({len(np.where(magic_clean.mask_step2 == True)[0])} pixels): {np.where(magic_clean.mask_step2 == True)[0]}")
        # print(f"Pixels selected after 3rd step ({len(np.where(magic_clean.mask_step3 == True)[0])} pixels): {np.where(magic_clean.mask_step3 == True)[0]}")

        event_image_mcp = calibrated_data_images.copy()
        event_image_mcp[~clean_mask] = 0

        # clipping for charges > 750
        event_image_mcp[event_image_mcp >= 750.0] = 750.0

        if not np.any(event_image_mcp):
            print(
                f"Event ID {event.index.event_id} for telescope {tel_id} does not have any surviving pixel. Skipping..."
            )
            continue

        # get max value for colorbar--------------------------------------------------------------------------------
        mcp_max = np.amax(event_image_mcp[clean_mask])
        mars_max = np.amax(event_image_mars[clean_mask_mars])
        if mcp_max >= mars_max:
            vmax = mcp_max
        else:
            vmax = mars_max

        vmin = 0
        # print(vmax)

        # find differences------------------------------------------------------------------------------------------
        charge_differences = abs(event_image_mars - event_image_mcp)
        clean_mask_pixels = charge_differences != 0

        print(np.where(clean_mask_pixels == True)[0])

        if len(np.where(clean_mask_pixels == True)[0]) == 0:
            errors = False
        else:
            if np.any(
                charge_differences >= 0.00000001 * vmax
            ):  # threshold for differences allowed. The difference cannot be higher than x% of the highest pixel charge
                errors = True

        if errors:
            comparison.append(event.index.event_id)
            print(f"errors found for {event.index.event_id}!")

        # differences between calibrated data before cleaning and mcp/mars data
        pix_diff_mars = abs(event_image_mars - calibrated_data_images)
        pix_diff_mcp = abs(event_image_mcp - calibrated_data_images)

        # create output file that contains the pixel values and differences------------------------------------------------

        df_pixel = pd.DataFrame()
        data = np.transpose(
            np.array([event_image_mars, event_image_mcp, calibrated_data_images])
        )

        df_pixel = pd.DataFrame(
            data, columns=["MARS_charge", "mcp_charge", "calibrated_data"]
        )
        df_pixel["difference_MARS-mcp"] = np.transpose(charge_differences)
        df_pixel["relative_error_MARS-mcp"] = np.transpose(
            charge_differences / event_image_mcp
        )
        # alternatively one could also calculate the relative error with the MARS charge value
        # df_pixel["relative_error_MARS-mcp"] = np.transpose(charge_differences/event_image_mars)
        # df_pix_diff = df_pixel.loc[df_pixel["difference_MARS-mcp"] > 0]
        # print(df_pix_diff)

        # saving the output
        # the file only gets saved if there are differences between the images
        if config["save_only_when_differences"] == True:
            if any(charge_differences) == True:
                df_pixel.to_hdf(
                    f"{run_num}_image_comparison.h5",
                    f"/{event.index.event_id}_M{tel_id}",
                    "a",
                )
                # df_pix_diff.to_hdf(f"{out_path}{run_num}_{id_event}_M{telescope_id}_pixel_diff.h5", "/pixel_differences", "w")

        # the file gets saved in any case
        elif config["save_only_when_differences"] == False:
            df_pixel.to_hdf(
                f"{run_num}_image_comparison.h5",
                f"/{event.index.event_id}_M{tel_id}",
                "a",
            )

        # plotting ------------------------------------------------------------------------------------------------------
        grid_shape = (2, 3)

        # original data
        disp1.image = original_data_images
        # pixels whose original value is negative
        # negative_mask = calibrated_data_images < 0
        disp1.set_limits_minmax(vmin, vmax)
        ax1.set_title("original data")

        # mars_data
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        disp_mars.image = event_image_mars
        disp_mars.set_limits_minmax(vmin, vmax)
        disp_mars.highlight_pixels(
            clean_mask_pixels[:1039], color="red", alpha=1, linewidth=1
        )
        # disp_mars.highlight_pixels(unsuitable_mask[:1039], color="magenta", alpha=1, linewidth=1)
        ax2.set_title("MARS data")

        # mcp_data
        disp_mcp.image = event_image_mcp
        disp_mcp.set_limits_minmax(vmin, vmax)
        disp_mcp.highlight_pixels(
            clean_mask_pixels[:1039], color="red", alpha=1, linewidth=1
        )
        ax3.set_title("magic_cta_pipe data")

        # differences between MARS and mcp
        disp2.image = charge_differences[:1039]
        disp2.highlight_pixels(
            clean_mask_pixels[:1039], color="red", alpha=1, linewidth=1
        )
        ax4.set_title("differences MARS-mcp")

        # the white outline shows the pixels used for the image after cleaning, the ones that are filled yellow show where differences are
        # differences between MARS and the calibrated data
        pix_diff_mars_copy = np.array(pix_diff_mars).copy()
        pix_diff_mars_copy[np.array(event_image_mars) == 0] = 0
        disp3.image = pix_diff_mars_copy > 0
        disp3.highlight_pixels(
            np.array(event_image_mars) != 0, color="white", alpha=1, linewidth=3
        )
        ax5.set_title("differences MARS-original data")

        # mcp-orig-data-diff
        pix_diff_mcp_copy = np.array(pix_diff_mcp).copy()
        pix_diff_mcp_copy[np.array(event_image_mcp) == 0] = 0
        disp4.image = pix_diff_mcp_copy > 0
        disp4.highlight_pixels(
            np.array(event_image_mcp) != 0, color="white", alpha=1, linewidth=3
        )
        ax6.set_title("differences mcp-original data")

        fig.suptitle(
            f"Comparison_MARS_magic-cta-pipe: Event ID {event.index.event_id}, {run_num}, M{tel_id}",
            fontsize=16,
        )
        plt.savefig(
            f"{out_path}/image-comparison-{run_num}_{event.index.event_id}_M{tel_id}.png"
        )
        # print(f"{out_path}/image-comparison-{run_num}_{event.index.event_id}_M{tel_id}.png")
        # fig.savefig(f"{out_path}/image-comparison-{run_num}_{event.index.event_id}_M{tel_id}.pdf")

    return comparison
