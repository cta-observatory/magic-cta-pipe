# coding: utf-8

import os
import time
import copy
import glob
import scipy
import select
import argparse
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord

from ctapipe.io import SimTelEventSource
from ctapipe.io import HDF5TableWriter
from ctapipe.calib import CameraCalibrator
from ctapipe.coordinates import TelescopeFrame
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.morphology import number_of_islands

from ctapipe.reco import HillasReconstructor
from ctapipe.containers import ImageParametersContainer
from ctapipe.visualization import ArrayDisplay

from magicctapipe.image import MAGICClean
from magicctapipe.utils import calculate_impact
from magicctapipe.utils.filedir import *
from magicctapipe.utils.utils import *
from magicctapipe.utils.tels import *
from magicctapipe.reco.stereo import *
from magicctapipe.reco.image import *


PARSER = argparse.ArgumentParser(
    description="Stereo Reconstruction MAGIC + LST",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
PARSER.add_argument(
    "-cfg",
    "--config_file",
    type=str,
    required=True,
    help="Configuration file, yaml format",
)
PARSER.add_argument(
    "-mtr",
    "--only_mc_train",
    action="store_true",
    required=False,
    default=False,
    help="Consider only mc train files",
)
PARSER.add_argument(
    "-mte",
    "--only_mc_test",
    action="store_true",
    required=False,
    default=False,
    help="Consider only mc test files",
)
PARSER.add_argument(
    "-dtr",
    "--only_data_train",
    action="store_true",
    required=False,
    default=False,
    help="Consider only data train files",
)
PARSER.add_argument(
    "-dte",
    "--only_data_test",
    action="store_true",
    required=False,
    default=False,
    help="Consider only data test files",
)
PARSER.add_argument(
    "-d",
    "--display",
    action="store_true",
    required=False,
    default=False,
    help="Display plots",
)


def call_stereo_reco_MAGIC_LST(kwargs):
    """Stereo Reconstruction for MAGIC and/or LST array, looping on all given data

    Parameters
    ----------
    kwargs : dict
        parser options
    """
    print_title("Stereo Reconstruction")

    cfg = load_cfg_file(kwargs["config_file"])

    if kwargs["only_mc_train"]:
        k1, k2 = ["mc"], ["train_sample"]
    elif kwargs["only_mc_test"]:
        k1, k2 = ["mc"], ["test_sample"]
    elif kwargs["only_data_train"]:
        k1, k2 = ["data"], ["train_sample"]
    elif kwargs["only_data_test"]:
        k1, k2 = ["data"], ["test_sample"]
    else:
        k1 = ["mc", "data"]
        k2 = ["train_sample", "test_sample"]

    for k1_ in k1:
        for k2_ in k2:
            stereo_reco_MAGIC_LST(k1=k1_, k2=k2_, cfg=cfg, display=kwargs["display"])


def stereo_reco_MAGIC_LST(k1, k2, cfg, display=False):
    """Stereo Reconstruction for MAGIC and/or LST array

    Parameters
    ----------
    k1 : str
        first key in `cfg["data_files"][k1][k2]`
    k2 : str
        second key in `cfg["data_files"][k1][k2]`
    cfg: dict
        configurations loaded from configuration file
    display : bool, optional
        display plots, by default False
    """

    tel_ids, tel_ids_LST, tel_ids_MAGIC = check_tel_ids(cfg)
    if len(tel_ids) < 2:
        print("Select at least two telescopes in the MAGIC + LST array")
        return

    consider_LST = len(tel_ids_LST) > 0
    consider_MAGIC = len(tel_ids_MAGIC) > 0

    if consider_MAGIC:
        use_MARS_cleaning = cfg["MAGIC"].get("use_MARS_cleaning", True)

    file_list = glob.glob(cfg["data_files"][k1][k2]["mask_sim"])
    print(file_list)

    previous_event_id = 0

    # --- Loop on files ---
    for file in file_list:
        print(f"Analyzing file:\n{file}")

        # --- Output file DL1 ---
        out_file = os.path.join(
            os.path.dirname(cfg["data_files"][k1][k2]["hillas_h5"]),
            out_file_h5(in_file=file),
        )
        print(f"Output file:\n{out_file}")
        check_folder(os.path.dirname(out_file))
        # Init the writer for the output file
        writer = HDF5TableWriter(filename=out_file, group_name="dl1", overwrite=True)

        # --- Open simtel file ---
        if "max_events_run" in cfg["all_tels"]:
            max_events_run = cfg["all_tels"]["max_events_run"]
        else:
            max_events_run = 0  # I read all events in the file

        # Init source
        source = SimTelEventSource(file, max_events=max_events_run)

        # Init calibrator, both for MAGIC and LST
        calibrator = CameraCalibrator(subarray=source.subarray)

        hillas_reconstructor = HillasReconstructor(source.subarray)
        tel_positions = source.subarray.positions

        # Init MAGIC MARS cleaning, if selected
        if consider_MAGIC and use_MARS_cleaning:
            magic_clean = MAGICClean(
                camera=source.subarray.tel[tel_ids_MAGIC[0]].camera.geometry,
                configuration=cfg["MAGIC"]["cleaning_config"]
            )

        # --- Write MC HEADER ---
        # Problem: impossible to write/read with the following function a
        # list, so we assign to run_array_direction an empty list, in order to
        # make the sotware NOT write the run_array_direction
        source.mc_header.run_array_direction = []  # dummy value
        writer.write("mc_header", source.mc_header)

        if display:
            fig, ax = plt.subplots()
            go, first_time_display, cont = True, True, False

        # --- Loop on events in source ---
        for event in source:
            if previous_event_id == event.index.event_id:
                continue
            previous_event_id = copy.copy(event.index.event_id)

            if display and go:
                print("Event %d" % event.count)
            elif event.count % 100 == 0:
                print("Event %d" % event.count)

            # Process only if I have at least two tel_ids of the selected array
            # sel_tels: selected telescopes with data in the event
            sel_tels = list(set(event.r0.tels_with_data).intersection(tel_ids))
            if len(sel_tels) < 2:
                continue

            # Inits
            hillas_p, leakage_p, timing_p = {}, {}, {}
            telescope_pointings, time_grad, event_info = {}, {}, {}
            failed = False

            # --- Calibrate ---
            # Call the calibrator, both for MAGIC and LST
            calibrator(event)

            # Loop on triggered telescopes
            for tel_id, dl1 in event.dl1.tel.items():
                # Exclude telescopes not selected
                if tel_id not in sel_tels:
                    continue
                try:
                    geom_camera_frame = source.subarray.tels[tel_id].camera.geometry
                    geom = geom_camera_frame.transform_to(TelescopeFrame())
                    image = dl1.image  # == event_image
                    peakpos = dl1.peak_time  # == event_pulse_time

                    # --- Cleaning ---
                    if geom.camera_name == cfg["LST"]["camera_name"]:
                        # Apply tailcuts clean on LST. From ctapipe
                        clean = tailcuts_clean(
                            geom=geom, image=image, **cfg["LST"]["cleaning_config"]
                        )
                        # Ignore if less than n pixels after cleaning
                        if clean.sum() < cfg["LST"]["min_pixel"]:
                            continue
                        # Number of islands: LST. From ctapipe
                        num_islands, island_ids = number_of_islands(
                            geom=geom, mask=clean
                        )
                    elif (
                        geom.camera_name == cfg["MAGIC"]["camera_name"]
                        and not use_MARS_cleaning
                    ):
                        # Apply tailcuts clean on MAGIC. From ctapipe
                        clean = tailcuts_clean(
                            geom=geom,
                            image=image,
                            **cfg["MAGIC"]["cleaning_config_ctapipe"],
                        )
                        # Ignore if less than n pixels after cleaning
                        if clean.sum() < cfg["MAGIC"]["min_pixel"]:
                            continue
                        # Number of islands: LST. From ctapipe
                        num_islands, island_ids = number_of_islands(
                            geom=geom, mask=clean
                        )
                    elif (
                        geom.camera_name == cfg["MAGIC"]["camera_name"]
                        and use_MARS_cleaning
                    ):
                        # Apply MAGIC MARS cleaning. From magic-cta-pipe
                        clean, image, peakpos = magic_clean.clean_image(
                            event_image=image, event_pulse_time=peakpos
                        )
                        # Ignore if less than n pixels after cleaning
                        if clean.sum() < cfg["MAGIC"]["min_pixel"]:
                            continue
                        # Number of islands: MAGIC. From magic-cta-pipe
                        num_islands = get_num_islands_MAGIC(
                            camera=geom, clean_mask=clean, event_image=image
                        )
                    else:
                        continue
                    # --- Analize cleaned image ---
                    # Evaluate Hillas, leakeage, timing
                    (
                        hillas_p[tel_id],
                        leakage_p[tel_id],
                        timing_p[tel_id],
                    ) = clean_image_params(
                        geom=geom, image=image, clean=clean, peakpos=peakpos
                    )
                    # Get time gradients
                    time_grad[tel_id] = timing_p[tel_id].slope.value

                    if "cuts" in cfg["all_tels"]:
                        cuts = cfg["all_tels"]["cuts"]
                        if (
                            (hillas_p[tel_id].intensity < cuts["intensity_low"])
                            or (hillas_p[tel_id].intensity > cuts["intensity_high"])
                            or (hillas_p[tel_id].length < cuts["length_low"])
                            or (
                                leakage_p[tel_id].intensity_width_1
                                > cuts["intensity_width_1_high"]
                            )
                        ):
                            hillas_p.pop(tel_id, None)
                            leakage_p.pop(tel_id, None)
                            timing_p.pop(tel_id, None)
                            continue

                    # Evaluate telescope pointings
                    telescope_pointings[tel_id] = SkyCoord(
                        alt=event.pointing.tel[tel_id].altitude,
                        az=event.pointing.tel[tel_id].azimuth,
                        frame=horizon_frame,
                    )

                    # Preparing metadata
                    event_info[tel_id] = StereoInfoContainer(
                        obs_id=event.index.obs_id,
                        event_id=scipy.int32(event.index.event_id),
                        tel_id=tel_id,
                        true_energy=event.mc.energy,
                        true_alt=event.mc.alt.to(u.rad),
                        true_az=event.mc.az.to(u.rad),
                        tel_alt=event.pointing.tel[tel_id].altitude.to(u.rad),
                        tel_az=event.pointing.tel[tel_id].azimuth.to(u.rad),
                        num_islands=num_islands,
                    )
                except Exception as e:
                    print(f"Image not reconstructed (tel_id={tel_id}):", e)
                    failed = True
                    break
            # --- END LOOP on tel_ids ---

            # --- Check if event is fine ---
            # Ignore events with less than two telescopes
            if len(hillas_p.keys()) < 2:
                print(f"EVENT with LESS than 2 hillas_p (sel_tels={sel_tels})")
                continue

            # Check hillas parameters for stereo reconstruction
            if not check_stereo(event=event, tel_id=tel_id, hillas_p=hillas_p):
                print("STEREO CHECK NOT PASSED")
                continue

            tel_ids_written = list(event_info.keys())

            for tel_id in tel_ids_written:

                event.dl1.tel[tel_id].parameters = ImageParametersContainer(hillas=hillas_p[tel_id])

                hillas_reconstructor(event)

                stereo_params = event.dl2.stereo.geometry['HillasReconstructor']

                if stereo_params.az < 0:
                    stereo_params.az += u.Quantity(360, u.deg)

                impact_p = calculate_impact(
                    core_x=stereo_params.core_x,
                    core_y=stereo_params.core_y,
                    az=stereo_params.az,
                    alt=stereo_params.alt,
                    tel_pos_x=tel_positions[tel_id][0],
                    tel_pos_y=tel_positions[tel_id][1],
                    tel_pos_z=tel_positions[tel_id][2],
                )

            # --- Store DL1 data ---
            # Store hillas params

            # Loop on triggered telescopes
            for tel_id in tel_ids_written:
                # Write them
                write_hillas(
                    writer=writer,
                    event_info=event_info[tel_id],
                    hillas_p=hillas_p[tel_id],
                    leakage_p=leakage_p[tel_id],
                    timing_p=timing_p[tel_id],
                    impact_p=impact_p[tel_id],
                )

            write_stereo(
                stereo_params=stereo_params,
                stereo_id=cfg["all_tels"]["stereo_id"],
                event_info=event_info[tel_ids_written[0]],
                writer=writer,
            )

            # --- Display plot ---
            if display and go:
                go, cont = _display_plots(
                    source=source,
                    event=event,
                    hillas_p=hillas_p,
                    time_grad=time_grad,
                    stereo_params=stereo_params,
                    first=first_time_display,
                    cont=cont,
                    fig=fig,
                    ax=ax,
                )
                first_time_display = False

        # --- END LOOP event in source ---

        # --- Close DL1 writer ---
        writer.close()

    # --- END LOOP file in file_list ---

    return


def _display_plots(
    source, event, hillas_p, time_grad, stereo_params, first, cont, fig, ax
):

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Distance (m)")
    # Display the top-town view of the MAGIC-LST telescope array
    disp = ArrayDisplay(
        subarray=source.subarray, axes=ax, tel_scale=1, title="MAGIC-LST Monte Carlo"
    )
    # # Set the vector angle and length from Hillas parameters
    # disp.set_vector_hillas(
    #     hillas_dict=hillas_p,
    #     time_gradient=time_grad,
    #     angle_offset=event.pointing.array_azimuth,
    #     length=500,
    # )
    # Estimated and true impact
    plt.scatter(
        event.mc.core_x, event.mc.core_y, s=20, c="k", marker="x", label="True Impact"
    )
    plt.scatter(
        stereo_params.core_x,
        stereo_params.core_y,
        s=20,
        c="r",
        marker="x",
        label="Estimated Impact",
    )
    plt.legend()
    if first:
        plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()
    go = True
    if not cont:
        c = input("Press Enter to continue, s to stop, c to go continously: ")
    plt.cla()
    if not cont:
        if c == "s":
            go = False
        elif c == "c":
            print("Press Enter to stop the loop")
            cont = True
    if cont:
        i, o, e = select.select([sys.stdin], [], [], 0.0001)
        if i == [sys.stdin]:
            go = False
            input()
        time.sleep(0.1)
    return go, cont


if __name__ == "__main__":
    args = PARSER.parse_args()
    kwargs = args.__dict__
    start_time = time.time()
    call_stereo_reco_MAGIC_LST(kwargs)
    print_elapsed_time(start_time, time.time())
