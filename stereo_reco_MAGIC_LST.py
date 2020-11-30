import os
import time
import copy
import yaml
import glob
import scipy
import argparse
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz

from ctapipe.io import SimTelEventSource
from ctapipe.io import HDF5TableWriter
from ctapipe.calib import CameraCalibrator
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.morphology import number_of_islands

from ctapipe.reco import HillasReconstructor
from ctapipe.visualization import ArrayDisplay

from magicctapipe.utils import MAGIC_Badpixels
from magicctapipe.utils import MAGIC_Cleaning
from magicctapipe.utils.utils import *
from magicctapipe.utils.tels import *
from magicctapipe.reco.stereo import *
from magicctapipe.reco.image import *


PARSER = argparse.ArgumentParser(
    description="Stereo Reconstruction MAGIC + LST",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
PARSER.add_argument('-cfg', '--config_file', type=str, required=False,
                    default='./config/config_MAGIC_LST.yaml',
                    help='Config file')
PARSER.add_argument('-n', '--max_events', type=int, required=False,
                    default=0,
                    help='Max events, 0 for all')
PARSER.add_argument('-d', '--display', action='store_true', required=False,
                    default=False,
                    help='Display plots')


def stereo_reco_MAGIC_LST(config_file, max_events=0, display=False):
    """Stereo Reconstruction MAGIC + LST

    Parameters
    ----------
    config_file : str
        configuration file, .yaml format
    max_events : int, optional
        max events, 0 for all, by default 0
    display : bool, optional
        display plots, by default False
    """
    with open(config_file, 'r') as f_:
        cfg = yaml.safe_load(f_)

    tels_ids, tels_ids_LST, tels_ids_MAGIC = \
        intersec_tel_ids(
            all_tel_ids_LST=cfg['LST']['tel_ids'],
            all_tel_ids_MAGIC=cfg['MAGIC']['tel_ids'],
            tel_ids_sel=cfg['all_tels']['tel_ids']
        )
    if(len(tels_ids) < 2):
        print("Select at least two telescopes in the MAGIC + LST array")
        return
    consider_LST = len(tels_ids_LST) > 0
    consider_MAGIC = len(tels_ids_MAGIC) > 0

    file_list = glob.glob(cfg['data_files']['mc']['train_sample']['mask_sim'])

    # Output file
    # out_file = out_file_h5(in_file=file_list[0], li=3, hi=6)
    out_file = cfg['data_files']['mc']['train_sample']['hillas_h5']
    print("Output file:\n%s" % out_file)

    writer = HDF5TableWriter(
        filename=out_file, group_name='dl1', overwrite=True
    )

    previous_event_id = 0

    # Opening the output file
    for file in file_list:
        print("Analyzing file:\n%s" % file)
        # Open simtel file
        source = SimTelEventSource(file, max_events=max_events)
        # Init calibrator, both for MAGIC and LST
        calibrator = CameraCalibrator(subarray=source.subarray)
        # Init MAGIC cleaning
        if(consider_MAGIC):
            magic_clean = MAGIC_Cleaning.magic_clean(
                camera=source.subarray.tel[tels_ids_MAGIC[0]].camera.geometry,
                configuration=cfg['MAGIC']['cleaning_config']
            )
            badpixel_calculator = MAGIC_Badpixels.MAGICBadPixelsCalc(
                config=cfg['MAGIC']['bad_pixel_config']
            )
        horizon_frame = AltAz()
        hillas_reco = HillasReconstructor()

        for event in source:
            if previous_event_id == event.index.event_id:
                continue
            previous_event_id = copy.copy(event.index.event_id)

            if(display):
                print("Event %d" % event.count)
            elif(event.count % 10 == 0):
                print("Event %d" % event.count)

            # Process only if I have at least two tels_ids of the selected array
            sel_tels = \
                list(set(event.r0.tels_with_data).intersection(tels_ids))
            if(len(sel_tels) < 2):
                continue

            telescope_pointings, hillas_p, time_grad = {}, {}, {}

            # Eval pointing
            array_pointing = SkyCoord(
                az=event.pointing.array_azimuth,
                alt=event.pointing.array_altitude,
                frame=horizon_frame
            )

            # Calibrate event, both for MAGIC and LST
            calibrator(event)

            # Loop on triggered telescopes
            for tel_id, dl1 in event.dl1.tel.items():
                # Exclude telescopes not selected
                if(not tel_id in tels_ids):
                    continue
                try:
                    geom = source.subarray.tels[tel_id].camera.geometry
                    image = dl1.image  # == event_image
                    peakpos = dl1.peak_time  # == event_pulse_time

                    # Cleaning
                    if geom.camera_name == cfg['LST']['camera_name']:
                        # Apply tailcuts clean. From ctapipe
                        clean = tailcuts_clean(
                            geom=geom,
                            image=image,
                            **cfg['LST']['cleaning_config']
                        )
                        # Ignore if less than n pixels after cleaning
                        if clean.sum() < cfg['LST']['min_pixel']:
                            continue
                        # Number of islands: LST. From ctapipe
                        num_islands, island_ids = number_of_islands(
                            geom=geom,
                            mask=clean
                        )
                    elif geom.camera_name == cfg['MAGIC']['camera_name']:
                        # badrmspixel_mask = \
                        #     badpixel_calculator.get_badrmspixel_mask(event)
                        # deadpixel_mask = \
                        #     badpixel_calculator.get_deadpixel_mask(event)
                        # unsuitable_mask = np.logical_or(
                        #     badrmspixel_mask[tel_id-1],
                        #     deadpixel_mask[tel_id-1]
                        # )
                        # Apply MAGIC cleaning. From magic-cta-pipe
                        clean, image, peakpos = magic_clean.clean_image(
                            event_image=image,
                            event_pulse_time=peakpos
                        )
                        # Ignore if less than n pixels after cleaning
                        if clean.sum() < cfg['MAGIC']['min_pixel']:
                            continue
                        # Number of islands: MAGIC. From magic-cta-pipe
                        num_islands = get_num_islands_MAGIC(
                            camera=geom,
                            clean_mask=clean,
                            event_image=image
                        )
                    else:
                        continue
                    # Analize cleaned image: Hillas, leakeage, timing
                    hillas_p[tel_id], leakage_p, timing_p = clean_image_params(
                        geom=geom,
                        image=image,
                        clean=clean,
                        peakpos=peakpos
                    )
                    # Get time gradients
                    time_grad[tel_id] = timing_p.slope.value

                    telescope_pointings[tel_id] = SkyCoord(
                        alt=event.pointing.tel[tel_id].altitude,
                        az=event.pointing.tel[tel_id].azimuth,
                        frame=horizon_frame,
                    )

                    # Preparing metadata
                    event_info = StereoInfoContainer(
                        obs_id=event.index.obs_id,
                        event_id=scipy.int32(event.index.event_id),
                        tel_id=tel_id,
                        true_energy=event.mc.energy,
                        true_alt=event.mc.alt.to(u.rad),
                        true_az=event.mc.az.to(u.rad),
                        tel_alt=event.pointing.tel[tel_id].altitude.to(u.rad),
                        tel_az=event.pointing.tel[tel_id].azimuth.to(u.rad),
                        num_islands=num_islands
                    )
                    # Store hillas results
                    write_hillas(
                        writer=writer,
                        event_info=event_info,
                        hillas_p=hillas_p[tel_id],
                        leakage_p=leakage_p,
                        timing_p=timing_p
                    )
                except Exception as e:
                    print("Image not reconstructed:", e)
                    break
            # --- END LOOP on tel_ids ---

            # Ignore events with less than two telescopes
            if(len(hillas_p) < 2):
                continue
            # Eval stereo parameters and write them
            stereo_p = check_write_stereo(
                event=event,
                tel_id=tel_id,
                hillas_p=hillas_p,
                hillas_reco=hillas_reco,
                subarray=source.subarray,
                array_pointing=array_pointing,
                telescope_pointings=telescope_pointings,
                event_info=event_info,
                writer=writer
            )
            # Display plot
            if(display):
                _display_plots(
                    source=source,
                    event=event,
                    hillas_p=hillas_p,
                    time_grad=time_grad,
                    stereo_p=stereo_p
                )
        # --- END LOOP event in source ---
    # --- END LOOP file in file_list ---
    writer.close()
    return


def _display_plots(source, event, hillas_p, time_grad, stereo_p):
    fig, ax = plt.subplots()
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Distance (m)")
    # Display the top-town view of the MAGIC-LST telescope array
    disp = ArrayDisplay(
        subarray=source.subarray,
        axes=ax,
        tel_scale=1,
        title='MAGIC-LST Monte Carlo'
    )
    # # Set the vector angle and length from Hillas parameters
    # disp.set_vector_hillas(
    #     hillas_dict=hillas_p,
    #     time_gradient=time_grad,
    #     angle_offset=event.pointing.array_azimuth,
    #     length=500,
    # )
    # Estimated and true impact
    plt.scatter(event.mc.core_x, event.mc.core_y,
                s=20, c="k", marker="x", label="True Impact")
    plt.scatter(stereo_p.core_x, stereo_p.core_y,
                s=20, c="r", marker="x", label="Estimated Impact")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    args = PARSER.parse_args()
    kwargs = args.__dict__
    start_time = time.time()
    stereo_reco_MAGIC_LST(
        config_file=kwargs['config_file'],
        max_events=kwargs['max_events'],
        display=kwargs['display']
    )
    print("Execution time: %.2f s" % (time.time() - start_time))
