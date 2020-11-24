import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy

import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz

from ctapipe.io import SimTelEventSource
from ctapipe.io import HDF5TableWriter
from ctapipe.calib import CameraCalibrator
from ctapipe.image.cleaning import tailcuts_clean
from ctapipe.image.morphology import number_of_islands

from ctapipe.reco import HillasReconstructor
from ctapipe.visualization import ArrayDisplay

from magicctapipe.utils import MAGIC_Cleaning
from magicctapipe.reco.stereo import *
from magicctapipe.reco.reco import *


PARSER = argparse.ArgumentParser(
    description="Stereo Reconstruction MAGIC + LST",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
PARSER.add_argument('-f', '--in_file', type=str, required=True,
                    default='',
                    help='Input file')
PARSER.add_argument('-t', '--telescopes', type=str, required=False,
                    default='1,2,3,4,5,6',
                    help='Telescopes to be analyzed. 1,2,3,4: LST, 5,6: MAGIC')
PARSER.add_argument('-max', '--max_events', type=int, required=False,
                    default=0,
                    help='Max events, 0 for all')
PARSER.add_argument('-d', '--display', action='store_true', required=False,
                    default=False,
                    help='Display plots')


def stereo_reco_MAGIC_LST(file, tels, max_events=0, display=False):
    """Stereo Reconstruction MAGIC + LST

    Parameters
    ----------
    file : str
        input file, simtel.gz format
    tels : list
        telescopes to be analyzed. 1,2,3,4: LST, 5,6: MAGIC
    max_events : int, optional
        max events, 0 for all, by default 0
    display : bool, optional
        display plots, by default False
    """
    id_LST = [1, 2, 3, 4]
    id_MAGIC = [5, 6]
    tels = list(set(tels).intersection(id_LST+id_MAGIC))
    if(len(tels) < 2):
        print("Select at least two telescopes in the MAGIC + LST array")
        return
    consider_LST = any([t_ in id_LST for t_ in tels])
    consider_MAGIC = any([t_ in id_MAGIC for t_ in tels])

    # Define cleaning levels for LST. From lstchain
    # 1. From reco/tests/test_volume_reducer.py
    cleaning_config_LST = {
        'picture_thresh': 8,
        'boundary_thresh': 4,
        'keep_isolated_pixels': True,
        'min_number_picture_neighbors': 0
    }
    # 2. standard
    # cleaning_config_LST = {}

    # Define cleaning level for MAGIC. From magic-cta-pipe
    # (hillas_preprocessing_MAGICCleaning_stereo.py)
    cleaning_config_MAGIC = {
        'picture_thresh': 6,
        'boundary_thresh': 3.5,
        'max_time_off': 4.5 * 1.64,
        'max_time_diff': 1.5 * 1.64,
        'usetime': True,
        'usesum': True,
        'findhotpixels': False,
    }

    # Output file
    out_file = '%s.h5' % (file.rstrip('.simtel.gz'))

    # Open simtel file
    source = SimTelEventSource(file, max_events=max_events)

    # Init calibrator, both for MAGIC and LST
    calibrator = CameraCalibrator(subarray=source.subarray)

    if(consider_MAGIC):
        # Init MAGIC cleaning
        magic_clean = MAGIC_Cleaning.magic_clean(
            camera=source.subarray.tel[id_MAGIC[0]].camera.geometry,
            configuration=cleaning_config_MAGIC
        )

    horizon_frame = AltAz()
    hillas_reco = HillasReconstructor()
    p_ = dict(filename=out_file, group_name='dl1', overwrite=True)

    with HDF5TableWriter(**p_) as writer:
        for event in source:
            if(display):
                print("Event %d" % event.count)
            elif(event.count % 10 == 0):
                print("Event %d" % event.count)

            # Process only if I have at least two tels of the selected array
            sel_tels = list(set(event.r0.tels_with_data).intersection(tels))
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
                if(not tel_id in tels):
                    continue
                try:
                    geom = source.subarray.tels[tel_id].camera.geometry
                    image = dl1.image  # == event_image
                    peakpos = dl1.peak_time  # == event_pulse_time

                    # Cleaning
                    if geom.camera_name == "LSTCam":
                        # Apply tailcuts clean. From ctapipe
                        clean = tailcuts_clean(
                            geom=geom,
                            image=image,
                            **cleaning_config_LST
                        )
                        # Ignore images with less than 5 pixels after cleaning
                        if clean.sum() < 5:
                            continue
                        # Number of islands: LST. From ctapipe
                        num_islands, island_ids = number_of_islands(
                            geom=geom,
                            mask=clean
                        )
                    elif geom.camera_name == "MAGICCam":
                        # Apply MAGIC cleaning. From magic-cta-pipe
                        clean, event_image, peakpos = magic_clean.clean_image(
                            event_image=image,
                            event_pulse_time=peakpos
                        )
                        # Ignore images with less than 5 pixels after cleaning
                        if clean.sum() < 5:
                            continue
                        # Number of islands: MAGIC. From magic-cta-pipe
                        num_islands = get_num_islands_MAGIC(
                            camera=geom,
                            clean_mask=clean,
                            event_image=event_image
                        )
                    else:
                        continue

                    # Analize cleaned image: Hillas, leakeage, timing
                    hillas_p[tel_id], leakage_p, timing_p, time_grad[tel_id] = \
                        clean_image_params(
                            geom=geom,
                            image=image,
                            clean=clean,
                            peakpos=peakpos
                    )

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
                    # Store results
                    r_ = (event_info, hillas_p[tel_id], leakage_p, timing_p)
                    writer.write("hillas_params", r_)
                except Exception as e:
                    print("Image not reconstructed:", e)
                    break

            # End loop on tel_id
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
        # end loop on event
    # close HDF5TableWriter
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
    # Set the vector angle and length from Hillas parameters
    disp.set_vector_hillas(
        hillas_dict=hillas_p,
        time_gradient=time_grad,
        angle_offset=event.pointing.array_azimuth,
        length=500,
    )
    # Estimated and true impact
    plt.scatter(event.mc.core_x, event.mc.core_y,
                s=20, c="k", marker="x", label="True Impact")
    plt.scatter(stereo_p.core_x, stereo_p.core_y,
                s=20, c="r", marker="x", label="Estimated Impact")
    plt.legend()
    plt.show()


def _check_kwargs(kwargs):
    if(not os.path.exists(kwargs['in_file'])):
        print("File %s does not exists" % kwargs['in_file'])
        return False
    if(len(kwargs['telescopes'].split(',')) < 2):
        print("Select at least two telescopes")
        return False
    return True


if __name__ == '__main__':
    args = PARSER.parse_args()
    kwargs = args.__dict__
    if not _check_kwargs(kwargs):
        exit()
    start_time = time.time()
    stereo_reco_MAGIC_LST(
        file=kwargs['in_file'],
        tels=[int(t_) for t_ in kwargs['telescopes'].split(',')],
        max_events=kwargs['max_events'],
        display=kwargs['display']
    )
    print("Execution time: %.2f s" % (time.time() - start_time))
