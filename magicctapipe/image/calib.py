"""
Module for calibration
"""
import numpy as np
from ctapipe.image import apply_time_delta_cleaning, number_of_islands, tailcuts_clean
from ctapipe.instrument import CameraGeometry
from lstchain.image.cleaning import apply_dynamic_cleaning
from lstchain.image.modifier import (
    add_noise_in_pixels,
    random_psf_smearer,
    set_numba_seed,
)

from .cleaning import MAGICClean

__all__ = ["calibrate"]


def calibrate(
    event,
    tel_id,
    config,
    calibrator,
    is_lst,
    obs_id=None,
    camera_geoms=None,
    magic_clean=None,
):
    """
    This function calibrates the camera image for a single event of a telescope

    Parameters
    ----------
    event : ctapipe.containers.ArrayEventContainer
        From an EventSource
    tel_id : int
        Telescope ID
    config : dict
        Parameters for image extraction and calibration
    calibrator : ctapipe.calib.CameraCalibrator
        `ctapipe` object needed to calibrate the camera
    is_lst : bool
        Whether the telescope is a LST
    obs_id : int, optional
        Observation ID. Unused in case of LST telescope, by default None
    camera_geoms : ctapipe.instrument.camera.geometry.CameraGeometry, optional
        Camera geometry. Used in case of LST telescope, by default None
    magic_clean : dict, optional
        Each entry is a MAGICClean object using the telescope camera geometry.
        Used in case of MAGIC telescope, by default None

    Returns
    -------
    tuple
        Mask of the pixels selected by the cleaning,
        array of number of p.e. in the camera pixels,
        array of the signal peak time in the camera pixels
    """
    if (not is_lst) and (magic_clean is None):
        raise ValueError(
            "Check the provided parameters and the telescope type; MAGIC calibration not possible if magic_clean not provided"
        )
    if (is_lst) and (obs_id is None):
        raise ValueError(
            "Check the provided parameters and the telescope type; LST calibration not possible if obs_id not provided"
        )
    if (is_lst) and (camera_geoms is None):
        raise ValueError(
            "Check the provided parameters and the telescope type; LST calibration not possible if gamera_geoms not provided"
        )
    if (not is_lst) and (type(magic_clean[tel_id]) != MAGICClean):
        raise ValueError(
            "Check the provided magic_clean parameter; MAGIC calibration not possible if magic_clean not a dictionary of MAGICClean objects"
        )
    if (is_lst) and (type(camera_geoms[tel_id]) != CameraGeometry):
        raise ValueError(
            "Check the provided camera_geoms parameter; LST calibration not possible if camera_geoms not a dictionary of CameraGeometry objects"
        )

    calibrator._calibrate_dl0(event, tel_id)
    calibrator._calibrate_dl1(event, tel_id)

    image = event.dl1.tel[tel_id].image.astype(np.float64)
    peak_time = event.dl1.tel[tel_id].peak_time.astype(np.float64)

    if not is_lst:
        use_charge_correction = config["charge_correction"]["use"]

        if use_charge_correction:
            # Scale the charges by the correction factor
            image *= config["charge_correction"]["factor"]

        # Apply the image cleaning
        signal_pixels, image, peak_time = magic_clean[tel_id].clean_image(
            event_image=image, event_pulse_time=peak_time
        )

    nsb_dict = "increase_nsb"
    if not is_lst:
        if config["mc_tel_ids"]["MAGIC-I"] == tel_id:
            nsb_dict += "_m1"
        if config["mc_tel_ids"]["MAGIC-II"] == tel_id:
            nsb_dict += "_m2"

    if nsb_dict in config:
        increase_nsb = config[nsb_dict].pop("use")
        if increase_nsb:
            rng = np.random.default_rng(obs_id + event.index.event_id)
            # Add extra noise in pixels
            image = add_noise_in_pixels(rng, image, **config[nsb_dict])
            config[nsb_dict]["use"] = increase_nsb

    if is_lst:
        increase_psf = config["increase_psf"]["use"]
        use_time_delta_cleaning = config["time_delta_cleaning"].pop("use")
        use_dynamic_cleaning = config["dynamic_cleaning"].pop("use")
        use_only_main_island = config["use_only_main_island"]

        if increase_psf:
            set_numba_seed(obs_id)
            # Smear the image
            image = random_psf_smearer(
                image=image,
                fraction=config["increase_psf"]["fraction"],
                indices=camera_geoms[tel_id].neighbor_matrix_sparse.indices,
                indptr=camera_geoms[tel_id].neighbor_matrix_sparse.indptr,
            )

        # Apply the image cleaning
        signal_pixels = tailcuts_clean(
            camera_geoms[tel_id], image, **config["tailcuts_clean"]
        )

        if use_time_delta_cleaning:
            signal_pixels = apply_time_delta_cleaning(
                geom=camera_geoms[tel_id],
                mask=signal_pixels,
                arrival_times=peak_time,
                **config["time_delta_cleaning"],
            )

        if use_dynamic_cleaning:
            signal_pixels = apply_dynamic_cleaning(
                image, signal_pixels, **config["dynamic_cleaning"]
            )

        if use_only_main_island:
            _, island_labels = number_of_islands(camera_geoms[tel_id], signal_pixels)
            n_pixels_on_island = np.bincount(island_labels.astype(np.int64))

            # The first index means the pixels not surviving
            # the cleaning, so should not be considered
            n_pixels_on_island[0] = 0
            max_island_label = np.argmax(n_pixels_on_island)
            signal_pixels[island_labels != max_island_label] = False

        config["time_delta_cleaning"]["use"] = use_time_delta_cleaning
        config["dynamic_cleaning"]["use"] = use_dynamic_cleaning

    return signal_pixels, image, peak_time
