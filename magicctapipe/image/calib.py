

import numpy as np


from ctapipe.image import (
    apply_time_delta_cleaning,   
    number_of_islands,
    tailcuts_clean,   
)

from lstchain.image.cleaning import apply_dynamic_cleaning
from lstchain.image.modifier import (
    add_noise_in_pixels,
    random_psf_smearer,    
)

__all__ = [
    "Calibrate_LST", "Calibrate_MAGIC"
]

def Calibrate_LST(event, tel_id, rng, config_lst, camera_geoms, calibrator_lst, increase_nsb, use_time_delta_cleaning, use_dynamic_cleaning ):

    """
    This function computes and returns signal_pixels, image, and peak_time for LST
    """
    
    calibrator_lst._calibrate_dl0(event, tel_id)
    calibrator_lst._calibrate_dl1(event, tel_id)

    image = event.dl1.tel[tel_id].image.astype(np.float64)
    peak_time = event.dl1.tel[tel_id].peak_time.astype(np.float64)
    
    increase_psf = config_lst["increase_psf"]["use"]
    use_only_main_island = config_lst["use_only_main_island"]
    
    if increase_nsb:
        # Add extra noise in pixels
        image = add_noise_in_pixels(rng, image, **config_lst["increase_nsb"])

    if increase_psf:
        # Smear the image
        image = random_psf_smearer(
            image=image,
            fraction=config_lst["increase_psf"]["fraction"],
            indices=camera_geoms[tel_id].neighbor_matrix_sparse.indices,
            indptr=camera_geoms[tel_id].neighbor_matrix_sparse.indptr,
        )

    # Apply the image cleaning
    signal_pixels = tailcuts_clean(
        camera_geoms[tel_id], image, **config_lst["tailcuts_clean"]
    )

    if use_time_delta_cleaning:
        signal_pixels = apply_time_delta_cleaning(
            geom=camera_geoms[tel_id],
            mask=signal_pixels,
            arrival_times=peak_time,
            **config_lst["time_delta_cleaning"],
        )

    if use_dynamic_cleaning:
        signal_pixels = apply_dynamic_cleaning(
            image, signal_pixels, **config_lst["dynamic_cleaning"]
        )

    if use_only_main_island:
        _, island_labels = number_of_islands(camera_geoms[tel_id], signal_pixels)
        n_pixels_on_island = np.bincount(island_labels.astype(np.int64))

        # The first index means the pixels not surviving
        # the cleaning, so should not be considered
        n_pixels_on_island[0] = 0
        max_island_label = np.argmax(n_pixels_on_island)
        signal_pixels[island_labels != max_island_label] = False

    return signal_pixels, image, peak_time
    

def Calibrate_MAGIC(event, tel_id, config_magic, magic_clean, calibrator_magic):

    """
    This function computes and returns signal_pixels, image, and peak_time for MAGIC
    """
    
    calibrator_magic._calibrate_dl0(event, tel_id)
    calibrator_magic._calibrate_dl1(event, tel_id)

    image = event.dl1.tel[tel_id].image.astype(np.float64)
    peak_time = event.dl1.tel[tel_id].peak_time.astype(np.float64)
    use_charge_correction = config_magic["charge_correction"]["use"]
    
    if use_charge_correction:
        # Scale the charges by the correction factor
        image *= config_magic["charge_correction"]["factor"]

    # Apply the image cleaning
    signal_pixels, image, peak_time = magic_clean[tel_id].clean_image(
        event_image=image, event_pulse_time=peak_time
    )
    return signal_pixels, image, peak_time
