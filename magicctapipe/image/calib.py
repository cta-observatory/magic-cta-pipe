import numpy as np
from ctapipe.image import (
    apply_time_delta_cleaning,   
    number_of_islands,
    tailcuts_clean,   
)
from ctapipe.instrument import CameraGeometry
from lstchain.image.cleaning import apply_dynamic_cleaning
from lstchain.image.modifier import (
    add_noise_in_pixels,
    random_psf_smearer,  
    set_numba_seed  
)
from magicctapipe.image import MAGICClean


__all__ = [
    "calibrate"
]


def calibrate(event, tel_id, config, calibrator, LST_bool, obs_id=None, camera_geoms=None, magic_clean=None):

    """
    This function calibrates the camera image for a single event of a telescope 

    Parameters
    ----------
    event: event 
        From an EventSource
    tel_id: int
        Telescope ID     
    config: dictionary
        Parameters for image extraction and calibration    
    calibrator: CameraCalibrator (ctapipe.calib)
        ctapipe object needed to calibrate the camera
    LST_bool: bool
        Whether the telescope is a LST
    obs_id: int
        Observation ID. Unsed in case of LST telescope
    camera_geoms: telescope.camera.geometry
        Camera geometry. Used in case of LST telescope
    magic_clean: dictionary (1 entry per MAGIC telescope)
        Each entry is a MAGICClean object using the telescope camera geometry. Used in case of MAGIC telescope
    

    Returns
    -------
    signal_pixels: boolean mask
        Mask of the pixels selected by the cleaning
    image: numpy array
        Array of number of p.e. in the camera pixels
    peak_time: numpy array
        Array of the signal peak time in the camera pixels

    """
    if (LST_bool==False) and (magic_clean==None):
        raise ValueError("Check the provided parameters and the telescope type; calibration was not possible") 
    if (LST_bool==True) and (obs_id==None):
        raise ValueError("Check the provided parameters and the telescope type; calibration was not possible") 
    if (LST_bool==True) and (camera_geoms==None):
        raise ValueError("Check the provided parameters and the telescope type; calibration was not possible") 
    if (LST_bool==False) and (type(magic_clean[tel_id])!=MAGICClean):
        raise ValueError("Check the provided magic_clean parameter; calibration was not possible") 
    if (LST_bool==True) and (type(camera_geoms[tel_id])!=CameraGeometry):
        raise ValueError("Check the provided camera_geoms parameter; calibration was not possible")

    calibrator._calibrate_dl0(event, tel_id)
    calibrator._calibrate_dl1(event, tel_id)

    image = event.dl1.tel[tel_id].image.astype(np.float64)
    peak_time = event.dl1.tel[tel_id].peak_time.astype(np.float64)
    
    if LST_bool==False:
        use_charge_correction = config["charge_correction"]["use"]

        if use_charge_correction:
            # Scale the charges by the correction factor
            image *= config["charge_correction"]["factor"]

        # Apply the image cleaning
        signal_pixels, image, peak_time = magic_clean[tel_id].clean_image(
            event_image=image, event_pulse_time=peak_time
        )
            
    elif LST_bool==True:
        increase_nsb = config["increase_nsb"].pop("use")
        increase_psf = config["increase_psf"]["use"]
        use_time_delta_cleaning = config["time_delta_cleaning"].pop("use")
        use_dynamic_cleaning = config["dynamic_cleaning"].pop("use")
        use_only_main_island = config["use_only_main_island"]
    
        if increase_nsb:
            rng = np.random.default_rng(obs_id)
            # Add extra noise in pixels
            image = add_noise_in_pixels(rng, image, **config["increase_nsb"])

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

        config["increase_nsb"]["use"]=increase_nsb
        config["time_delta_cleaning"]["use"]=use_time_delta_cleaning
        config["dynamic_cleaning"]["use"]=use_dynamic_cleaning        
    
    return signal_pixels, image, peak_time
