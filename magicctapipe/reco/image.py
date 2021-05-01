import numpy as np

from scipy.sparse.csgraph import connected_components

import astropy.units as u


from ctapipe.image import (
    leakage,
    hillas_parameters,
    tailcuts_clean,
    apply_time_delta_cleaning,
)
from ctapipe.image.morphology import number_of_islands
from ctapipe.image.timing import timing_parameters
from ctapipe.coordinates import GroundFrame
from ctapipe.core.container import Container, Field

# from lstchain.calib.camera.pixel_threshold_estimation import get_threshold_from_dl1_file

from astropy.coordinates import SkyCoord, AltAz


def get_num_islands_MAGIC(camera, clean_mask, event_image):
    """Eval num islands for MAGIC

    Parameters
    ----------
    camera : ctapipe.instrument.camera.geometry.CameraGeometry
        camera geometry
    clean_mask : numpy.ndarray
        clean mask
    event_image : numpy.ndarray
        event image

    Returns
    -------
    int
        num_islands
    """
    # Identifying connected islands
    neighbors = camera.neighbor_matrix_sparse
    clean_neighbors = neighbors[clean_mask][:, clean_mask]
    num_islands, labels = connected_components(clean_neighbors, directed=False)
    return num_islands


def clean_image_params(geom, image, clean, peakpos):
    """Evaluate cleaned image parameters

    Parameters
    ----------
    geom : ctapipe.instrument.camera.geometry.CameraGeometry
        camera geometry
    image : numpy.ndarray
        image
    clean : numpy.ndarray
        clean mask
    peakpos : numpy.ndarray
        peakpos

    Returns
    -------
    tuple
        ctapipe.containers.HillasParametersContainer
            hillas_p
        ctapipe.containers.LeakageContainer
            leakage_p
        ctapipe.containers.TimingParametersContainer
            timing_p
    """
    # Hillas parameters, same for LST and MAGIC. From ctapipe
    hillas_p = hillas_parameters(geom=geom[clean], image=image[clean])
    # Leakage, same for LST and MAGIC. From ctapipe
    leakage_p = leakage(geom=geom, image=image, cleaning_mask=clean)
    # Timing parameters, same for LST and MAGIC. From ctapipe
    timing_p = timing_parameters(
        geom=geom[clean],
        image=image[clean],
        peak_time=peakpos[clean],
        hillas_parameters=hillas_p,
    )

    # Make sure each telescope get's an arrow
    # if abs(time_grad[tel_id]) < 0.2:
    #     time_grad[tel_id] = 1

    return hillas_p, leakage_p, timing_p


def eval_impact(subarray, hillas_p, stereo_params):
    # Impact parameter for energy estimation (/ tel)
    ground_frame = GroundFrame()
    impact_p = {}

    class ImpactContainer(Container):
        impact = Field(-1, "Impact")

    for tel_id in hillas_p.keys():
        pos = subarray.positions[tel_id]
        tel_ground = SkyCoord(pos[0], pos[1], pos[2], frame=ground_frame)

        core_ground = SkyCoord(
            stereo_params.core_x, stereo_params.core_y, 0 * u.m, frame=ground_frame,
        )
        # Should be better handled (tilted frame)
        impact_ = np.sqrt(
            (core_ground.x - tel_ground.x) ** 2 + (core_ground.y - tel_ground.y) ** 2
        )
        impact_p[tel_id] = ImpactContainer(impact=impact_)
    return impact_p


def tailcuts_clean_lstchain(geom, image, peak_time, input_file, cleaning_parameters):
    """Apply tailcuts cleaning lstchain mode

    Parameters
    ----------
    geom: `ctapipe.instrument.CameraGeometry`
        Camera geometry information
    image: array
        pixel values
    peak_time : array
        dl1.peak_time
    cleaning_parameters : dict
        dictionary composed by the cleaning parameters for tailcuts and the ones
        for lstchain tailcuts ('delta_time' and 'use_only_main_island')

    Returns
    -------
    tuple
        signal_pixels, num_islands, island_labels
    """

    # pop delta_time and use_main_island, so we can cleaning_parameters to tailcuts
    delta_time = cleaning_parameters.pop("delta_time", None)
    use_main_island = cleaning_parameters.pop("use_only_main_island", True)

    sigma = cleaning_parameters["sigma"]
    pedestal_thresh = get_threshold_from_dl1_file(input_file, sigma)
    picture_thresh_cfg = cleaning_parameters["cleaning_parameters"]
    print(
        f"Fraction of pixel cleaning thresholds above picture thr.:"
        f"{np.sum(pedestal_thresh>picture_thresh_cfg) / len(pedestal_thresh):.3f}"
    )
    picture_thresh = np.clip(pedestal_thresh, picture_thresh_cfg, None)

    signal_pixels = tailcuts_clean(
        geom=geom,
        image=image,
        picture_thresh=picture_thresh,
        boundary_thresh=cleaning_parameters["boundary_thresh"],
        keep_isolated_pixels=cleaning_parameters["keep_isolated_pixels"],
        min_number_picture_neighbors=cleaning_parameters["min_n_neighbors"],
    )

    n_pixels = np.count_nonzero(signal_pixels)
    if n_pixels > 0:
        num_islands, island_labels = number_of_islands(camera_geom, signal_pixels)
        n_pixels_on_island = np.bincount(island_labels.astype(np.int64))
        # first island is no-island and should not be considered
        n_pixels_on_island[0] = 0
        max_island_label = np.argmax(n_pixels_on_island)
        if use_only_main_island:
            signal_pixels[island_labels != max_island_label] = False

        # if delta_time has been set, we require at least one
        # neighbor within delta_time to accept a pixel in the image:
        if delta_time is not None:
            cleaned_pixel_times = peak_time
            # makes sure only signal pixels are used in the time
            # check:
            cleaned_pixel_times[~signal_pixels] = np.nan
            new_mask = apply_time_delta_cleaning(
                camera_geom, signal_pixels, cleaned_pixel_times, 1, delta_time
            )
            signal_pixels = new_mask

        # count the surviving pixels
        n_pixels = np.count_nonzero(signal_pixels)

    return signal_pixels, num_islands, island_labels
