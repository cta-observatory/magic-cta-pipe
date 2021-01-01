from scipy.sparse.csgraph import connected_components

from ctapipe.image import leakage, hillas_parameters
from ctapipe.image.timing import timing_parameters


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
    hillas_p = hillas_parameters(geom=geom, image=image[clean])
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
