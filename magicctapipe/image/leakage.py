import numpy as np
from ctapipe.containers import LeakageContainer

__all__ = [
    "get_leakage",
]

border_cache = dict()


def get_border_masks_mars(geom):
    """
    Get a mask for pixels at the border of the camera
    for width 1 and 2 using MARS definition.

    Parameters
    ----------
    geom : ctapipe.instrument.CameraGeometry
        Camera geometry information

    Returns
    -------
    tuple
        Tuple with the two masks
    """

    if geom.camera_name in border_cache:
        if 1 in border_cache[geom.camera_name] and 2 in border_cache[geom.camera_name]:
            return border_cache[geom.camera_name][1], border_cache[geom.camera_name][2]

    neighbors = geom.neighbor_matrix_sparse

    # find pixels in the outermost ring
    outermostring = [pix for pix in range(geom.n_pixels) if neighbors[pix].getnnz() < 5]

    # find pixels in the second outermost ring
    outerring = []
    for pix in range(geom.n_pixels):
        if pix in outermostring:
            continue
        for neigh in np.where(neighbors[pix][0, :].toarray() == True)[1]:
            if neigh in outermostring:
                outerring.append(pix)

    # needed because outerring has some pixels appearing more than once
    outerring = np.unique(outerring).tolist()
    outermostring_mask = np.zeros(geom.n_pixels, dtype=bool)
    outermostring_mask[outermostring] = True
    outerring_mask = np.zeros(geom.n_pixels, dtype=bool)
    outerring_mask[outerring] = True

    border_cache[geom.camera_name] = {
        1: outermostring_mask,
        2: outerring_mask,
    }

    return outermostring_mask, outerring_mask


def get_leakage(geom, event_image, clean_mask):
    """Calculate the leakage as done in MARS.

    Parameters
    ----------
    geom : ctapipe.instrument.CameraGeometry
        Camera geometry information
    event_image : np.ndarray
        Event image
    clean_mask : np.ndarray
        Cleaning mask

    Returns
    -------
    ctapipe.containers.LeakageContainer
    """

    outermostring_mask, outerring_mask = get_border_masks_mars(geom)

    # intersection between 1st outermost ring and cleaning mask
    mask1 = np.array(outermostring_mask) & clean_mask
    # intersection between 2nd outermost ring and cleaning mask
    mask2 = np.array(outerring_mask) & clean_mask

    leakage_pixel1 = np.count_nonzero(mask1)
    leakage_pixel2 = np.count_nonzero(mask2)

    leakage_intensity1 = np.sum(event_image[mask1])
    leakage_intensity2 = np.sum(event_image[mask2])

    size = np.sum(event_image[clean_mask])

    return LeakageContainer(
        pixels_width_1=leakage_pixel1 / geom.n_pixels,
        pixels_width_2=leakage_pixel2 / geom.n_pixels,
        intensity_width_1=leakage_intensity1 / size,
        intensity_width_2=leakage_intensity2 / size,
    )
