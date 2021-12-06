import numpy as np
from ctapipe.containers import LeakageContainer

__all__ = [
    "get_leakage",
]

def get_leakage(camera, event_image, clean_mask):
    """Calculate the leakage as done in MARS.

    Parameters
    ----------
    camera : CameraGeometry
        Description
    event_image : np.array
        Event image
    clean_mask : np.array
        Cleaning mask

    Returns
    -------
    LeakageContainer
    """

    neighbors = camera.neighbor_matrix_sparse

    # find pixels in the outermost ring
    outermostring = []
    for pix in range(camera.n_pixels):
        if neighbors[pix].getnnz() < 5:
            outermostring.append(pix)

    # find pixels in the second outermost ring
    outerring = []
    for pix in range(camera.n_pixels):
        if pix in outermostring:
            continue
        for neigh in np.where(neighbors[pix][0,:].toarray() == True)[1]:
            if neigh in outermostring:
                outerring.append(pix)

    # needed because outerring has some pixels appearing more than once
    outerring = np.unique(outerring).tolist()
    outermostring_mask = np.zeros(camera.n_pixels, dtype=bool)
    outermostring_mask[outermostring] = True
    outerring_mask = np.zeros(camera.n_pixels, dtype=bool)
    outerring_mask[outerring] = True
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
        pixels_width_1=leakage_pixel1 / camera.n_pixels,
        pixels_width_2=leakage_pixel2 / camera.n_pixels,
        intensity_width_1=leakage_intensity1 / size,
        intensity_width_2=leakage_intensity2 / size,
    )