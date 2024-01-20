"""
Utilities for camera geometry transformations
"""
import astropy.units as u
from ctapipe.instrument import CameraGeometry

__all__ = [
    "reflected_camera_geometry_mars",
]


def reflected_camera_geometry_mars(camera_geom):
    """Reflect camera geometry (x->-y, y->-x)

    Parameters
    ----------
    camera_geom : ctapipe.instrument.camera.geometry.CameraGeometry
        Camera geometry

    Returns
    -------
    ctapipe.instrument.camera.geometry.CameraGeometry
        Reflected camera geometry
    """

    return CameraGeometry(
        name="MAGICCam",
        pix_id=camera_geom.pix_id,
        pix_x=-1.0 * camera_geom.pix_y,
        pix_y=-1.0 * camera_geom.pix_x,
        pix_area=camera_geom.pix_area,
        pix_type=camera_geom.pix_type,
        pix_rotation=30 * u.deg - camera_geom.pix_rotation,
        cam_rotation=camera_geom.cam_rotation,
    )
