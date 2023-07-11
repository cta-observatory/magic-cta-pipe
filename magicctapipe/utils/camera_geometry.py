from ctapipe.instrument import CameraGeometry

__all__ = [
    "scale_camera_geometry",
    "reflected_camera_geometry",
]


def scale_camera_geometry(camera_geom, factor):
    """Scale given camera geometry of a given (constant) factor

    Parameters
    ----------
    camera : CameraGeometry
        Camera geometry
    factor : float
        Scale factor

    Returns
    -------
    CameraGeometry
        Scaled camera geometry
    """
    pix_x_scaled = factor * camera_geom.pix_x
    pix_y_scaled = factor * camera_geom.pix_y
    pix_area_scaled = camera_geom.guess_pixel_area(
        pix_x_scaled, pix_y_scaled, camera_geom.pix_type
    )

    return CameraGeometry(
        camera_name="MAGICCam",
        pix_id=camera_geom.pix_id,
        pix_x=pix_x_scaled,
        pix_y=pix_y_scaled,
        pix_area=pix_area_scaled,
        pix_type=camera_geom.pix_type,
        pix_rotation=camera_geom.pix_rotation,
        cam_rotation=camera_geom.cam_rotation,
    )


def reflected_camera_geometry(camera_geom):
    """Reflect camera geometry (x->-y, y->-x)

    Parameters
    ----------
    camera_geom : CameraGeometry
        Camera geometry

    Returns
    -------
    CameraGeometry
        Reflected camera geometry
    """

    return CameraGeometry(
        camera_name="MAGICCam",
        pix_id=camera_geom.pix_id,
        pix_x=-1.0 * camera_geom.pix_y,
        pix_y=-1.0 * camera_geom.pix_x,
        pix_area=camera_geom.guess_pixel_area(
            camera_geom.pix_x, camera_geom.pix_y, camera_geom.pix_type
        ),
        pix_type=camera_geom.pix_type,
        pix_rotation=camera_geom.pix_rotation,
        cam_rotation=camera_geom.cam_rotation,
    )
