import numpy as np

def calc_impact(core_x, core_y, az, alt, tel_pos_x, tel_pos_y, tel_pos_z):

    """Calculate the Impact parameter with core positions and the Alt/Az direction of the shower axis

    Parameters
    ----------
    core_x : np.array, unit [m]
        core position on the x-axis (CORSIKA coordinate, from South to North)
    core_y : np.array, unit [m]
        core position on the y-axis (CORSIKA coordinate, from West to East)
    az : np.array, unit [rad]
        azimuth direction of the shower axis
    alt : np.array, unit [rad]
        altitude direction of the shower axis
    tel_pos_x : float, unit [m]
        telescope position on the x axis (CORSIKA coordinate, from South to North)
    tel_pos_y : float, unit [m]
        telescope position on the y axis (CORSIKA coordinate, from West to East)
    tel_pos_z : float, unit [m]
        telescope position on the z axis (CORSIKA coordinate, from the reference height = 2158 [m])

    Returns
    -------
    impact : float
        minimum distance from the telescope position to the shower axis
    """

    t = (tel_pos_x - core_x) * np.cos(alt) * np.cos(az) - (tel_pos_y - core_y) * np.cos(alt) * np.sin(az) + tel_pos_z * np.sin(alt)

    impact = np.sqrt((core_x - tel_pos_x + t * np.cos(alt) * np.cos(az))**2 + \
                     (core_y - tel_pos_y - t * np.cos(alt) * np.sin(az))**2 + (t * np.sin(alt) - tel_pos_z)**2)

    return impact