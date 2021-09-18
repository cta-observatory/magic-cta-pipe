import numpy as np
from astropy import units as u

__all__ = [
    'calc_impact',
    'calc_nsim',
    'transform_telcoords'
    # 'transform_to_radec'
]

def calc_impact(
    core_x, core_y, az, alt, tel_pos_x, tel_pos_y, tel_pos_z
    ): 

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

    t = (tel_pos_x - core_x) * np.cos(alt) * np.cos(az) - \
        (tel_pos_y - core_y) * np.cos(alt) * np.sin(az) + \
        tel_pos_z * np.sin(alt)    

    impact = np.sqrt((core_x - tel_pos_x + t * np.cos(alt) * np.cos(az))**2 + \
                     (core_y - tel_pos_y - t * np.cos(alt) * np.sin(az))**2 + \
                     (t * np.sin(alt) - tel_pos_z)**2)
    
    return impact

def calc_nsim(
        n_events_sim, eslope_sim, emin_sim, emax_sim, cscat_sim, viewcone_sim, 
        emin, emax, distmin, distmax, angmin, angmax
    ): 

    """Calculate the number of events simulated in the interval of energy, core, and viewcone

    Parameters
    ----------
    nshow_sim : float
        the number of simulated events (CORSIKA "NSHOW" parameter multiplied by CORSIKA "NSCAT" parameter and the number of runs)
    eslope_sim : float
        spectral index of simulations (CORSIKA "ESLOPE" parameter)
    emin_sim : float, unit [TeV]
        minimum energy of simulations (CORSIKA "EMIN" parameter)
    emax_sim : float, unit [TeV]
        maximum energy of simulations (CORSIKA "EMAX" parameter)
    cscat_sim : float, unit [m]
        core range of simulations (CORSIKA "CSCAT" parameter)
    viewcone_sim : float, unit [rad]
        viewcone of simulations (CORSIKA "VIEWCONE" parameter)
    emin : np.array, unit [TeV]
        minimum energy of the energy interval 
    emax : np.array, unit [TeV]
        maximum energy of the energy interval
    distmin : float, unit [m]
        minimum distance of the core interval
    distmax : float, unit [m]
        maximum distance of the core interval
    angmin : float, unit [deg]
        minimum angle of the solid angle interval
    angmax : float, unit [deg]
        maximum angle of the solid angle interval

    Returns
    -------
    nsim : float
        the number of simulated events within the interval
    """

    norm_eng = (emax**(eslope_sim+1) - emin**(eslope_sim+1))/(emax_sim**(eslope_sim+1) - emin_sim**(eslope_sim+1))
    norm_dist = (distmax**2 - distmin**2)/cscat_sim**2
    norm_ang = (np.cos(np.deg2rad(angmin)) - np.cos(np.deg2rad(angmax)))/(1 - np.cos(np.deg2rad(viewcone_sim)))

    nsim = norm_eng * norm_dist * norm_ang * n_events_sim
    
    return nsim

def transform_telcoords(tel_positions):

    tel_id_list = tel_positions.keys()

    tel_pos_x = [tel_positions[tel_id][0].value for tel_id in tel_id_list]
    tel_pos_y = [tel_positions[tel_id][1].value for tel_id in tel_id_list]
    tel_pos_z = [tel_positions[tel_id][2].value for tel_id in tel_id_list]

    tel_positions_cog = {}

    for i_tel, tel_id in enumerate(tel_id_list):

        tel_pos_x_cog = tel_pos_x[i_tel] - np.mean(tel_pos_x)
        tel_pos_y_cog = tel_pos_y[i_tel] - np.mean(tel_pos_y)
        tel_pos_z_cog = tel_pos_z[i_tel] - np.mean(tel_pos_z)

        tel_positions_cog[tel_id] = [tel_pos_x_cog, tel_pos_y_cog, tel_pos_z_cog]*u.m

    return tel_positions_cog

# def transform_to_radec(az, alt, timestamps):

#     obs_location:
#     name: 'ORM_MAGIC'
#     lat: 28.76177  # unit: [deg]
#     lon: -17.89064  # unit: [deg]
#     height: 2199.835  # unit: [m]



#     config_loc = config['obs_location']
#     location = EarthLocation.from_geodetic(lat=config_loc['lat']*u.deg, lon=config_loc['lon']*u.deg, height=config_loc['height']*u.m)

#     df = data_stereo.query('tel_id == {}'.format(config['tel_ids']['LST-1']))
#     ts_type = config['coincidence']['timestamp_lst']

#     timestamps = Time(df[ts_type].values, format='unix', scale='utc')
#     horizon_frames = AltAz(location=location, obstime=timestamps)

#     event_coords = SkyCoord(alt=container['alt'], az=container['az'], unit='rad', frame=horizon_frames)
#     event_coords = event_coords.transform_to('fk5')

#     container['ra'] = event_coords.ra.value
#     container['dec'] = event_coords.dec.value



