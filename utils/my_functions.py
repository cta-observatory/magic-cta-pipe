import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord  



__all__ = [
    'calc_impact',
    'calc_nsim',
    'transform_telcoords_cog',
    'transform_to_radec'
]


def calc_impact(core_x, core_y, az, alt, tel_pos_x, tel_pos_y, tel_pos_z): 

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

    norm_eng = (emax**(eslope_sim+1) - emin**(eslope_sim+1))/(emax_sim**(eslope_sim+1) - emin_sim**(eslope_sim+1))
    norm_dist = (distmax**2 - distmin**2)/cscat_sim**2
    norm_ang = (np.cos(np.deg2rad(angmin)) - np.cos(np.deg2rad(angmax)))/(1 - np.cos(np.deg2rad(viewcone_sim)))

    nsim = norm_eng * norm_dist * norm_ang * n_events_sim
    
    return nsim


def transform_telcoords_cog(tel_positions, allowed_tels):

    tel_pos_x = [tel_positions[tel_id][0].value for tel_id in allowed_tels]
    tel_pos_y = [tel_positions[tel_id][1].value for tel_id in allowed_tels]
    tel_pos_z = [tel_positions[tel_id][2].value for tel_id in allowed_tels]

    tel_positions_cog = {}

    for i_tel, tel_id in enumerate(allowed_tels):

        tel_pos_x_cog = tel_pos_x[i_tel] - np.mean(tel_pos_x)
        tel_pos_y_cog = tel_pos_y[i_tel] - np.mean(tel_pos_y)
        tel_pos_z_cog = tel_pos_z[i_tel] - np.mean(tel_pos_z)

        tel_positions_cog[tel_id] = [tel_pos_x_cog, tel_pos_y_cog, tel_pos_z_cog]*u.m

    return tel_positions_cog


def transform_to_radec(alt, az, timestamp):

    lat_orm = u.Quantity(28.76177, u.deg)      
    lon_orm = u.Quantity(-17.89064, u.deg)     
    height_orm = u.Quantity(2199.835, u.m)
  
    location = EarthLocation.from_geodetic(lat=lat_orm, lon=lon_orm, height=height_orm)

    horizon_frames = AltAz(location=location, obstime=timestamp)

    event_coords = SkyCoord(alt=alt, az=az, frame=horizon_frames)

    print('Transforming Alt/Az to RA/Dec...')
    event_coords = event_coords.transform_to('fk5')

    return event_coords.ra.value, event_coords.dec.value