import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord

__all__ = [
    'calc_impact',
    'calc_nsim',
    'transform_telcoords_cog',
    'transform_to_radec',
    'calc_offset_rotation'
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
        emin=None, emax=None, distmin=None, distmax=None, angmin=None, angmax=None
    ): 

    if (emin != None) & (emax != None):
        norm_eng = (emax**(eslope_sim+1) - emin**(eslope_sim+1))/(emax_sim**(eslope_sim+1) - emin_sim**(eslope_sim+1))
    else:
        norm_eng = 1
    
    if (distmin != None) & (distmax != None):
        norm_dist = (distmax**2 - distmin**2)/cscat_sim**2
    else:
        norm_dist = 1

    if (angmin != None) & (angmax != None):
        norm_ang = (np.cos(angmin) - np.cos(angmax))/(1 - np.cos(viewcone_sim))
    else:
        norm_ang = 1

    nsim = n_events_sim * norm_eng * norm_dist * norm_ang 
    
    return nsim.value


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
    event_coords = event_coords.transform_to('icrs')

    return event_coords.ra, event_coords.dec


def calc_offset_rotation(ra1, dec1, ra2, dec2):
    
    diff_ra = ra2 - ra1
    diff_dec = dec2 - dec1
    
    offset = np.arccos(np.cos(dec1)*np.cos(dec2)*np.cos(diff_ra) + np.sin(dec1)*np.sin(dec2))

    rotation = np.arctan((np.sin(dec2)*np.cos(dec1) - np.sin(dec1)*np.cos(dec2)*np.cos(diff_ra)) / 
                         (np.cos(dec2)*np.sin(diff_ra)))

    offset = offset.to(u.deg)
    rotation = rotation.to(u.deg)

    rotation[(diff_ra < 0) & (diff_dec > 0)] += 180*u.deg
    rotation[(diff_ra < 0) & (diff_dec < 0)] += 180*u.deg
    rotation[(diff_ra > 0) & (diff_dec < 0)] += 360*u.deg

    return offset, rotation