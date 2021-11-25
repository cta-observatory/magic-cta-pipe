import re
import glob
import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.coordinates.builtin_frames import SkyOffsetFrame
from gammapy.stats import WStatCountsStatistic


__all__ = [
    'crab_magic',
    'get_obs_ids_from_name',
    'calc_impact',
    'calc_nsim',
    'transform_telcoords_cog',
    'transform_to_radec',
    'calc_offset_rotation',
    'calc_angular_separation',
]


def crab_magic(E):

    f0 = 3.23e-11 / u.TeV / u.cm ** 2 / u.s
    alpha = -2.47
    beta = -0.24
    e0 = 1. * u.TeV

    dFdE = f0 * np.power(E / e0, alpha + beta * np.log10(E / e0)) 

    return dFdE.to(1 / u.cm ** 2 / u.s / u.TeV)


def get_obs_ids_from_name(input_data_mask):

    paths_list = glob.glob(input_data_mask)
    paths_list.sort()

    obs_ids_list = []

    for path in paths_list:

        obs_id = re.findall('.*Run(\d+)\.(\d+)\.h5', path)[0][0]
        obs_ids_list.append(obs_id)

    obs_ids_list = np.unique(obs_ids_list)

    return obs_ids_list


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


def calc_offset_rotation(ra_on, dec_on, ra_tel, dec_tel):
    
    diff_ra = ra_tel - ra_on
    diff_dec = dec_tel - dec_on
    
    offset = np.arccos(np.cos(dec_on)*np.cos(dec_tel)*np.cos(diff_ra) + np.sin(dec_on)*np.sin(dec_tel))

    rotation = np.arctan((np.sin(dec_tel)*np.cos(dec_on) - np.sin(dec_on)*np.cos(dec_tel)*np.cos(diff_ra)) / 
                         (np.cos(dec_tel)*np.sin(diff_ra)))

    offset = offset.to(u.deg)
    rotation = rotation.to(u.deg)

    rotation[(diff_ra < 0) & (diff_dec > 0)] += 180*u.deg
    rotation[(diff_ra < 0) & (diff_dec < 0)] += 180*u.deg
    rotation[(diff_ra > 0) & (diff_dec < 0)] += 360*u.deg

    return offset, rotation


def calc_angular_separation(on_coord, event_coords, tel_coords, n_off_region):

    theta_on = on_coord.separation(event_coords)

    offset, rotation = calc_offset_rotation(
        ra_on=on_coord.ra, dec_on=on_coord.dec, ra_tel=tel_coords.ra, dec_tel=tel_coords.dec
    )

    mean_offset = np.mean(offset).to(u.deg).value
    mean_rot = np.mean(rotation).to(u.deg).value

    print(f'mean_offset = {mean_offset:.3f} [deg], mean_rot = {mean_rot:.1f} [deg]')

    skyoffset_frame = SkyOffsetFrame(origin=on_coord, rotation=-mean_rot*u.deg)

    wobble_coord = SkyCoord(mean_offset*u.deg, 0*u.deg, frame=skyoffset_frame)
    wobble_coord = wobble_coord.transform_to('icrs')

    rots_list = np.arange(mean_rot, mean_rot+359, int(360/(n_off_region+1)))
    rots_list[rots_list > 360] -= 360
    
    diff = np.round(np.abs(rots_list - mean_rot), 0)
    rots_list = rots_list[diff != 180]

    theta_off = {}
    off_coords = {}

    for i_off, rot in enumerate(rots_list):
        
        skyoffset_frame = SkyOffsetFrame(origin=wobble_coord, rotation=-rot*u.deg)
        
        off_coords[i_off+1] = SkyCoord(mean_offset*u.deg, 0*u.deg, frame=skyoffset_frame)
        off_coords[i_off+1] = off_coords[i_off+1].transform_to('icrs')

        theta_off[i_off+1] = off_coords[i_off+1].separation(event_coords)

    return theta_on, theta_off, off_coords


