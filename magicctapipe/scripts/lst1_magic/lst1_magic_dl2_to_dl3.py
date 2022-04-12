#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script creates a DL3 data file with input DL2 data and IRF files.
Event cuts are extracted from the input IRF file and are applied to the input DL2 events.

Usage:
$ python lst1_magic_dl2_to_dl3.py
--input-file-dl2 ./data/dl2_LST-1_MAGIC.Run03265.h5
--input-file-irf ./data/irf_40deg_90deg_off0.4deg_LST-1_MAGIC_software_gam_global0.6_theta_global0.2.fits.gz
--output-dir ./data
--config-file ./config.yaml
"""

import re
import sys
import time
import yaml
import logging
import argparse
import operator
import numpy as np
import pandas as pd
from pathlib import Path
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.table import QTable
from astropy.coordinates import SkyCoord
from pyirf.cuts import evaluate_binned_cut
from magicctapipe.utils import (
    get_dl2_mean,
    check_tel_combination,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

ORM_LAT = 28.76177   # unit: [deg]
ORM_LON = -17.89064   # unit: [deg]
ORM_HEIGHT = 2199.835   # unit: [m]

MJDREF = Time(0, format='unix', scale='utc').mjd

__all__ = [
    'load_dl2_data_file',
    'create_event_list',
    'create_gti_table',
    'create_pointing_table',
    'dl2_to_dl3',
]


def load_dl2_data_file(input_file, quality_cuts, irf_type):
    """
    Loads an input DL2 data file.

    Parameters
    ----------
    input_file: str
        Path to an input DL2 data file
    quality_cuts: str
        Quality cuts applied to the input events
    irf_type: str
        Type of the LST-1 + MAGIC IRFs

    Returns
    -------
    event_table: astropy.table.table.QTable
        Astropy table of DL2 events
    obs_id_lst:
        LST observation ID of the input data
    """

    df_events = pd.read_hdf(input_file, key='events/parameters')
    df_events.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    df_events.sort_index(inplace=True)

    obs_id_lst = np.unique(df_events['obs_id_lst'])[0]

    # Apply the quality cuts:
    if quality_cuts is not None:

        logger.info('\nApplying the quality cuts...')

        df_events.query(quality_cuts, inplace=True)
        df_events['multiplicity'] = df_events.groupby(['obs_id', 'event_id']).size()
        df_events.query('multiplicity > 1', inplace=True)

    combo_types = check_tel_combination(df_events)
    df_events.update(combo_types)

    # Select the events of the specified IRF type:
    if irf_type == 'software':
        logger.info('\nExtracting the 3-tels events...')
        df_events.query('combo_type == 3', inplace=True)

    elif irf_type == 'hardware':
        logger.info('\nThe hardware trigger has not yet been used for observations. Exiting.')
        sys.exit()

    # Compute the mean of the DL2 parameters:
    df_dl2_mean = get_dl2_mean(df_events)
    df_dl2_mean.reset_index(inplace=True)

    # Convert the pandas data frame to the astropy QTable:
    event_table = QTable.from_pandas(df_dl2_mean)

    event_table['pointing_alt'] *= u.rad
    event_table['pointing_az'] *= u.rad
    event_table['pointing_ra'] *= u.deg
    event_table['pointing_dec'] *= u.deg
    event_table['reco_alt'] *= u.deg
    event_table['reco_az'] *= u.deg
    event_table['reco_ra'] *= u.deg
    event_table['reco_dec'] *= u.deg
    event_table['reco_energy'] *= u.TeV

    return event_table, obs_id_lst


def create_event_list(event_table, obs_id_lst, effective_time, elapsed_time,
                      source_name=None, source_ra=None, source_dec=None):
    """
    Creates an event list and its header.

    Parameters
    ----------
    event_table: astropy.table.table.QTable
        Astropy table of the DL2 events surviving gammaness cuts
    obs_id_lst: int
        LST observation ID of the input data
    effective_time: float
        Effective time of the input data
    elapsed_time: float
        Elapsed time of the input data
    source_name: str
        Name of the observed source
    source_ra:
        Right ascension of the observed source
    source_dec:
        Declination of the observed source

    Returns
    -------
    event_list: astropy.table.table.QTable
        Astropy table of the DL2 events for DL3 data
    event_header: astropy.io.fits.header.Header
        Astropy header for the event list
    """

    event_coords = SkyCoord(ra=event_table['reco_ra'], dec=event_table['reco_dec'], frame='icrs')

    if source_name is not None:
        source_coord = SkyCoord.from_name(source_name)
    else:
        source_coord = SkyCoord(ra=source_ra, dec=source_dec, frame='icrs')

    time_start = Time(event_table['timestamp'][0], format='unix', scale='utc')
    time_end = Time(event_table['timestamp'][-1], format='unix', scale='utc')

    delta_time = time_end.value - time_start.value

    deadc = effective_time / elapsed_time

    # Create an event list:
    event_list = QTable({
        'EVENT_ID': event_table['event_id'],
        'TIME': event_table['timestamp'],
        'RA': event_table['reco_ra'],
        'DEC': event_table['reco_dec'],
        'ENERGY': event_table['reco_energy'],
        'GAMMANESS': event_table['gammaness'],
        'MULTIP': event_table['multiplicity'],
        'GLON': event_coords.galactic.l.to(u.deg),
        'GLAT': event_coords.galactic.b.to(u.deg),
        'ALT': event_table['reco_alt'],
        'AZ': event_table['reco_az'],
    })

    # Create an event header:
    event_header = fits.Header()
    event_header['CREATED'] = Time.now().utc.iso
    event_header['HDUCLAS1'] = 'EVENTS'
    event_header['OBS_ID'] = obs_id_lst
    event_header['DATE-OBS'] = time_start.to_value('iso', 'date')
    event_header['TIME-OBS'] = time_start.to_value('iso', 'date_hms')[11:]
    event_header['DATE-END'] = time_end.to_value('iso', 'date')
    event_header['TIME-END'] = time_end.to_value('iso', 'date_hms')[11:]
    event_header['TSTART'] = time_start.value
    event_header['TSTOP'] = time_end.value
    event_header['MJDREFI'] = np.modf(MJDREF)[1]
    event_header['MJDREFF'] = np.modf(MJDREF)[0]
    event_header['TIMEUNIT'] = 's'
    event_header['TIMESYS'] = 'UTC'
    event_header['TIMEREF'] = 'TOPOCENTER'
    event_header['ONTIME'] = elapsed_time
    event_header['TELAPSE'] = delta_time
    event_header['DEADC'] = deadc
    event_header['LIVETIME'] = effective_time
    event_header['OBJECT'] = source_name
    event_header['OBS_MODE'] = 'WOBBLE'
    event_header['N_TELS'] = 3
    event_header['TELLIST'] = 'LST-1_MAGIC'
    event_header['INSTRUME'] = 'LST-1_MAGIC'
    event_header['RA_PNT'] = event_table['pointing_ra'][0].value
    event_header['DEC_PNT'] = event_table['pointing_dec'][0].value
    event_header['ALT_PNT'] = event_table['pointing_alt'][0].to_value(u.deg)
    event_header['AZ_PNT'] = event_table['pointing_az'][0].to_value(u.deg)
    event_header['RA_OBJ'] = source_coord.ra.to_value(u.deg)
    event_header['DEC_OBJ'] = source_coord.dec.to_value(u.deg)
    event_header['FOVALIGN'] = 'RADEC'

    return event_list, event_header


def create_gti_table(event_table):
    """
    Creates a GTI table and its header.

    Parameters
    ----------
    event_table: astropy.table.table.QTable
        Astropy table of the DL2 events surviving gammaness cuts

    Returns
    -------
    gti_table: astropy.table.table.QTable
        Astropy table of the GTI information
    gti_header: astropy.io.fits.header.Header
        Astropy header for the GTI table
    """

    gti_table = QTable({
        'START': u.Quantity(event_table['timestamp'][0], unit=u.s, ndmin=1),
        'STOP': u.Quantity(event_table['timestamp'][-1], unit=u.s, ndmin=1),
    })

    gti_header = fits.Header()
    gti_header['CREATED'] = Time.now().utc.iso
    gti_header['HDUCLAS1'] = 'GTI'
    gti_header['OBS_ID'] = np.unique(event_table['obs_id'])[0]
    gti_header['MJDREFI'] = np.modf(MJDREF)[1]
    gti_header['MJDREFF'] = np.modf(MJDREF)[0]
    gti_header['TIMEUNIT'] = 's'
    gti_header['TIMESYS'] = 'UTC'
    gti_header['TIMEREF'] = 'TOPOCENTER'

    return gti_table, gti_header


def create_pointing_table(event_table):
    """
    Creates a pointing table and its header.

    Parameters
    ----------
    event_table: astropy.table.table.QTable
        Astropy table of the DL2 events surviving gammaness cuts

    Returns
    -------
    pnt_table: astropy.table.table.QTable
        Astropy table of the pointing information
    pnt_header: astropy.io.fits.header.Header
        Astropy header for the pointing table
    """

    pnt_table = QTable({
        'TIME': u.Quantity(event_table['timestamp'][0], unit=u.s, ndmin=1),
        'RA_PNT': u.Quantity(event_table['pointing_ra'][0].value, ndmin=1),
        'DEC_PNT': u.Quantity(event_table['pointing_dec'][0].value, ndmin=1),
        'ALT_PNT': u.Quantity(event_table['pointing_alt'][0].to_value(u.deg), ndmin=1),
        'AZ_PNT': u.Quantity(event_table['pointing_az'][0].to_value(u.deg), ndmin=1),
    })

    pnt_header = fits.Header()
    pnt_header['CREATED'] = Time.now().utc.iso
    pnt_header['HDUCLAS1'] = 'POINTING'
    pnt_header['OBS_ID'] = np.unique(event_table['obs_id'])[0]
    pnt_header['MJDREFI'] = np.modf(MJDREF)[1]
    pnt_header['MJDREFF'] = np.modf(MJDREF)[0]
    pnt_header['TIMEUNIT'] = 's'
    pnt_header['TIMESYS'] = 'UTC'
    pnt_header['TIMEREF'] = 'TOPOCENTER'
    pnt_header['OBSGEO-L'] = (ORM_LON, 'Geographic longitude of the telescopes (deg)')
    pnt_header['OBSGEO-B'] = (ORM_LAT, 'Geographic latitude of the telescopes (deg)')
    pnt_header['OBSGEO-H'] = (ORM_HEIGHT, 'Geographic height of the telescopes (m)')

    return pnt_table, pnt_header


def dl2_to_dl3(input_file_dl2, input_file_irf, output_dir, config):
    """
    Creates a DL3 data file with input DL2 data and IRF files.

    Parameters
    ----------
    input_file_dl2: str
        Path to an input DL2 data file
    input_file_irf: str
        Path to an input IRF file
    output_dir: str
        Path to a directory where to save an output DL3 data file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    config_dl3 = config['dl2_to_dl3']

    logger.info('\nConfiguration for the DL3 process:')
    logger.info(config_dl3)

    hdus = fits.HDUList([fits.PrimaryHDU(), ])

    # Load the input IRF file:
    logger.info('\nLoading the input IRF file:')
    logger.info(input_file_irf)

    hdus_irf = fits.open(input_file_irf)
    header = hdus_irf[1].header

    quality_cuts = header['QUAL_CUT']
    irf_type = header['IRF_TYPE']

    logger.info(f'\nQuality cuts: {quality_cuts}')
    logger.info(f'IRF type: {irf_type}')

    hdus += hdus_irf[1:]

    # Load the input DL2 data file:
    logger.info('\nLoading the input DL2 data file:')
    logger.info(input_file_dl2)

    event_table, obs_id_lst = load_dl2_data_file(input_file_dl2, quality_cuts, irf_type)

    # ToBeUpdated: how to compute the effective time for the software coincidence?
    # At the moment it does not consider any dead times, which slightly underestimates a source flux.
    elapsed_time = event_table['timestamp'][-1] - event_table['timestamp'][0]
    effective_time = elapsed_time

    # Apply gammaness cuts:
    if 'GH_CUT' in header:
        logger.info('\nApplying a global gammaness cut...')

        global_gam_cut = header['GH_CUT']
        event_table = event_table[event_table['gammaness'] > global_gam_cut]

    else:
        logger.info('\nApplying dynamic gammaness cuts...')

        cut_table_gh = QTable.read(input_file_irf, 'GH_CUTS')

        mask_gh_gamma = evaluate_binned_cut(
            values=event_table['gammaness'],
            bin_values=event_table['reco_energy'],
            cut_table=cut_table_gh,
            op=operator.ge,
        )

        event_table = event_table[mask_gh_gamma]

    # Create a event list HDU:
    logger.info('\nCreating an event list HDU...')

    event_list, event_header = create_event_list(event_table, obs_id_lst,
                                                 effective_time, elapsed_time, **config_dl3)

    hdu_event = fits.BinTableHDU(event_list, header=event_header, name='EVENTS')
    hdus.append(hdu_event)

    # Create a GTI table:
    logger.info('Creating a GTI HDU...')

    gti_table, gti_header = create_gti_table(event_table)

    hdu_gti = fits.BinTableHDU(gti_table, header=gti_header, name='GTI')
    hdus.append(hdu_gti)

    # Create a pointing table:
    logger.info('Creating a pointing HDU...')

    pnt_table, pnt_header = create_pointing_table(event_table)

    hdu_pnt = fits.BinTableHDU(pnt_table, header=pnt_header, name='POINTING')
    hdus.append(hdu_pnt)

    # Save the data in an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    regex = r'dl2_(\S+)\.h5'
    file_name = Path(input_file_dl2).resolve().name

    if re.fullmatch(regex, file_name):
        parser = re.findall(regex, file_name)[0]
        output_file = f'{output_dir}/dl3_{parser}.fits.gz'
    else:
        raise RuntimeError('Could not parse information from the input file name.')

    hdus.writeto(output_file, overwrite=True)

    logger.info('\nOutput file')
    logger.info(output_file)


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file-dl2', '-d', dest='input_file_dl2', type=str, required=True,
        help='Path to an input DL2 data file.',
    )

    parser.add_argument(
        '--input-file-irf', '-i', dest='input_file_irf', type=str, required=True,
        help='Path to an input IRF file.',
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='./data',
        help='Path to a directory where to save an output DL3 data file.',
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
       help='Path to a yaml configuration file.',
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    # Process the input data:
    dl2_to_dl3(args.input_file_dl2, args.input_file_irf, args.output_dir, config)

    logger.info('\nDone.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()