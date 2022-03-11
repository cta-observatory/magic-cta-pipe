#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script processes DL2 data to DL3.

Usage:
$ python lst1_magic_dl2_to_dl3.py
--input-file-dl2 ./data/dl2_LST-1_MAGIC.Run03265.h5
--input-file-irf ./data/irf_LST-1_MAGIC.fits.gz
--output-dir ./data
--config-file ./config.yaml
"""

import re
import time
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.table import QTable
from astropy.coordinates import SkyCoord
from magicctapipe import __version__
from magicctapipe.utils import (
    get_dl2_mean,
    check_tel_combination,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


__all__ = [
    'dl2_to_dl3',
]

def dl2_to_dl3(input_file_dl2, input_file_irf, output_dir, config):
    """
    Process DL2 to DL3.

    Parameters
    ----------
    input_file_dl2: str
        Path to an input DL2 data file.
    input_file_irf: str
        Path to an input IRF file
    output_dir: str
        Path to a directory where to save an output DL3 data file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    config_dl3 = config['create_irf_dl3']

    data = pd.read_hdf(input_file_dl2, 'events/parameters')
    data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    data.sort_index(inplace=True)

    check_tel_combination(data)

    if config_dl3['quality_cuts'] is not None:

        logger.info('\nApplying the following quality cuts:')
        logger.info(config_dl3['quality_cuts'])

        data.query(config_dl3['quality_cuts'],  inplace=True)
        data['multiplicity'] = data.groupby(['obs_id', 'event_id']).size()
        data.query('multiplicity > 1', inplace=True)

        combo_types = check_tel_combination(data)
        data.update(combo_types)

    # Compute the mean of the DL2 parameters:
    dl2_mean = get_dl2_mean(data)

    # ToBeUpdated: at the moment extract only the 3-tels events:
    dl2_mean.query('combo_type == 3', inplace=True)

    # Convert the pandas data frame to astropy QTable:
    dl2_mean.reset_index(inplace=True)
    data_qtable = QTable.from_pandas(dl2_mean)

    data_qtable['pointing_alt'] *= u.rad
    data_qtable['pointing_az'] *= u.rad
    data_qtable['pointing_ra'] *= u.deg
    data_qtable['pointing_dec'] *= u.deg
    data_qtable['reco_alt'] *= u.deg
    data_qtable['reco_az'] *= u.deg
    data_qtable['reco_ra'] *= u.deg
    data_qtable['reco_dec'] *= u.deg
    data_qtable['reco_energy'] *= u.TeV

    # Compute the effective time and elapsed time.
    # ToBeUpdated: how to compute the effective time for the software coincidence?
    elapsed_time = data_qtable['timestamp'][-1] - data_qtable['timestamp'][0]
    effective_time = elapsed_time

    # Compute angular distances:
    source_coord = SkyCoord.from_name(config_dl3['source_name'])
    source_coord = source_coord.transform_to('icrs')

    event_coords = SkyCoord(
        ra=data_qtable['reco_ra'], dec=data_qtable['reco_dec'], frame='icrs',
    )

    theta = source_coord.separation(event_coords)
    data_qtable['theta'] = theta.to(u.deg)

    # Apply the gammaness cut:
    gam_cut = config_dl3['gammaness_cut']
    mask_gam = (data_qtable['gammaness'] > gam_cut)

    data_qtable = data_qtable[mask_gam]
    event_coords = event_coords[mask_gam]

    hdus = fits.HDUList([fits.PrimaryHDU(), ])

    # Create a event HDU:
    logger.info('\nCreating an event HDU...')

    event_table = QTable({
        'EVENT_ID': data_qtable['event_id'],
        'TIME': data_qtable['timestamp'],
        'RA': data_qtable['reco_ra'],
        'DEC': data_qtable['reco_dec'],
        'ENERGY': data_qtable['reco_energy'],
        'GAMMANESS': data_qtable['gammaness'],
        'MULTIP': u.Quantity(np.repeat(3, len(data_qtable)), dtype=int),
        'GLON': event_coords.galactic.l.to(u.deg),
        'GLAT': event_coords.galactic.b.to(u.deg),
        'ALT': data_qtable['reco_alt'],
        'AZ': data_qtable['reco_az'],
    })

    ev_header = fits.Header()
    ev_header["CREATED"] = Time.now().utc.iso
    ev_header["HDUCLAS1"] = 'EVENTS'
    ev_header["OBS_ID"] = np.unique(data_qtable['obs_id'])[0]
    ev_header["DATE-OBS"] = Time(data_qtable['timestamp'][0], format='unix', scale='utc').to_value('iso', 'date_hms')[:10]
    ev_header["TIME-OBS"] = Time(data_qtable['timestamp'][0], format='unix', scale='utc').to_value('iso', 'date_hms')[11:]
    ev_header["DATE-END"] = Time(data_qtable['timestamp'][-1], format='unix', scale='utc').to_value('iso', 'date_hms')[:10]
    ev_header["TIME-END"] = Time(data_qtable['timestamp'][-1], format='unix', scale='utc').to_value('iso', 'date_hms')[11:]
    ev_header["TSTART"] = data_qtable['timestamp'][0]
    ev_header["TSTOP"] = data_qtable['timestamp'][-1]
    ev_header["MJDREFI"] = int(Time("1970-01-01T00:00", scale="utc").mjd)
    ev_header["MJDREFF"] = Time("1970-01-01T00:00", scale="utc").mjd - ev_header["MJDREFI"]
    ev_header["TIMEUNIT"] = "s"
    ev_header["TIMESYS"] = "UTC"
    ev_header["TIMEREF"] = "TOPOCENTER"
    ev_header["ONTIME"] = elapsed_time
    ev_header["TELAPSE"] = elapsed_time
    ev_header["DEADC"] = effective_time / elapsed_time
    ev_header["LIVETIME"] = effective_time
    ev_header["OBJECT"] = config_dl3['source_name']
    ev_header["OBS_MODE"] = 'WOBBLE'
    ev_header["N_TELS"] = 3
    ev_header["TELLIST"] = 'LST-1_MAGIC'
    ev_header["INSTRUME"] = 'LST-1_MAGIC'
    ev_header["RA_PNT"] = data_qtable['pointing_ra'][0].to_value(u.deg)
    ev_header["DEC_PNT"] = data_qtable['pointing_dec'][0].to_value(u.deg)
    ev_header["ALT_PNT"] = data_qtable['pointing_alt'][0].to_value(u.deg)
    ev_header["AZ_PNT"] = data_qtable['pointing_az'][0].to_value(u.deg)
    ev_header["RA_OBJ"] = source_coord.ra.to_value(u.deg)
    ev_header["DEC_OBJ"] = source_coord.dec.to_value(u.deg)
    ev_header["FOVALIGN"] = "RADEC"

    hdu_event = fits.BinTableHDU(event_table, header=ev_header, name="EVENTS")
    hdus.append(hdu_event)

    # Create a GTI table:
    logger.info('Creating a GTI HDU...')

    gti_table = QTable({
        'START': u.Quantity(data_qtable['timestamp'][0], unit=u.s, ndmin=1),
        'STOP': u.Quantity(data_qtable['timestamp'][-1], unit=u.s, ndmin=1),
    })

    gti_header = fits.Header()
    gti_header['CREATED'] = Time.now().utc.iso
    gti_header['HDUCLAS1'] = 'GTI'
    gti_header['OBS_ID'] = np.unique(data_qtable['obs_id'])[0]
    gti_header['MJDREFI'] = ev_header['MJDREFI']
    gti_header['MJDREFF'] = ev_header['MJDREFF']
    gti_header['TIMESYS'] = ev_header['TIMESYS']
    gti_header['TIMEUNIT'] = ev_header['TIMEUNIT']
    gti_header['TIMEREF'] = ev_header['TIMEREF']

    hdu_gti = fits.BinTableHDU(gti_table, header=gti_header, name='GTI')
    hdus.append(hdu_gti)

    # Create a pointing table:
    logger.info('Creating a pointing HDU...')

    pnt_table = QTable({
        'TIME': u.Quantity(data_qtable['timestamp'][0], unit=u.s, ndmin=1),
        'RA_PNT': u.Quantity(data_qtable['pointing_ra'][0].to_value(u.deg), ndmin=1),
        'DEC_PNT': u.Quantity(data_qtable['pointing_dec'][0].to_value(u.deg), ndmin=1),
        'ALT_PNT': u.Quantity(data_qtable['pointing_alt'][0].to_value(u.deg), ndmin=1),
        'AZ_PNT': u.Quantity(data_qtable['pointing_az'][0].to_value(u.deg), ndmin=1),
    })

    pnt_header = fits.Header()
    pnt_header['CREATED'] = Time.now().utc.iso
    pnt_header['HDUCLAS1'] = 'POINTING'
    pnt_header['OBS_ID'] = np.unique(data_qtable['obs_id'])[0]
    pnt_header['MJDREFI'] = ev_header['MJDREFI']
    pnt_header['MJDREFF'] = ev_header['MJDREFF']
    pnt_header['TIMEUNIT'] = ev_header['TIMEUNIT']
    pnt_header['TIMESYS'] = ev_header['TIMESYS']
    pnt_header['OBSGEO-L'] = (28.76177, 'Geographic longitude of telescope (deg)')
    pnt_header["OBSGEO-B"] = (-17.89064, 'Geographic latitude of telescope (deg)')
    pnt_header["OBSGEO-H"] = (2199.835, 'Geographic latitude of telescope (m)')
    pnt_header["TIMEREF"] = ev_header["TIMEREF"]

    hdu_pnt = fits.BinTableHDU(pnt_table, header=pnt_header, name="POINTING")
    hdus.append(hdu_pnt)

    # Add the IRF HDUs:
    logger.info('Adding the IRF HDUs...')
    hdu_irf = fits.open(input_file_irf)
    hdus += hdu_irf[1:]

    # Save in an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    base_name = Path(input_file_dl2).resolve().name
    regex = r'dl2_(\S+)\.h5'

    parser = re.findall(regex, base_name)[0]
    output_file = f'{output_dir}/dl3_{parser}.fits.gz'

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