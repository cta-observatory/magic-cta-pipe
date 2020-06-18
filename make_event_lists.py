#!/usr/bin/env python
# coding: utf-8

import glob
import argparse
import pandas
import yaml
import scipy

import astropy.io.fits as pyfits
from astropy import units
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz

import sys
sys.path.append('../')
import gti



# =================
# === Main code ===
# =================

# --------------------------
# Adding the argument parser
arg_parser = argparse.ArgumentParser(description="""
This tools produces the FITS event lists out of the earlier processed files.
""")

arg_parser.add_argument("--config", default="config.yaml",
                        help='Configuration file to steer the code execution.')

parsed_args = arg_parser.parse_args()
# --------------------------

# ------------------------------
# Reading the configuration file

file_not_found_message = """
Error: can not load the configuration file {:s}.
Please check that the file exists and is of YAML or JSON format.
Exiting.
"""

try:
    config = yaml.load(open(parsed_args.config, "r"))
except IOError:
    print(file_not_found_message.format(parsed_args.config))
    exit()
# ------------------------------


path = config['data_files']['data']['test_sample']['magic1']['reco_output']
df = pandas.read_hdf(path, key='dl3/reco')

obs_ids = df.index.levels[0].values

for obs_id in obs_ids:
    obs_df = df.xs(obs_id, level="obs_id")
    obs_df = obs_df.reset_index()

    # Assiging event IDs
    obs_df['cta_event_id'] = scipy.int64((obs_id << 32) | obs_df['event_id'].values)

    obs_df = obs_df.query(config['event_list']['cuts']['selection'])


    # AltAz -> Eq conversion

    observatory_location = EarthLocation.of_site("Roque de los Muchachos")

    event_times = Time(obs_df["mjd"],
                    format='mjd',
                    location=observatory_location)

    alt_az_frame = AltAz(obstime=event_times, location=observatory_location)

    event_coord_altaz_ref = SkyCoord(alt=scipy.degrees(obs_df['alt_reco_mean']),
                                    az=scipy.degrees(obs_df['az_reco_mean']),
                                    frame=alt_az_frame,
                                    unit='deg')


    event_ra = event_coord_altaz_ref.fk5.ra
    event_dec = event_coord_altaz_ref.fk5.dec

    # Reference time
    time_ref = obs_df['mjd'].min()
    # Applying the time offset
    obs_df['mjd'] = 86400 * (obs_df['mjd'] - time_ref)

    # GTI generation
    file_list = glob.glob(config['data_files']['data']['test_sample']['magic1']['input_mask'])
    file_list.sort()
    file_list = list(filter(lambda name: str(obs_id) in name, file_list))

    gti_generator = gti.GTIGenerator(config, verbose=True)
    obs_gti_list = gti_generator.process_files(file_list)


    # Preparing GTI HDU
    gti_start, gti_stop = zip(*obs_gti_list)

    gti_start = list(map(lambda t: (t-time_ref)*86400, gti_start))
    gti_stop = list(map(lambda t: (t-time_ref)*86400, gti_stop))

    col_tstart = pyfits.Column(name='START', 
                            unit='s', 
                            format='E', 
                            array=gti_start)

    col_tstop = pyfits.Column(name='STOP', 
                            unit='s', 
                            format='E', 
                            array=gti_stop)

    columns = [
        col_tstart, 
        col_tstop,
    ]

    # Creating HDU
    col_defs = pyfits.ColDefs(columns)
    gti_hdu = pyfits.BinTableHDU.from_columns(col_defs)
    gti_hdu.name = 'GTI'

    # Preparing Events HDU

    col_event_id = pyfits.Column(name='EVENT_ID', 
                            unit='', 
                            format='K', 
                            array=obs_df['cta_event_id'].values)

    col_time = pyfits.Column(name='TIME', 
                            unit='s', 
                            format='E', 
                            array=obs_df['mjd'].values)

    col_ra = pyfits.Column(name='RA', 
                        unit='deg', 
                        format='E', 
                        array=event_ra.to(units.deg).value)

    col_dec = pyfits.Column(name='DEC', 
                            unit='deg', 
                            format='E', 
                            array=event_dec.to(units.deg).value)

    col_energy = pyfits.Column(name='ENERGY', 
                            unit='TeV', 
                            format='E', 
                            array=obs_df['energy_reco_mean'].values)

    columns = [
        col_event_id, 
        col_time,
        col_ra,
        col_dec,
        col_energy,
    ]

    # Creating HDU
    colDefs = pyfits.ColDefs(columns)
    events_hdu = pyfits.BinTableHDU.from_columns(colDefs)
    events_hdu.name = 'EVENTS'

    # Events HDU header
    events_hdu.header['HDUDOC'] = 'https://github.com/open-gamma-ray-astro/gamma-astro-data-formats'
    events_hdu.header['HDUVERS'] = '0.2'
    events_hdu.header['HDUCLASS'] = 'EVENTS'
    events_hdu.header['HDUCLAS1'] = ''
    events_hdu.header['HDUCLAS2'] = ''
    events_hdu.header['HDUCLAS3'] = ''
    events_hdu.header['HDUCLAS4'] = ''

    events_hdu.header['OBS_ID'] = obs_id

    events_hdu.header['TSTART'] = 0.0
    events_hdu.header['TSTOP'] = 86400 * (obs_df['mjd'].max() - obs_df['mjd'].min())
    events_hdu.header['ONTIME'] = events_hdu.header['TSTOP'] - events_hdu.header['TSTART']
    events_hdu.header['LIVETIME'] = events_hdu.header['TSTOP'] - events_hdu.header['TSTART']
    events_hdu.header['DEADC'] = events_hdu.header['LIVETIME'] / events_hdu.header['ONTIME']

    events_hdu.header['RA_PNT'] = event_ra.mean().to(units.deg).value
    events_hdu.header['DEC_PNT'] = event_dec.mean().to(units.deg).value

    events_hdu.header['ALT_PNT'] = -1
    events_hdu.header['AZ_PNT'] = -1

    events_hdu.header['EQUINOX'] = 2000.0
    events_hdu.header['RADECSYS'] = 'FK5'
    events_hdu.header['ORIGIN'] = 'MAGIC'
    events_hdu.header['TELESCOP'] = 'MAGIC'
    events_hdu.header['INSTRUME'] = 'M12'
    events_hdu.header['CALDB'] = 'dev'
    events_hdu.header['IRF'] = 'crab'
    events_hdu.header['CREATOR'] = 'MAGIC-ctapipe converter'

    events_hdu.header['MJDREFI'] = int(scipy.floor(time_ref))
    events_hdu.header['MJDREFF'] = time_ref - scipy.floor(time_ref)
    events_hdu.header['TIMEUNIT'] = 's'
    events_hdu.header['TIMESYS'] = 'UTC'
    events_hdu.header['TIMEREF'] = 'LOCAL'

    # Saving to FITS

    output_name = f"events_{obs_id}.fits"

    primary_hdu = pyfits.PrimaryHDU()
            
    hdu_list = pyfits.HDUList([primary_hdu, events_hdu, gti_hdu])
    hdu_list.writeto(output_name, overwrite=True)
