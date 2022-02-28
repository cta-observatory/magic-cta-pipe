#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script processes DL1-stereo data and reconstructs the DL2 parameters (i.e., energy, direction and gammaness) with trained RFs.
So far, the RFs will be applied per telescope combination and per telescope type.

Usage:
$ python lst1_magic_dl1_to_dl2.py
--input-file ./data/dl1_stereo/dl1_stereo_LST-1_MAGIC.Run03265.0040.h5
--input-dir-rfs ./data/rfs
--output-dir ./data/dl2
"""

import re
import glob
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from astropy import units as u
from astropy.time import Time
from ctapipe.instrument import SubarrayDescription
from magicctapipe.utils import (
    set_event_types,
    transform_to_radec,
    save_data_to_hdf,
)
from magicctapipe.reco import (
    EnergyRegressor,
    DirectionRegressor,
    EventClassifier,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

nsec2sec = 1e-9

__all__ = [
    'dl1_to_dl2',
]


def apply_rfs(data, estimator):
    """
    Applies trained RFs to an input DL1-stereo data.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Pandas data frame containing events
    estimator: EnergyRegressor, DirectionRegressor or EventClassifier
        Trained estimator

    Returns
    -------
    reco_params: pandas.core.frame.DataFrame
        Pandas data frame containing the DL2 parameters
    """

    tel_ids = list(estimator.telescope_rfs.keys())

    df = data.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})')
    df.dropna(subset=estimator.feature_names, inplace=True)

    df['multiplicity'] = df.groupby(['obs_id', 'event_id']).size()
    df.query(f'multiplicity == {len(tel_ids)}', inplace=True)

    n_events = len(df.groupby(['obs_id', 'event_id']).size())

    if n_events > 0:
        logger.info(f'--> {n_events} events are found. Applying...')
        reco_params = estimator.predict(df)
    else:
        logger.warning('--> No corresponding events are found. Skipping.')
        reco_params = pd.DataFrame()

    return reco_params


def dl1_to_dl2(input_file, input_dir_rfs, output_dir):
    """
    Processes DL1-stereo data to DL2.

    Parameters
    ----------
    input_file: str
        Path to an input DL1-stereo data file
    input_dir_rfs: str
        Path to a directory where trained RFs are stored
    output_dir: str
        Path to a directory where to save an output DL2 data file
    """

    logger.info('\nLoading the input file:')
    logger.info(input_file)

    input_data = pd.read_hdf(input_file, key='events/params')
    input_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    input_data.sort_index(inplace=True)

    input_data = set_event_types(input_data)

    is_simulation = ('mc_energy' in input_data.columns)

    # Reconstruct energy:
    input_rfs_energy = glob.glob(f'{input_dir_rfs}/*energy*.joblib')
    input_rfs_energy.sort()

    if len(input_rfs_energy) > 0:

        logger.info('\nReconstructing energy...')
        energy_regressor = EnergyRegressor()

        df_reco_energy = pd.DataFrame()

        for input_rfs in input_rfs_energy:

            logger.info(input_rfs)
            energy_regressor.load(input_rfs)

            reco_params = apply_rfs(input_data, energy_regressor)
            df_reco_energy = df_reco_energy.append(reco_params)

        input_data = input_data.join(df_reco_energy)

        del energy_regressor

    # Reconstruct arrival directions:
    input_rfs_direction = glob.glob(f'{input_dir_rfs}/*direction*.joblib')
    input_rfs_direction.sort()

    if len(input_rfs_direction) > 0:

        logger.info('\nReconstructing arrival directions...')
        direction_regressor = DirectionRegressor()

        df_reco_direction = pd.DataFrame()

        for input_rfs in input_rfs_direction:

            logger.info(input_rfs)
            direction_regressor.load(input_rfs)

            reco_params = apply_rfs(input_data, direction_regressor)
            df_reco_direction = df_reco_direction.append(reco_params)

        input_data = input_data.join(df_reco_direction)

        if not is_simulation:

            logger.info('\nTransforming the Alt/Az coordinate to the RA/Dec one...')

            if 'timestamp' in input_data.columns:
                timestamps = Time(input_data['timestamp'].to_numpy(), format='unix', scale='utc')
            else:
                time_sec = input_data['time_sec'].to_numpy()
                time_nanosec = input_data['time_nanosec'].to_numpy() * nsec2sec
                timestamps = Time(time_sec + time_nanosec, format='unix', scale='utc')

            ra_tel, dec_tel = transform_to_radec(
                alt=u.Quantity(input_data['alt_tel'].values, u.rad),
                az=u.Quantity(input_data['az_tel'].values, u.rad),
                timestamp=timestamps,
            )

            reco_ra, reco_dec = transform_to_radec(
                alt=u.Quantity(input_data['reco_alt'].values, u.deg),
                az=u.Quantity(input_data['reco_az'].values, u.deg),
                timestamp=timestamps,
            )

            input_data['ra_tel'] = ra_tel.to(u.deg).value
            input_data['dec_tel'] = dec_tel.to(u.deg).value
            input_data['reco_ra'] = reco_ra.to(u.deg).value
            input_data['reco_dec'] = reco_dec.to(u.deg).value

        del direction_regressor

    # Reconstruct the event types:
    input_rfs_classifier = glob.glob(f'{input_dir_rfs}/*classifier*.joblib')
    input_rfs_classifier.sort()

    if len(input_rfs_classifier) > 0:

        logger.info('\nReconstructing the event types...')
        event_classifier = EventClassifier()

        df_reco_types = pd.DataFrame()

        for input_rfs in input_rfs_classifier:

            logger.info(input_rfs)
            event_classifier.load(input_rfs)

            reco_params = apply_rfs(input_data, event_classifier)
            df_reco_types = df_reco_types.append(reco_params)

        input_data = input_data.join(df_reco_types)

        del event_classifier

    # Save in the output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    base_name = Path(input_file).resolve().name
    regex = r'dl1_stereo_(\S+)\.h5'

    parser = re.findall(regex, base_name)[0]
    output_file = f'{output_dir}/dl2_{parser}.h5'

    input_data.reset_index(inplace=True)
    save_data_to_hdf(input_data, output_file, '/events', 'params')

    subarray = SubarrayDescription.from_hdf(input_file)
    subarray.to_hdf(output_file)

    if is_simulation:
        sim_config = pd.read_hdf(input_file, 'simulation/config')
        save_data_to_hdf(sim_config, output_file, '/simulation', 'config')

    logger.info('\nOutput file:')
    logger.info(output_file)


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str,
        help='Path to an input DL1-stereo data file.',
    )

    parser.add_argument(
        '--input-dir-rfs', '-r', dest='input_dir_rfs', type=str,
        help='Path to a directory where trained RFs are stored.',
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='./data',
        help='Path to a directory where to save an output DL2 data.',
    )

    args = parser.parse_args()

    dl1_to_dl2(args.input_file, args.input_dir_rfs, args.output_dir)

    logger.info('\nDone.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
