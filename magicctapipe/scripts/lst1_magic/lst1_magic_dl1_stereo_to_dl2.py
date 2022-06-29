#!/usr/bin/env python
# coding: utf-8

"""
This script processes DL1-stereo events and reconstructs the DL2 parameters
(i.e., energy, direction and gammaness) with trained RFs.
The RFs are currently applied per telescope combination and per telescope type.

Usage:
$ python lst1_magic_dl1_stereo_to_dl2.py
--input-file-dl1 ./data/dl1_stereo/dl1_stereo_LST-1_MAGIC.Run03265.0040.h5
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
from magicctapipe.reco import (
    EnergyRegressor,
    DirectionRegressor,
    EventClassifier,
)
from magicctapipe.utils import (
    check_tel_combination,
    transform_altaz_to_radec,
    save_pandas_to_table,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

nsec2sec = 1e-9

__all__ = [
    'apply_rfs',
    'dl1_stereo_to_dl2',
]


def apply_rfs(event_data, estimator):
    """
    Applies trained RFs to input DL1-stereo data.

    Parameters
    ----------
    event_data: pandas.core.frame.DataFrame
        Pandas data frame of shower events
    estimator: magicctapipe.reco.estimator
        Trained regressor or classifier

    Returns
    -------
    reco_params: pandas.core.frame.DataFrame
        Pandas data frame of the DL2 parameters
    """

    tel_ids = list(estimator.telescope_rfs.keys())

    df_events = event_data.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})').copy()
    df_events['multiplicity'] = df_events.groupby(['obs_id', 'event_id']).size()
    df_events.query(f'multiplicity == {len(tel_ids)}', inplace=True)

    n_events = len(df_events.groupby(['obs_id', 'event_id']).size())

    if n_events > 0:
        logger.info(f'--> {n_events} events are found. Applying...')
        reco_params = estimator.predict(df_events)
    else:
        logger.warning('--> No corresponding events are found. Skipping.')
        reco_params = pd.DataFrame()

    return reco_params


def dl1_stereo_to_dl2(input_file_dl1, input_dir_rfs, output_dir):
    """
    Processes DL1-stereo events and
    reconstructs the DL2 parameters with trained RFs.

    Parameters
    ----------
    input_file_dl1: str
        Path to an input DL1-stereo data file
    input_dir_rfs: str
        Path to a directory where trained RFs are stored
    output_dir: str
        Path to a directory where to save an output DL2 data file
    """

    logger.info('\nLoading the input DL1-stereo data file:')
    logger.info(input_file_dl1)

    event_data = pd.read_hdf(input_file_dl1, key='events/parameters')
    event_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    event_data.sort_index(inplace=True)

    check_tel_combination(event_data)

    is_simulation = ('true_energy' in event_data.columns)

    # Reconstruct energy:
    mask_energy_rf = f'{input_dir_rfs}/energy_regressor*.joblib'

    input_rfs_energy = glob.glob(mask_energy_rf)
    input_rfs_energy.sort()

    if len(input_rfs_energy) > 0:
        logger.info('\nReconstructing energy...')

        reco_params = pd.DataFrame()
        energy_regressor = EnergyRegressor()

        for input_rfs in input_rfs_energy:

            logger.info(input_rfs)

            energy_regressor.load(input_rfs)
            df_reco_energy = apply_rfs(event_data, energy_regressor)

            reco_params = reco_params.append(df_reco_energy)

        event_data = event_data.join(reco_params)

        del energy_regressor

    # Reconstruct arrival directions:
    mask_direction_rf = f'{input_dir_rfs}/direction_regressor*.joblib'

    input_rfs_direction = glob.glob(mask_direction_rf)
    input_rfs_direction.sort()

    if len(input_rfs_direction) > 0:
        logger.info('\nReconstructing arrival directions...')

        reco_params = pd.DataFrame()
        direction_regressor = DirectionRegressor()

        for input_rfs in input_rfs_direction:

            logger.info(input_rfs)

            direction_regressor.load(input_rfs)
            df_reco_direction = apply_rfs(event_data, direction_regressor)

            reco_params = reco_params.append(df_reco_direction)

        event_data = event_data.join(reco_params)

        if not is_simulation:
            logger.info('\nTransforming the Alt/Az coordinate to the RA/Dec coordinate...')

            if 'timestamp' in event_data.columns:
                timestamps = Time(event_data['timestamp'].to_numpy(), format='unix', scale='utc')

            else:
                time_sec = event_data['time_sec'].to_numpy()
                time_nanosec = event_data['time_nanosec'].to_numpy() * nsec2sec

                timestamps = Time(time_sec + time_nanosec, format='unix', scale='utc')

                event_data['timestamp'] = timestamps.value
                event_data.drop(columns=['time_sec', 'time_nanosec'], inplace=True)

            pointing_ra, pointing_dec = transform_altaz_to_radec(
                alt=u.Quantity(event_data['pointing_alt'].values, u.rad),
                az=u.Quantity(event_data['pointing_az'].values, u.rad),
                timestamp=timestamps,
            )

            reco_ra, reco_dec = transform_altaz_to_radec(
                alt=u.Quantity(event_data['reco_alt'].values, u.deg),
                az=u.Quantity(event_data['reco_az'].values, u.deg),
                timestamp=timestamps,
            )

            event_data['pointing_ra'] = pointing_ra.to(u.deg).value
            event_data['pointing_dec'] = pointing_dec.to(u.deg).value
            event_data['reco_ra'] = reco_ra.to(u.deg).value
            event_data['reco_dec'] = reco_dec.to(u.deg).value

        del direction_regressor

    # Reconstruct the gammaness:
    mask_classifier = f'{input_dir_rfs}/event_classifier*.joblib'

    input_rfs_classifier = glob.glob(mask_classifier)
    input_rfs_classifier.sort()

    if len(input_rfs_classifier) > 0:
        logger.info('\nReconstructing the gammaness...')

        reco_params = pd.DataFrame()
        event_classifier = EventClassifier()

        for input_rfs in input_rfs_classifier:

            logger.info(input_rfs)

            event_classifier.load(input_rfs)
            df_reco_class = apply_rfs(event_data, event_classifier)

            reco_params = reco_params.append(df_reco_class)

        event_data = event_data.join(reco_params)

        del event_classifier

    # Save the data in an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    regex = r'dl1_stereo_(\S+)\.h5'
    file_name = Path(input_file_dl1).name

    if re.fullmatch(regex, file_name):
        parser = re.findall(regex, file_name)[0]
        output_file = f'{output_dir}/dl2_{parser}.h5'
    else:
        raise RuntimeError('Could not parse information from the input file name.')

    event_data.reset_index(inplace=True)
    save_pandas_to_table(event_data, output_file, group_name='/events', table_name='parameters', mode='w')

    subarray = SubarrayDescription.from_hdf(input_file_dl1)
    subarray.to_hdf(output_file)

    if is_simulation:
        sim_config = pd.read_hdf(input_file_dl1, key='simulation/config')
        save_pandas_to_table(sim_config, output_file, group_name='/simulation', table_name='config', mode='a')

    logger.info('\nOutput file:')
    logger.info(output_file)


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file-dl1', '-d', dest='input_file_dl1', type=str, required=True,
        help='Path to an input DL1-stereo data file.',
    )

    parser.add_argument(
        '--input-dir-rfs', '-r', dest='input_dir_rfs', type=str, required=True,
        help='Path to a directory where trained RFs are stored.',
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='./data',
        help='Path to a directory where to save an output DL2 data file.',
    )

    args = parser.parse_args()

    # Process the input data:
    dl1_stereo_to_dl2(args.input_file_dl1, args.input_dir_rfs, args.output_dir)

    logger.info('\nDone.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
