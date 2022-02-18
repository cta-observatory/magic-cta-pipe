#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script reconstructs the DL2 parameters (i.e., energy, direction and gammaness) with trained RFs.
The RFs will be applied per telescope combination and per telescope type.

Usage:
$ python lst1_magic_dl1_to_dl2.py
--input-file ./data/dl1_stereo/dl1_stereo_lst1_magic_run03265.0040.h5
--input-dir-rfs ./data/rfs
--output-dir ./data/dl2
"""

import re
import glob
import time
import tables
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from astropy import units as u
from astropy.time import Time
from ctapipe.instrument import SubarrayDescription
from magicctapipe.utils import transform_to_radec
from magicctapipe.reco import (
    EnergyRegressor,
    DirectionRegressor,
    EventClassifier,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

warnings.simplefilter('ignore')

__all__ = [
    'dl1_to_dl2',
]


def apply_rfs(data, estimator, subarray=None):
    """
    Applies input RFs to an input DL1-stere data.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Pandas data frame containing feature parameters
    estimator: EnergyRegressor, DirectionRegressor or EventClassifier
        trained energy, direction or event class estimators
    subarray: SubarrayDescription
        Subarray description of an input data

    Returns
    -------
    reco_params: pandas.core.frame.DataFrame
        Pandas data frame containing the DL2 parameters
    """

    tel_ids = list(estimator.telescope_rfs.keys())
    feature_names = estimator.feature_names

    df = data.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})')
    df.dropna(subset=feature_names, inplace=True)

    df['multiplicity'] = df.groupby(['obs_id', 'event_id']).size()
    df.query(f'multiplicity == {len(tel_ids)}', inplace=True)

    n_events = len(df)

    if n_events == 0:
        logger.warining('--> No corresponding events are found. Skipping.')
        reco_params = pd.DataFrame()

    else:
        logger.info(f'--> {n_events} events are found. Applying...')

        if subarray is not None:
            tel_descriptions = subarray.tel
            reco_params = estimator.predict(df, tel_descriptions)
        else:
            reco_params = estimator.predict(df)

    return reco_params


def dl1_to_dl2(input_file, input_dir_rfs, output_dir):
    """
    Processes input DL1-stereo data to DL2.

    Parameters
    ----------
    input_file: str
        Path to an input DL1-stereo data file
    input_dir_rfs: str
        Path to a directory where trained RFs are stored
    output_dir: str
        Path to a directory where to save an output DL2 data file
    """

    logger.info(f'\nLoading the input data file:\n{input_file}')

    data_joint = pd.read_hdf(input_file, key='events/params')
    data_joint.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    data_joint.sort_index(inplace=True)

    is_simulation = ('mc_energy' in data_joint.columns)

    subarray = SubarrayDescription.from_hdf(input_file)

    # Reconstruct energy:
    input_rfs_energy = glob.glob(f'{input_dir_rfs}/*energy*.joblib')
    input_rfs_energy.sort()

    if len(input_rfs_energy) > 0:

        logger.info('\nReconstucting energy...')
        energy_regressor = EnergyRegressor()

        df_reco_energy = pd.DataFrame()

        for input_rfs in input_rfs_energy:

            logger.info(input_rfs)
            energy_regressor.load(input_rfs)

            reco_params = apply_rfs(data_joint, energy_regressor)
            df_reco_energy = df_reco_energy.append(reco_params)

        data_joint = data_joint.join(df_reco_energy)

        del energy_regressor

    # Reconstruct arrival direction:
    input_rfs_direction = glob.glob(f'{input_dir_rfs}/*direction*.joblib')
    input_rfs_direction.sort()

    if len(input_rfs_direction) > 0:

        logger.info('\nReconstucting arrival direction...')
        direction_regressor = DirectionRegressor()

        df_reco_direction = pd.DataFrame()

        for input_rfs in input_rfs_direction:

            logger.info(input_rfs)
            direction_regressor.load(input_rfs)

            reco_params = apply_rfs(data_joint, direction_regressor, subarray)
            df_reco_direction = df_reco_direction.append(reco_params)

        data_joint = data_joint.join(df_reco_direction)

        if not is_simulation:

            logger.info('Transforming Alt/Az to RA/Dec coordinate...\n')

            timestamps = Time(data_joint['timestamp'].values, format='unix', scale='utc')

            ra_tel, dec_tel = transform_to_radec(
                alt=u.Quantity(data_joint['alt_tel'].values, u.rad),
                az=u.Quantity(data_joint['az_tel'].values, u.rad),
                timestamp=timestamps,
            )

            ra_tel_mean, dec_tel_mean = transform_to_radec(
                alt=u.Quantity(data_joint['alt_tel_mean'].values, u.rad),
                az=u.Quantity(data_joint['az_tel_mean'].values, u.rad),
                timestamp=timestamps,
            )

            reco_ra, reco_dec = transform_to_radec(
                alt=u.Quantity(data_joint['reco_alt'].values, u.deg),
                az=u.Quantity(data_joint['reco_az'].values, u.deg),
                timestamp=timestamps,
            )

            reco_ra_mean, reco_dec_mean = transform_to_radec(
                alt=u.Quantity(data_joint['reco_alt_mean'].values, u.deg),
                az=u.Quantity(data_joint['reco_az_mean'].values, u.deg),
                timestamp=timestamps,
            )

            data_joint['ra_tel'] = ra_tel.to(u.deg).value
            data_joint['dec_tel'] = dec_tel.to(u.deg).value
            data_joint['ra_tel_mean'] = ra_tel_mean.to(u.deg).value
            data_joint['dec_tel_mean'] = dec_tel_mean.to(u.deg).value
            data_joint['reco_ra'] = reco_ra.to(u.deg).value
            data_joint['reco_dec'] = reco_dec.to(u.deg).value
            data_joint['reco_ra_mean'] = reco_ra_mean.to(u.deg).value
            data_joint['reco_dec_mean'] = reco_dec_mean.to(u.deg).value

        del direction_regressor

    # Reconstructing the event types:
    input_rfs_classifier = glob.glob(f'{input_dir_rfs}/*classifier*.joblib')
    input_rfs_classifier.sort()

    if len(input_rfs_classifier) > 0:

        logger.info('\nReconstructing the event types...')
        event_classifier = EventClassifier()

        df_reco_types = pd.DataFrame()

        for input_rfs in input_rfs_classifier:

            logger.info(input_rfs)
            event_classifier.load(input_rfs)

            reco_params = apply_rfs(data_joint, event_classifier)
            df_reco_types = df_reco_types.append(reco_params)

        data_joint = data_joint.join(df_reco_types)

        del event_classifier

    # Prepare for saving the data to an output file.
    # Here we parse run information from the input file name:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    base_name = Path(input_file).resolve().name
    regex_run = r'dl1_stereo_(\S+)_run(\d+)\.h5'
    regex_subrun = rf'dl1_stereo_(\S+)_run(\d+)\.(\d+)\.h5'

    if re.fullmatch(regex_run, base_name):
        parser = re.findall(regex_run, base_name)[0]
        output_file = f'{output_dir}/dl2_{parser[0]}_run{parser[1]}.h5'

    elif re.fullmatch(regex_subrun, base_name):
        parser = re.findall(regex_subrun, base_name)[0]
        output_file = f'{output_dir}/dl2_{parser[0]}_run{parser[1]}.{parser[2]}.h5'

    # Save in the output file:
    with tables.open_file(output_file, mode='w') as f_out:

        data_joint.reset_index(inplace=True)
        event_values = [tuple(array) for array in data_joint.to_numpy()]
        dtypes = np.dtype([(name, dtype) for name, dtype in zip(data_joint.dtypes.index, data_joint.dtypes)])

        event_table = np.array(event_values, dtype=dtypes)
        f_out.create_table('/events', 'params', createparents=True, obj=event_table)

        if is_simulation:
            with tables.open_file(input_file) as f_in:
                sim_table = f_in.root.simulation.config.read()
                f_out.create_table('/simulation', 'config', createparents=True, obj=sim_table)

    # Save the subarray description:
    subarray.to_hdf(output_file)

    logger.info(f'\nOutput file:\n{output_file}')
    logger.info('\nDone.')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str,
        help='Path to an input DL1-stereo data file.'
    )

    parser.add_argument(
        '--input-dir-rfs', '-r', dest='input_dir_rfs', type=str,
        help='Path to a directory where trained estimators are stored.'
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='./data',
        help='Path to a directory where to save an output DL2 data.'
    )

    args = parser.parse_args()

    dl1_to_dl2(
        input_file=args.input_file,
        input_dir_rfs=args.input_dir_rfs,
        output_dir=args.output_dir,
    )

    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
