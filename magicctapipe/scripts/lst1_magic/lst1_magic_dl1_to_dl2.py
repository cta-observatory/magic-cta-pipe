#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

Reconstruct the DL2 parameters (i.e., energy, direction and gammaness) with trained RFs.
The RFs will be applied per telescope combination and per telescope type.
If real data is input, the parameters in the Alt/Az coordinate will be transformed to the RA/Dec coordinate.

Usage:
$ python lst1_magic_dl1_to_dl2.py
--input-file "./data/dl1_stereo/dl1_stereo_lst1_magic_run03265.0040.h5"
--output-file "./data/dl2/dl2_lst1_magic_run03265.0040.h5"
--energy-regressors "./data/rfs/energy_regressors_*.joblib"
--direction-regressors "./data/rfs/direction_regressors_*.joblib"
--event-classifiers "./data/rfs/event_classifiers_*.joblib"
"""

import glob
import time
import tables
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time

from ctapipe.instrument import SubarrayDescription

from magicctapipe.reco import (
    EnergyRegressor,
    DirectionRegressor,
    EventClassifier,
)

from magicctapipe.utils import transform_to_radec


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

warnings.simplefilter('ignore')

__all__ = ['dl1_to_dl2']


def apply_rfs(data, rfs_mask, estimator, tel_descriptions=None):

    file_paths = glob.glob(rfs_mask)
    file_paths.sort()

    reco_params = pd.DataFrame()

    for path in file_paths:

        logger.info(path)
        estimator.load(path)

        tel_ids = list(estimator.telescope_rfs.keys())

        df = data.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})')
        df.dropna(subset=estimator.feature_names, inplace=True)
        df['multiplicity'] = df.groupby(['obs_id', 'event_id']).size()
        df.query(f'multiplicity == {len(tel_ids)}', inplace=True)

        n_events = len(df.groupby(['obs_id', 'event_id']).size())

        if n_events == 0:
            logger.info('--> No corresponding events are found. Skipping.')
            continue

        logger.info(f'--> {n_events} events are found. Applying...')

        if tel_descriptions is not None:
            df_reco = estimator.predict(df, tel_descriptions)
        else:
            df_reco = estimator.predict(df)

        reco_params = reco_params.append(df_reco)

    reco_params.sort_index(inplace=True)

    return reco_params


def dl1_to_dl2(
        input_file, output_file,
        energy_regressors=None, direction_regressors=None, event_classifiers=None
    ):

    logger.info(f'\nLoading the input data file:\n{input_file}')

    data_joint = pd.read_hdf(input_file, key='events/params')
    data_joint.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    data_joint.sort_index(inplace=True)

    data_type = 'mc' if ('mc_energy' in data_joint.columns) else 'real'
    subarray = SubarrayDescription.from_hdf(input_file)

    # --- reconstruct energy ---
    if energy_regressors is not None:

        estimator = EnergyRegressor()
        logger.info('\nReconstucting the energy...')

        reco_params = apply_rfs(data_joint, energy_regressors, estimator)
        data_joint = data_joint.join(reco_params)

    # --- reconstruct direction ---
    if direction_regressors is not None:

        estimator = DirectionRegressor()
        logger.info('\nReconstructing the direction...')

        tel_descriptions = subarray.tel

        reco_params = apply_rfs(data_joint, direction_regressors, estimator, tel_descriptions)
        data_joint = data_joint.join(reco_params)

        if data_type == 'real':

            logger.info('Transforming Alt/Az to RA/Dec coordinate...\n')

            timestamps = Time(data_joint['timestamp'].values, format='unix', scale='utc')

            ra_tel, dec_tel = transform_to_radec(
                alt=u.Quantity(data_joint['alt_tel'].values, u.rad),
                az=u.Quantity(data_joint['az_tel'].values, u.rad),
                timestamp=timestamps
            )

            ra_tel_mean, dec_tel_mean = transform_to_radec(
                alt=u.Quantity(data_joint['alt_tel_mean'].values, u.rad),
                az=u.Quantity(data_joint['az_tel_mean'].values, u.rad),
                timestamp=timestamps
            )

            reco_ra, reco_dec = transform_to_radec(
                alt=u.Quantity(data_joint['reco_alt'].values, u.deg),
                az=u.Quantity(data_joint['reco_az'].values, u.deg),
                timestamp=timestamps
            )

            reco_ra_mean, reco_dec_mean = transform_to_radec(
                alt=u.Quantity(data_joint['reco_alt_mean'].values, u.deg),
                az=u.Quantity(data_joint['reco_az_mean'].values, u.deg),
                timestamp=timestamps
            )

            data_joint['ra_tel'] = ra_tel.to(u.deg).value
            data_joint['dec_tel'] = dec_tel.to(u.deg).value
            data_joint['ra_tel_mean'] = ra_tel_mean.to(u.deg).value
            data_joint['dec_tel_mean'] = dec_tel_mean.to(u.deg).value
            data_joint['reco_ra'] = reco_ra.to(u.deg).value
            data_joint['reco_dec'] = reco_dec.to(u.deg).value
            data_joint['reco_ra_mean'] = reco_ra_mean.to(u.deg).value
            data_joint['reco_dec_mean'] = reco_dec_mean.to(u.deg).value

    # --- classify event type ---
    if event_classifiers is not None:

        estimator = EventClassifier()
        logger.info('\nClassifying the event type...')

        reco_params = apply_rfs(data_joint, event_classifiers, estimator)
        data_joint = data_joint.join(reco_params)

    # --- save the data frame ---
    with tables.open_file(output_file, mode='w') as f_out:

        data_joint.reset_index(inplace=True)
        event_values = [tuple(array) for array in data_joint.to_numpy()]
        dtypes = np.dtype([(name, dtype) for name, dtype in zip(data_joint.dtypes.index, data_joint.dtypes)])

        event_table = np.array(event_values, dtype=dtypes)
        f_out.create_table('/events', 'params', createparents=True, obj=event_table)

        if data_type == 'mc':
            with tables.open_file(input_file) as f_in:
                sim_table = f_in.root.simulation.config.read()
                f_out.create_table('/simulation', 'config', createparents=True, obj=sim_table)

    subarray.to_hdf(output_file)

    logger.info(f'\nOutput data file: {output_file}')
    logger.info('\nDone.')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str,
        help='Path to an input DL1-stereo data file.'
    )

    parser.add_argument(
        '--output-file', '-o', dest='output_file', type=str, default='./dl2.h5',
        help='Path to an output DL2 data file.'
    )

    parser.add_argument(
        '--energy-regressors', '-e', dest='energy_regressors', type=str, default=None,
        help='Path to trained energy regressors.'
    )

    parser.add_argument(
        '--direction-regressors', '-d', dest='direction_regressors', type=str, default=None,
        help='Path to trained direction regressors.'
    )

    parser.add_argument(
        '--event-classifiers', '-c', dest='event_classifiers', type=str, default=None,
        help='Path to trained event classifiers.'
    )

    args = parser.parse_args()

    dl1_to_dl2(
        args.input_file, args.output_file,
        args.energy_regressors, args.direction_regressors, args.event_classifiers
    )

    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
