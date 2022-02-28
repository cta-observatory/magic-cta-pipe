#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script reconstructs the stereo parameters of the events containing more than one telescope information.
The cuts specified in a configuration file apply to events before the reconstruction. When an input file is
"real" data containing LST-1 and MAGIC events, the script checks the angular distance of the LST-1 and MAGIC
pointing directions. Then, if the distance is more than 0.1 degree, it stops the process to avoid the reconstruction
of mis-pointing data. For example, the event coincidence can happen even though wobble offsets are different.

Usage:
$ python lst1_magic_stereo_reco.py
--input-file ./data/dl1_coincidence/dl1_LST-1_MAGIC.Run03265.0040.h5
--output-dir ./data/dl1_stereo
--config-file ./config.yaml
"""

import re
import sys
import time
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from astropy import units as u
from astropy.coordinates import (
    Angle,
    SkyCoord,
    angular_separation,
)
from ctapipe.reco import HillasReconstructor
from ctapipe.containers import (
    ArrayEventContainer,
    ImageParametersContainer,
    CameraHillasParametersContainer,
)
from ctapipe.instrument import SubarrayDescription
from magicctapipe.utils import (
    set_event_types,
    calc_impact,
    save_data_to_hdf,
    calc_mean_direction,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

tel_id_lst = 1

theta_uplim = u.Quantity(0.1, u.deg)

__all__ = [
    'stereo_reco',
]


def check_angular_distance(input_data, theta_uplim):
    """
    Checks the angular distance of the LST-1 and MAGIC
    pointing directions.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Pandas data frame containing LST-1 and MAGIC events
    theta_uplim: astropy.units.quantity.Quantity
        Upper limit of the angular distance
    """

    df_lst = input_data.query('tel_id == 1')
    obs_ids_joint = df_lst.index.get_level_values('obs_id').tolist()
    event_ids_joint = df_lst.index.get_level_values('event_id').tolist()

    multi_indices = pd.MultiIndex.from_arrays(
        [obs_ids_joint, event_ids_joint], names=['obs_id', 'event_id'],
    )

    df_magic = input_data.query('tel_id == [2, 3]')
    df_magic.reset_index(level='tel_id', inplace=True)
    df_magic = df_magic.loc[multi_indices]

    az_magic_mean, alt_magic_mean = calc_mean_direction(df_magic['az_tel'], df_magic['alt_tel'])

    theta = angular_separation(
        lon1=u.Quantity(df_lst['az_tel'].to_numpy(), u.rad),
        lat1=u.Quantity(df_lst['alt_tel'].to_numpy(), u.rad),
        lon2=u.Quantity(az_magic_mean.to_numpy(), u.rad),
        lat2=u.Quantity(alt_magic_mean.to_numpy(), u.rad),
    )

    n_events_sep = np.sum(theta > theta_uplim)

    if n_events_sep > 0:
        logger.info(f'--> The pointing directions are separated by more than {theta_uplim.value} degree. ' \
                    'The data would be taken by different wobble offsets. Please check the input data. Exiting.\n')
        sys.exit()
    else:
        angle_max = np.max(theta.to(u.arcmin).value)
        logger.info(f'--> Maximum angular distance is {angle_max:.3f} arcmin. Continue.')


def stereo_reco(input_file, output_dir, config):
    """
    Reconstructs the stereo parameters of the events
    containing more than one telescope information.

    Parameters
    ----------
    input_file: str
        Path to an input DL1 data file
    output_dir: str
        Path to a directory where to save an output DL1-stereo data file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    logger.info('\nLoading the input file:')
    logger.info(input_file)

    input_data = pd.read_hdf(input_file, key='events/params')
    input_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    input_data.sort_index(inplace=True)

    input_data['multiplicity'] = input_data.groupby(['obs_id', 'event_id']).size()
    input_data.query('multiplicity > 1', inplace=True)

    n_events = len(input_data.groupby(['obs_id', 'event_id']).size())
    logger.info(f'--> {n_events} stereo events')

    is_simulation = ('mc_energy' in input_data.columns)

    # Read the subarray description:
    subarray = SubarrayDescription.from_hdf(input_file)
    tel_positions = subarray.positions

    logger.info('\nSubarray configuration:')
    for tel_id in subarray.tel.keys():
        logger.info(f'Telescope {tel_id}: {subarray.tel[tel_id].name}, position = {tel_positions[tel_id]}')

    # Apply the cuts before the reconstruction:
    event_cuts = config['stereo_reco']['event_cuts']

    logger.info('\nApplying the following cuts:')
    logger.info(event_cuts)

    input_data.query(event_cuts, inplace=True)
    input_data['multiplicity'] = input_data.groupby(['obs_id', 'event_id']).size()
    input_data.query('multiplicity > 1', inplace=True)

    input_data = set_event_types(input_data)

    # Check the angular distance of the pointing directions:
    telescope_ids = np.unique(input_data.index.get_level_values('tel_id'))

    if (not is_simulation) and (tel_id_lst in telescope_ids):
        logger.info('\nChecking the angular distance of the LST-1 and MAGIC pointing directions...')
        check_angular_distance(input_data)

    # Process the events:
    logger.info('\nReconstructing the stereo parameters...')

    event = ArrayEventContainer()
    hillas_reconstructor = HillasReconstructor(subarray)

    az_mean, alt_mean = calc_mean_direction(input_data)

    group = input_data.groupby(['obs_id', 'event_id']).size()

    observation_ids = group.index.get_level_values('obs_id')
    event_ids = group.index.get_level_values('event_id')

    # Loop over observation/event IDs.
    # Since the HillasReconstructor requires the ArrayEventContainer,
    # here we set the necessary information to the container event-by-event:
    for i_ev, (obs_id, ev_id) in enumerate(zip(observation_ids, event_ids)):

        if i_ev % 100 == 0:
            logger.info(f'{i_ev} events')

        df_ev = input_data.query(f'(obs_id == {obs_id}) & (event_id == {ev_id})')

        event.pointing.array_altitude = u.Quantity(df_ev['alt_tel_mean'].iloc[0], u.rad)
        event.pointing.array_azimuth = u.Quantity(df_ev['az_tel_mean'].iloc[0], u.rad)

        telescope_ids = df_ev.index.get_level_values('tel_id')

        for tel_id in telescope_ids:

            df_tel = df_ev.query(f'tel_id == {tel_id}')

            event.pointing.tel[tel_id].altitude = u.Quantity(alt_mean.iloc[i_ev], u.rad)
            event.pointing.tel[tel_id].azimuth = u.Quantity(az_mean.iloc[i_ev], u.rad)

            hillas_params = CameraHillasParametersContainer(
                intensity=float(df_tel['intensity'].iloc[0]),
                x=u.Quantity(df_tel['x'].iloc[0], u.m),
                y=u.Quantity(df_tel['y'].iloc[0], u.m),
                r=u.Quantity(df_tel['r'].iloc[0], u.m),
                phi=Angle(df_tel['phi'].iloc[0], u.deg),
                length=u.Quantity(df_tel['length'].iloc[0], u.m),
                width=u.Quantity(df_tel['width'].iloc[0], u.m),
                psi=Angle(df_tel['psi'].iloc[0], u.deg),
                skewness=float(df_tel['skewness'].iloc[0]),
                kurtosis=float(df_tel['kurtosis'].iloc[0]),
            )

            event.dl1.tel[tel_id].parameters = ImageParametersContainer(hillas=hillas_params)

        # Reconstruct the stereo parameters:
        hillas_reconstructor(event)
        stereo_params = event.dl2.stereo.geometry['HillasReconstructor']

        if stereo_params.az < 0:
            stereo_params.az += u.Quantity(360, u.deg)

        for tel_id in telescope_ids:

            # Calculate the impact parameter:
            impact = calc_impact(
                core_x=stereo_params.core_x,
                core_y=stereo_params.core_y,
                az=stereo_params.az,
                alt=stereo_params.alt,
                tel_pos_x=tel_positions[tel_id][0],
                tel_pos_y=tel_positions[tel_id][1],
                tel_pos_z=tel_positions[tel_id][2],
            )

            # Set the stereo parameters:
            input_data.loc[(obs_id, ev_id, tel_id), 'h_max'] = stereo_params.h_max.to(u.m).value
            input_data.loc[(obs_id, ev_id, tel_id), 'alt'] = stereo_params.alt.to(u.deg).value
            input_data.loc[(obs_id, ev_id, tel_id), 'alt_uncert'] = stereo_params.alt_uncert.to(u.deg).value
            input_data.loc[(obs_id, ev_id, tel_id), 'az'] = stereo_params.az.to(u.deg).value
            input_data.loc[(obs_id, ev_id, tel_id), 'az_uncert'] = stereo_params.az_uncert.to(u.deg).value
            input_data.loc[(obs_id, ev_id, tel_id), 'core_x'] = stereo_params.core_x.to(u.m).value
            input_data.loc[(obs_id, ev_id, tel_id), 'core_y'] = stereo_params.core_y.to(u.m).value
            input_data.loc[(obs_id, ev_id, tel_id), 'impact'] = impact.to(u.m).value

    n_events_processed = i_ev + 1
    logger.info(f'{n_events_processed} events')

    # Save in an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    base_name = Path(input_file).resolve().name
    regex = r'dl1_(\S+)\.h5'

    parser = re.findall(regex, base_name)[0]
    output_file = f'{output_dir}/dl1_stereo_{parser}.h5'

    input_data.reset_index(inplace=True)
    save_data_to_hdf(input_data, output_file, '/events', 'params')

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
        '--input-file', '-i', dest='input_file', type=str, required=True,
        help='Path to an input DL1 data file.',
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='./data',
        help='Path to a directory where to save an output DL1-stereo data file.',
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
       help='Path to a yaml configuration file.',
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    stereo_reco(args.input_file, args.output_dir, config)

    logger.info('\nDone.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
