#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script reconstructs the stereo parameters of the events containing more than one telescope information.
The cuts specified in a configuration file apply to events before the reconstruction. The script checks the
angular distance of the pointing directions when an input file is "real" data containing LST-1 and MAGIC events.
Then, the process stops when it is more than 0.1 degree to avoid the reconstruction of mispointing data.
For example, the event coincidence can happen even though the wobble offsets are different.

Usage:
$ python lst1_magic_stereo_reco.py
--input-file ./data/dl1_coincidence/dl1_lst1_magic_run03265.0040.h5
--output-dir ./data/dl1_stereo
--config-file ./config.yaml
"""

import re
import sys
import time
import yaml
import tables
import logging
import argparse
import warnings
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
from magicctapipe.utils import calc_impact

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

warnings.simplefilter('ignore')

tel_id_lst = 1

theta_lim = u.Quantity(0.1, u.deg)

tel_combinations = {
    'm1_m2': [2, 3],   # event_type = 0
    'lst1_m1': [1, 2],   # event_type = 1
    'lst1_m2': [1, 3],   # event_type = 2
    'lst1_m1_m2': [1, 2, 3],   # event_type = 3
}

__all__ = [
    'stereo_reco',
]


def calc_tel_mean_pointing(data):
    """
    This function calculates the mean telescope pointing direction.

    Parameters
    ----------
    data: pandas.core.frame.DataFrame
        Pandas data frame containing the telescope pointing directions

    Returns
    -------
    pointing: pandas.core.frame.DataFrame
        Pandas data frame containing the mean pointing direction
    """

    x_coords = np.cos(data['alt_tel']) * np.cos(data['az_tel'])
    y_coords = np.cos(data['alt_tel']) * np.sin(data['az_tel'])
    z_coords = np.sin(data['alt_tel'])

    x_coords_mean = x_coords.groupby(['obs_id', 'event_id']).mean()
    y_coords_mean = y_coords.groupby(['obs_id', 'event_id']).mean()
    z_coords_mean = z_coords.groupby(['obs_id', 'event_id']).mean()

    coord_mean = SkyCoord(
        x=x_coords_mean.to_numpy(),
        y=y_coords_mean.to_numpy(),
        z=z_coords_mean.to_numpy(),
        representation_type='cartesian',
    )

    pointing = pd.DataFrame(
        data={'alt_tel_mean': coord_mean.spherical.lat.to(u.rad).value,
              'az_tel_mean': coord_mean.spherical.lon.to(u.rad).value},
        index=data.groupby(['obs_id', 'event_id']).mean().index,
    )

    return pointing


def stereo_reco(input_file, output_dir, config):
    """
    This function reconstructs the stereo parameters of
    the events containing more than one telescope information.

    Parameters
    ----------
    input_file: str
        Path to an input DL1 data file
    output_dir: str
        Path to a directory where to save an output DL1-stereo data file
    config: dict
        Configuration for LST-1 + MAGIC analysis
    """

    logger.info(f'\nLoading the input data file:\n{input_file}')

    data_joint = pd.read_hdf(input_file, key='events/params')
    data_joint.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    data_joint.sort_index(inplace=True)

    data_joint['multiplicity'] = data_joint.groupby(['obs_id', 'event_id']).size()
    data_joint.query('multiplicity > 1', inplace=True)

    n_events = len(data_joint.groupby(['obs_id', 'event_id']).size())
    logger.info(f'--> {n_events} stereo events')

    is_simulation = ('mc_energy' in data_joint.columns)

    # Read the subarray description:
    subarray = SubarrayDescription.from_hdf(input_file)
    tel_positions = subarray.positions

    logger.info('\nSubarray configuration:')
    for tel_id in subarray.tel.keys():
        logger.info(f'Telescope {tel_id}: {subarray.tel[tel_id].name}, position = {tel_positions[tel_id]}')

    # Apply the cuts before the reconstruction:
    event_cuts = config['stereo_reco']['event_cuts']

    logger.info(f'\nApplying the following cuts:\n{event_cuts}')
    data_joint.query(event_cuts, inplace=True)

    data_joint['multiplicity'] = data_joint.groupby(['obs_id', 'event_id']).size()
    data_joint.query('multiplicity > 1', inplace=True)

    n_events_total = len(data_joint.groupby(['obs_id', 'event_id']).size())
    logger.info(f'\nIn total {n_events_total} stereo events are found:')

    for event_type, (tel_combo, tel_ids) in enumerate(tel_combinations.items()):

        df = data_joint.query(f'(tel_id == {tel_ids}) & (multiplicity == {len(tel_ids)})')
        df['multiplicity'] = df.groupby(['obs_id', 'event_id']).size()
        df.query(f'multiplicity == {len(tel_ids)}', inplace=True)

        n_events = len(df.groupby(['obs_id', 'event_id']).size())
        logger.info(f'{tel_combo}: {n_events:.0f} events ({n_events / n_events_total * 100:.1f}%)')

        data_joint.loc[df.index, 'event_type'] = event_type

    # Check the angular separation if the input file is "real" data:
    telescope_ids = np.unique(data_joint.index.get_level_values('tel_id'))

    if (not is_simulation) and (tel_id_lst in telescope_ids):

        logger.info('\nChecking the angular distance of LST-1 and MAGIC pointing directions...')

        df_lst = data_joint.query('tel_id == 1')

        obs_ids_joint = list(df_lst.index.get_level_values('obs_id'))
        event_ids_joint = list(df_lst.index.get_level_values('event_id'))

        multi_indices = pd.MultiIndex.from_arrays([obs_ids_joint, event_ids_joint], names=['obs_id', 'event_id'])

        df_magic = data_joint.query('tel_id == [2, 3]')
        df_magic.reset_index(level='tel_id', inplace=True)
        df_magic = df_magic.loc[multi_indices]

        # Calculate the mean pointing direction of the MAGIC telescopes:
        df_magic_pointing = calc_tel_mean_pointing(df_magic)

        theta = angular_separation(
            lon1=u.Quantity(df_lst['az_tel'].to_numpy(), u.rad),
            lat1=u.Quantity(df_lst['alt_tel'].to_numpy(), u.rad),
            lon2=u.Quantity(df_magic_pointing['az_tel_mean'].to_numpy(), u.rad),
            lat2=u.Quantity(df_magic_pointing['alt_tel_mean'].to_numpy(), u.rad),
        )

        n_events_sep = np.sum(theta > theta_lim)

        if n_events_sep > 0:
            logger.info(f'--> The pointing directions are separated more than {theta_lim.value} degree. ' \
                        'The data would be taken by different wobble offsets. Please check the input data. Exiting.\n')
            sys.exit()
        else:
            angle_max = np.max(theta.to(u.arcmin).value)
            logger.info(f'--> Maximum angular separation is {angle_max:.3f} arcmin. Continue.')

    # Start reconstructing the stereo parameters:
    logger.info('\nReconstructing the stereo parameters...')

    event = ArrayEventContainer()
    hillas_reconstructor = HillasReconstructor(subarray)

    df_mean_pointing = calc_tel_mean_pointing(data_joint)
    data_joint = data_joint.join(df_mean_pointing)

    group = data_joint.groupby(['obs_id', 'event_id']).size()

    observation_ids = group.index.get_level_values('obs_id')
    event_ids = group.index.get_level_values('event_id')

    # Loop over observation/event IDs.
    # Since the HillasReconstructor requires the ArrayEventContainer,
    # here we set the necessary information to it event-by-event:

    for i_ev, (obs_id, ev_id) in enumerate(zip(observation_ids, event_ids)):

        if i_ev % 100 == 0:
            logger.info(f'{i_ev} events')

        df_ev = data_joint.query(f'(obs_id == {obs_id}) & (event_id == {ev_id})')

        event.pointing.array_altitude = u.Quantity(df_ev['alt_tel_mean'].iloc[0], u.rad)
        event.pointing.array_azimuth = u.Quantity(df_ev['az_tel_mean'].iloc[0], u.rad)

        telescope_ids = df_ev.index.get_level_values('tel_id')

        for tel_id in telescope_ids:

            df_tel = df_ev.query(f'tel_id == {tel_id}')

            event.pointing.tel[tel_id].altitude = u.Quantity(df_tel['alt_tel'].iloc[0], u.rad)
            event.pointing.tel[tel_id].azimuth = u.Quantity(df_tel['az_tel'].iloc[0], u.rad)

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

            # Save the stereo parameters:
            data_joint.loc[(obs_id, ev_id, tel_id), 'h_max'] = stereo_params.h_max.to(u.m).value
            data_joint.loc[(obs_id, ev_id, tel_id), 'alt'] = stereo_params.alt.to(u.deg).value
            data_joint.loc[(obs_id, ev_id, tel_id), 'alt_uncert'] = stereo_params.alt_uncert.to(u.deg).value
            data_joint.loc[(obs_id, ev_id, tel_id), 'az'] = stereo_params.az.to(u.deg).value
            data_joint.loc[(obs_id, ev_id, tel_id), 'az_uncert'] = stereo_params.az_uncert.to(u.deg).value
            data_joint.loc[(obs_id, ev_id, tel_id), 'core_x'] = stereo_params.core_x.to(u.m).value
            data_joint.loc[(obs_id, ev_id, tel_id), 'core_y'] = stereo_params.core_y.to(u.m).value
            data_joint.loc[(obs_id, ev_id, tel_id), 'impact'] = impact.to(u.m).value

    n_events_processed = i_ev + 1
    logger.info(f'{n_events_processed} events processed.')

    # Prepare for saving the data to an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    base_name = Path(input_file).resolve().name
    regex_run = r'dl1_(\w+)_run(\d+)\.h5'
    regex_subrun = rf'dl1_(\w+)_run(\d+)\.(\d+)\.h5'

    if re.fullmatch(regex_run, base_name):
        parser = re.findall(regex_run, base_name)[0]
        output_file = f'{output_dir}/dl1_stereo_{parser[0]}_run{parser[1]}.h5'

    elif re.fullmatch(regex_subrun, base_name):
        parser = re.findall(regex_subrun, base_name)[0]
        output_file = f'{output_dir}/dl1_stereo_{parser[0]}_run{parser[1]}.{parser[2]}.h5'

    else:
        logger.warning('\nCould not parse run information from the input file name. Simply name the output file.')
        output_file = f'{output_dir}/dl1_stereo.h5'

    # Save the data in the file:
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

    stereo_reco(
        input_file=args.input_file,
        output_dir=args.output_dir,
        config=config,
    )

    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
