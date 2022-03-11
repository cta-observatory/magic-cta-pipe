#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script creates the IRFs.

Usage:
$ python lst1_magic_create_irf.py
--input-file ./data/dl2_gamma_40deg_90deg_off0.4deg_LST-1_MAGIC_run401_to_1000.h5
--output-dir ./data
--config-file ./config.yaml
"""

import time
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from astropy import units as u
from astropy.io import fits
from astropy.table import QTable
from pyirf.simulations import SimulatedEventsInfo
from pyirf.irf import (
    effective_area_per_energy,
    energy_dispersion,
    psf_table,
)
from pyirf.io.gadf import (
    create_aeff2d_hdu,
    create_energy_dispersion_hdu,
    create_psf_table_hdu,
)
from pyirf.utils import (
    calculate_theta,
    calculate_source_fov_offset,
)
from magicctapipe.utils import (
    get_dl2_mean,
    check_tel_combination,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

__all__ = [
    'create_irf',
]

def load_data(input_file, config_irf):
    """
    Load an input gamma MC data file.

    Parameters
    ----------
    input_file: str
        Path to an input gamma MC DL2 data file
    config_irf: dict
        Configuration for the IRF creation

    Returns
    -------
    data_qtable: astropy.table.QTable
        Astropy QTable of events surviving parameter cuts
    """

    data = pd.read_hdf(input_file, 'events/parameters')
    data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    data.sort_index(inplace=True)

    n_sim_runs = len(np.unique(data.index.get_level_values('obs_id')))

    check_tel_combination(data)

    if config_irf['quality_cuts'] is not None:

        logger.info('\nApplying the following quality cuts:')
        logger.info(config_irf['quality_cuts'])

        data.query(config_irf['quality_cuts'],  inplace=True)
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
    data_qtable['true_alt'] *= u.deg
    data_qtable['true_az'] *= u.deg
    data_qtable['reco_alt'] *= u.deg
    data_qtable['reco_az'] *= u.deg
    data_qtable['true_energy'] *= u.TeV
    data_qtable['reco_energy'] *= u.TeV

    # Compute angular distances:
    theta = calculate_theta(
        data_qtable, assumed_source_az=data_qtable['true_az'], assumed_source_alt=data_qtable['true_alt'],
    )

    true_source_fov_offset = calculate_source_fov_offset(data_qtable)
    reco_source_fov_offset = calculate_source_fov_offset(data_qtable, prefix='reco')

    data_qtable['theta'] = theta
    data_qtable['true_source_fov_offset'] = true_source_fov_offset
    data_qtable['reco_source_fov_offset'] = reco_source_fov_offset

    # Apply the gammaness/theta2 cuts:
    gammaness_cut = config_irf['gammaness_cut']
    theta_cut = u.Quantity(config_irf['theta_cut'], u.deg)

    data_qtable = data_qtable[data_qtable['gammaness'] > gammaness_cut]
    data_qtable = data_qtable[data_qtable['theta'] < theta_cut]

    # Load the simulation configuration:
    sim_config = pd.read_hdf(input_file, 'simulation/config')

    n_total_showers = sim_config['num_showers'].iloc[0] * sim_config['shower_reuse'].iloc[0] * n_sim_runs

    sim_evt_info = SimulatedEventsInfo(
        n_showers=n_total_showers,
        energy_min=u.Quantity(sim_config['energy_range_min'].iloc[0], u.TeV),
        energy_max=u.Quantity(sim_config['energy_range_max'].iloc[0], u.TeV),
        max_impact=u.Quantity(sim_config['max_scatter_range'].iloc[0], u.m),
        spectral_index=sim_config['spectral_index'].iloc[0],
        viewcone=u.Quantity(sim_config['max_viewcone_radius'].iloc[0], u.deg),
    )

    return data_qtable, sim_evt_info


def create_irf(input_file, output_dir, config):
    """
    Create the IRFs.

    Parameters
    ----------
    input_file: str
        Path to an input gamma MC DL2 data file
    output_dir: str
        Path to a directory where to save an output IRF file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    config_irf = config['create_irf_dl3']

    # Load the input file:
    data_qtable, sim_evt_info = load_data(input_file, config_irf)

    true_energy_bins = u.Quantity(np.logspace(np.log10(0.005), np.log10(50), 21), u.TeV)
    fov_offset_bins = u.Quantity([0.3, 0.5], u.deg)
    migration_bins = np.geomspace(0.2, 5, 31)
    source_offset_bins = u.Quantity(np.linspace(0.0001, 1.0001, 1000), u.deg)

    extra_headers = {
        'TELESCOP': 'CTA-N',
        'INSTRUME': 'LST-1_MAGIC',
        'FOVALIGN': 'RADEC',
    }

    extra_headers['GH_CUT'] = config_irf['gammaness_cut']
    extra_headers['RAD_MAX'] = (config_irf['theta_cut'], 'deg')

    hdus = [fits.PrimaryHDU(), ]

    # Create the effective area:
    logger.info('\nCreating the effective area...')

    aeff = effective_area_per_energy(
        selected_events=data_qtable,
        simulation_info=sim_evt_info,
        true_energy_bins=true_energy_bins,
    )

    hdu_aeff = create_aeff2d_hdu(
        effective_area=aeff[..., np.newaxis],
        true_energy_bins=true_energy_bins,
        fov_offset_bins=fov_offset_bins,
        point_like=True,
        extname='EFFECTIVE AREA',
        **extra_headers,
    )

    hdus.append(hdu_aeff)

    # Create the energy dispersion:
    logger.info('Creating the energy dispersion...')

    edisp = energy_dispersion(
        selected_events=data_qtable,
        true_energy_bins=true_energy_bins,
        fov_offset_bins=fov_offset_bins,
        migration_bins=migration_bins,
    )

    hdu_edisp = create_energy_dispersion_hdu(
        energy_dispersion=edisp,
        true_energy_bins=true_energy_bins,
        migration_bins=migration_bins,
        fov_offset_bins=fov_offset_bins,
        point_like=True,
        extname='ENERGY DISPERSION',
        **extra_headers,
    )

    hdus.append(hdu_edisp)

    # # Creating PSF:
    # logger.info('Creating PSF...')

    # psf = psf_table(
    #     events=data_qtable,
    #     true_energy_bins=true_energy_bins,
    #     source_offset_bins=source_offset_bins,
    #     fov_offset_bins=fov_offset_bins,
    # )

    # hdu_psf = create_psf_table_hdu(
    #     psf=psf,
    #     true_energy_bins=true_energy_bins,
    #     source_offset_bins=source_offset_bins,
    #     fov_offset_bins=fov_offset_bins,
    #     extname='PSF',
    # )

    # hdus.append(hdu_psf)

    # Save in an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    output_file = f'{output_dir}/irf_LST-1_MAGIC.fits.gz'
    fits.HDUList(hdus).writeto(output_file, overwrite=True)

    logger.info('\nOutput file:')
    logger.info(output_file)


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file', '-i', dest='input_file', type=str, required=True,
        help='Path to an input gamma MC DL2 data file.',
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='./data',
        help='Path to a directory where to save an output IRF file.',
    )

    parser.add_argument(
        '--config-file', '-c', dest='config_file', type=str, default='./config.yaml',
       help='Path to a yaml configuration file.',
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    # Create the IRFs:
    create_irf(args.input_file, args.output_dir, config)

    logger.info('\nDone.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()