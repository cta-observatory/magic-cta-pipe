#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script creates IRFs.

Usage:
$ python lst1_magic_create_irf.py
--input-file-gamma ./data/dl2_gamma_40deg_90deg_off0.4deg_LST-1_MAGIC_run401_to_1000.h5
--output-dir ./data/irf.fits.gz
--config-file ./config.yaml
"""

import time
import yaml
import logging
import argparse
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.table import QTable
from pyirf.simulations import SimulatedEventsInfo
from pyirf.irf import (
    effective_area,
    effective_area_per_energy,
    energy_dispersion,
    psf_table,
)
from pyirf.cuts import (
    calculate_percentile_cut,
    evaluate_binned_cut,
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

def create_irf(input_file_gamma, output_dir, config):
    """
    Create IRFs with gamma MC and background samples.

    Parameters
    ----------
    input_file_gamma: str
        Path to an input gamma MC DL2 data file
    output_dir: str
        Path to a directory where to save an output IRF file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    config_irf = config['create_irf']

    data_gamma = pd.read_hdf(input_file_gamma, 'events/parameters')
    data_gamma.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    data_gamma.sort_index(inplace=True)

    if config_irf['quality_cuts'] is not None:
        data_gamma.query(config_irf['quality_cuts'],  inplace=True)
        data_gamma['multiplicity'] = data_gamma.groupby(['obs_id', 'event_id']).size()
        data_gamma.query('multiplicity > 1', inplace=True)

        combo_types = check_tel_combination(data_gamma)
        data_gamma.update(combo_type)

    data_dl2 = get_dl2_mean(data_gamma)
    data_qtable = QTable.from_pandas(data_dl2)

    data_qtable['pointing_alt'] *= u.rad
    data_qtable['pointing_az'] *=  u.rad
    data_qtable['true_alt'] *= u.deg
    data_qtable['true_az'] *= u.deg
    data_qtable['reco_alt'] *= u.deg
    data_qtable['reco_az'] *= u.deg
    data_qtable['true_energy'] *= u.TeV
    data_qtable['reco_energy'] *= u.TeV

    theta = calculate_theta(data_qtable, data_qtable['true_az'], data_qtable['true_alt'])
    true_source_fov_offset = calculate_source_fov_offset(data_qtable)
    reco_source_fov_offset = calculate_source_fov_offset(data_qtable, prefix='reco')

    data_qtable['theta'] = theta
    data_qtable['true_source_fov_offset'] = true_source_fov_offset
    data_qtable['reco_source_fov_offset'] = reco_source_fov_offset

    data_qtable = data_qtable[data_qtable['gammaness'] > config_irf['gammaness_cut']]
    data_qtable = data_qtable[data_qtable['theta'] < u.Quantity(config_irf['theta_cut'], u.deg)]

    sim_config = pd.read_hdf(input_file_gamma, 'simulation/config')

    n_obs_ids = len(np.unique(data_gamma.index.get_level_values('obs_id')))
    n_total_showers = sim_config['num_showers'].iloc[0] * sim_config['shower_reuse'].iloc[0] * n_obs_ids

    sim_evt_info = SimulatedEventsInfo(
        n_showers=n_total_showers,
        energy_min=u.Quantity(sim_config['energy_range_min'].iloc[0], u.TeV),
        energy_max=u.Quantity(sim_config['energy_range_max'].iloc[0], u.TeV),
        max_impact=u.Quantity(sim_config['max_scatter_range'].iloc[0], u.m),
        spectral_index=sim_config['spectral_index'].iloc[0],
        viewcone=sim_config['max_viewcone_radius'].iloc[0] * u.deg,
    )

    true_energy_bins = u.Quantity(np.logspace(-1, 2, 16), u.TeV)
    fov_offset_bins = u.Quantity([0.3, 0.5], u.deg)
    migration_bins = np.geomspace(0.2, 5, 31)
    source_offset_bins = u.Quantity(np.linspace(0.0001, 1.0001, 1000), u.deg)

    hdus = [fits.PrimaryHDU(), ]

    logger.info('Creating Aeff...')
    aeff = effective_area_per_energy(data_qtable, sim_evt_info, true_energy_bins)

    hdu_aeff = create_aeff2d_hdu(
        aeff[..., np.newaxis], true_energy_bins,
        fov_offset_bins, point_like=True, extname='EFFECTIVE AREA',
    )

    logger.info('Creating Edisp...')
    edisp = energy_dispersion(
        data_qtable, true_energy_bins, fov_offset_bins, migration_bins,
    )

    hdu_edisp = create_energy_dispersion_hdu(
        edisp, true_energy_bins, migration_bins, fov_offset_bins,
        point_like=True, extname='ENERGY DISPERSION',
    )

    hdus.append(hdu_edisp)

    logger.info('Creating PSF...')
    psf = psf_table(
        data_qtable, true_energy_bins, fov_offset_bins, source_offset_bins,
    )

    hdu_psf = create_psf_table_hdu(
        psf, true_energy_bins, source_offset_bins, fov_offset_bins, extname='PSF',
    )

    hdus.append(hdu_psf)

    fits.HDUList(hdus).writeto(f'{output_dir}/irf.fits.gz', overwrite=True)


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file_gamma', '-g', dest='input_file_gamma', type=str, required=True,
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

    # Create IRFs:
    create_irf(args.input_file_gamma, args.output_dir, config)

    logger.info('\nDone.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()