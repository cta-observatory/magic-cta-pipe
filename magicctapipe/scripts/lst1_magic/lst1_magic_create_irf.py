#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script creates the IRFs with input MC DL2 files. Now it can create only point-like IRFs.
If input data is only gamma MC, it skips the creation of a background HDU and creates only effective area and energy dispersion HDUs.
The created HDUs will be saved in an output fits file.

Usage:
$ python lst1_magic_create_irf.py
--input-file-gamma ./data/dl2_gamma_40deg_90deg_off0.4deg_LST-1_MAGIC_run401_to_1000.h5
--output-dir ./data
--config-file ./config.yaml
"""

import re
import sys
import time
import yaml
import logging
import argparse
import operator
import numpy as np
import pandas as pd
from pathlib import Path
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.table import table, QTable
from pyirf.simulations import SimulatedEventsInfo
from pyirf.cuts import (
    calculate_percentile_cut,
    evaluate_binned_cut,
)
from pyirf.irf import (
    effective_area_per_energy,
    energy_dispersion,
    background_2d,
)
from pyirf.utils import (
    calculate_theta,
    calculate_source_fov_offset,
)
from pyirf.io.gadf import (
    create_aeff2d_hdu,
    create_energy_dispersion_hdu,
    create_background_2d_hdu,
    create_rad_max_hdu,
)
from pyirf.spectral import (
    PowerLaw,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
    calculate_event_weights,
)
from magicctapipe import __version__
from magicctapipe.utils import (
    get_dl2_mean,
    check_tel_combination,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

obs_time_irf = u.Quantity(50, u.hour)

migration_bins = np.geomspace(0.2, 5, 31)
bkg_fov_offset_bins = u.Quantity(np.linspace(0, 10, 21), u.deg)

__all__ = [
    'create_irf',
]


def load_dl2_data_file(input_file, config_irf):
    """
    Loads an input MC DL2 data file and
    returns an event table and a simulation info container.

    Parameters
    ----------
    input_file: str
        Path to an input MC DL2 data file
    config_irf: dict
        Configuration for the IRF creation

    Returns
    -------
    event_table: astropy.table.table.QTable
        Astropy table of MC DL2 events
    sim_info: pyirf.simulations.SimulatedEventsInfo
        Container of simulation information
    """

    df_events = pd.read_hdf(input_file, 'events/parameters')
    df_events.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    df_events.sort_index(inplace=True)

    check_tel_combination(df_events)

    # Apply the quality cuts:
    quality_cuts = config_irf['quality_cuts']

    if quality_cuts is not None:

        logger.info('\nApplying the following quality cuts:')
        logger.info(quality_cuts)

        df_events.query(quality_cuts, inplace=True)
        df_events['multiplicity'] = df_events.groupby(['obs_id', 'event_id']).size()
        df_events.query('multiplicity > 1', inplace=True)

        combo_types = check_tel_combination(df_events)
        df_events.update(combo_types)

    # Select the events of the specified IRF type:
    irf_type = config_irf['irf_type']

    if irf_type == 'software':
        logger.info('\nExtracting only the events having 3-tels information...')
        df_events.query('combo_type == 3', inplace=True)

        n_events = len(df_events.groupby(['obs_id', 'event_id']).size())
        logger.info(f'--> {n_events} stereo events')

    elif irf_type == 'software_with_any2':
        logger.info('\nExtracting only the events triggering two MAGICs...')
        df_events.query('magic_stereo == True', inplace=True)

        check_tel_combination(df_events)

    elif irf_type != 'hardware':
        raise KeyError(f'Unknown IRF type "{irf_type}". ' \
                       'Select "hardware", "software" or "software_with_any2".')

    # Compute the mean of the DL2 parameters:
    df_dl2_mean = get_dl2_mean(df_events)
    df_dl2_mean.reset_index(inplace=True)

    # Convert the pandas data frame to the astropy QTable:
    event_table = QTable.from_pandas(df_dl2_mean)

    event_table['pointing_alt'] *= u.rad
    event_table['pointing_az'] *= u.rad
    event_table['true_alt'] *= u.deg
    event_table['true_az'] *= u.deg
    event_table['reco_alt'] *= u.deg
    event_table['reco_az'] *= u.deg
    event_table['true_energy'] *= u.TeV
    event_table['reco_energy'] *= u.TeV

    event_table['theta'] = calculate_theta(
        events=event_table,
        assumed_source_az=event_table['true_az'],
        assumed_source_alt=event_table['true_alt'],
    )

    event_table['true_source_fov_offset'] = calculate_source_fov_offset(event_table)
    event_table['reco_source_fov_offset'] = calculate_source_fov_offset(event_table, prefix='reco')

    # Load the simulation configuration:
    sim_config = pd.read_hdf(input_file, 'simulation/config')

    n_sim_runs = len(np.unique(event_table['obs_id']))
    n_total_showers = sim_config['num_showers'][0] * sim_config['shower_reuse'][0] * n_sim_runs

    sim_info = SimulatedEventsInfo(
        n_showers=n_total_showers,
        energy_min=u.Quantity(sim_config['energy_range_min'][0], u.TeV),
        energy_max=u.Quantity(sim_config['energy_range_max'][0], u.TeV),
        max_impact=u.Quantity(sim_config['max_scatter_range'][0], u.m),
        spectral_index=sim_config['spectral_index'][0],
        viewcone=u.Quantity(sim_config['max_viewcone_radius'][0], u.deg),
    )

    return event_table, sim_info


def apply_dynamic_gammaness_cuts(
    event_table_gamma,
    event_table_bkg,
    energy_bins,
    gamma_efficiency,
):
    """
    Applies dynamic (energy-dependent) gammaness cuts
    to input events. The cuts are computed in each energy bin
    so as to keep the specified gamma efficiency.

    Parameters
    ----------
    event_table_gamma: astropy.table.table.QTable
        Astropy table of gamma MC DL2 events
    event_table_bkg: astropy.table.table.QTable
        Astropy table of background MC DL2 events
    energy_bins: astropy.units.quantity.Quantity
        Energy bins where to apply the cuts
    gamma_efficiency: float
        Efficiency of the gamma events surviving the cuts

    Returns
    -------
    event_table_gamma: astropy.table.table.QTable
        Astropy table of the gamma MC DL2 events surviving the cuts
    event_table_bkg: astropy.table.table.QTable
        Astropy table of the background MC DL2 events surviving the cuts
    cut_table: astropy.table.table.QTable
        Astropy table of the gammaness cuts
    """

    # Compute the cuts satisfying the efficiency:
    percentile = 100 * (1 - gamma_efficiency)

    cut_table = calculate_percentile_cut(
        values=event_table_gamma['gammaness'],
        bin_values=event_table_gamma['reco_energy'],
        bins=energy_bins,
        fill_value=1.0,
        percentile=percentile,
    )

    # Apply the cuts to the gamma and background events:
    mask_gh_gamma = evaluate_binned_cut(
        values=event_table_gamma['gammaness'],
        bin_values=event_table_gamma['reco_energy'],
        cut_table=cut_table,
        op=operator.ge,
    )

    event_table_gamma = event_table_gamma[mask_gh_gamma]

    if event_table_bkg is not None:

        mask_gh_bkg = evaluate_binned_cut(
            values=event_table_bkg['gammaness'],
            bin_values=event_table_bkg['reco_energy'],
            cut_table=cut_table,
            op=operator.ge,
        )

        event_table_bkg = event_table_bkg[mask_gh_bkg]

    return event_table_gamma, event_table_bkg, cut_table


def apply_dynamic_theta_cuts(
    event_table,
    energy_bins,
    gamma_efficiency,
):
    """
    Applies dynamic (energy-dependent) theta cuts
    to input events. The cuts are computed in each energy bin
    so as to keep the specified gamma efficiency.

    Parameters
    ----------
    event_table: astropy.table.table.QTable
        Astropy table of DL2 events
    energy_bins: astropy.units.quantity.Quantity
        Energy bins where to apply the cuts
    gamma_efficiency: float
        Efficiency of the events surviving the cuts

    Returns
    -------
    event_table: astropy.table.table.QTable
        Astropy table of DL2 events surviving the cuts
    cut_table: astropy.table.table.QTable
        Astropy table of the theta cuts
    """

    # Compute the cuts satisfying the efficiency:
    percentile = 100 * gamma_efficiency

    cut_table = calculate_percentile_cut(
        values=event_table['theta'],
        bin_values=event_table['reco_energy'],
        bins=energy_bins,
        fill_value=u.Quantity(0.0, u.deg),
        percentile=percentile,
    )

    # Apply the cuts to the input table:
    mask_theta = evaluate_binned_cut(
        values=event_table['theta'],
        bin_values=event_table['reco_energy'],
        cut_table=cut_table,
        op=operator.ge,
    )

    event_table = event_table[mask_theta]

    return event_table, cut_table


def create_irf(
    input_file_gamma,
    input_file_proton,
    input_file_electron,
    output_dir,
    config,
):
    """
    Creates the IRF HDUs and save them in an output fits file.

    Parameters
    ----------
    input_file_gamma: str
        Path to an input gamma MC DL2 data file
    input_file_proton: str
        Path to an input proton MC DL2 data file
    input_file_electron: str
        Path to an input electron MC DL2 data file
    output_dir: str
        Path to a directory where to save an output IRF file
    config: dict
        Configuration for the LST-1 + MAGIC analysis
    """

    config_irf = config['create_irf']

    logger.info('\nConfiguration for the IRF creation:')
    for key, value in config_irf.items():
        logger.info(f'{key}: {value}')

    energy_bins = np.logspace(
        np.log10(config_irf['energy_bins']['start']),
        np.log10(config_irf['energy_bins']['stop']),
        config_irf['energy_bins']['n_bins'] + 1,
    )

    energy_bins = u.Quantity(energy_bins, u.TeV)

    hdus = fits.HDUList([fits.PrimaryHDU(), ])

    extra_headers = {
        'TELESCOP': 'CTA-N',
        'INSTRUME': 'LST-1_MAGIC',
        'FOVALIGN': 'RADEC',
        'IRF_TYPE': config_irf['irf_type'],
        'QUAL_CUT': config_irf['quality_cuts'],
    }

    # Load the input gamma MC file:
    logger.info('\nLoading the gamma MC DL2 data file:')
    logger.info(input_file_gamma)

    event_table_gamma, sim_info_gamma = load_dl2_data_file(input_file_gamma, config_irf)

    if sim_info_gamma.viewcone.value != 0.0:
        logger.info('\nHave not yet implemented functions to create diffuse IRFs. Exiting.')
        sys.exit()

    only_gamma_mc = (input_file_proton is None) and (input_file_electron is None)

    if only_gamma_mc:
        event_table_bkg = None

    else:
        # Load the input proton MC file:
        logger.info('\nLoading the proton MC DL2 data file:')
        logger.info(input_file_proton)

        event_table_proton, sim_info_proton = load_dl2_data_file(input_file_proton, config_irf)
        simulated_spectrum_proton = PowerLaw.from_simulation(sim_info_proton, obs_time_irf)

        event_table_proton['weight'] = calculate_event_weights(
            true_energy=event_table_proton['true_energy'],
            target_spectrum=IRFDOC_PROTON_SPECTRUM,
            simulated_spectrum=simulated_spectrum_proton,
        )

        # Load the input electron MC file:
        logger.info('\nLoading the electron MC DL2 data file:')
        logger.info(input_file_electron)

        event_table_electron, sim_info_electron = load_dl2_data_file(input_file_electron, config_irf)
        simulated_spectrum_electron = PowerLaw.from_simulation(sim_info_electron, obs_time_irf)

        event_table_electron['weight'] = calculate_event_weights(
            true_energy=event_table_electron['true_energy'],
            target_spectrum=IRFDOC_ELECTRON_SPECTRUM,
            simulated_spectrum=simulated_spectrum_electron,
        )

        # Combine the proton and electron tables:
        event_table_bkg = table.vstack([event_table_proton, event_table_electron])

    # Apply the gammaness cuts:
    gam_cut_type = config_irf['gammaness']['cut_type']

    if gam_cut_type == 'global':

        logger.info('\nApplying the global gammaness cut...')
        global_gam_cut = config_irf['gammaness']['global_cut_value']

        event_table_gamma = event_table_gamma[event_table_gamma['gammaness'] > global_gam_cut]

        if not only_gamma_mc:
            event_table_bkg = event_table_bkg[event_table_bkg['gammaness'] > global_gam_cut]

        extra_headers['GH_CUT'] = global_gam_cut
        gam_cut_config = f'gam_global{global_gam_cut}'

    elif gam_cut_type == 'dynamic':

        logger.info('\nApplying the dynamic gammaness cuts...')
        gamma_efficiency = config_irf['gammaness']['gamma_efficiency']

        event_table_gamma, event_table_bkg, cut_table_gh = apply_dynamic_gammaness_cuts(event_table_gamma, event_table_bkg, energy_bins, gamma_efficiency)

        extra_headers['GH_EFF'] = (gamma_efficiency, 'gamma efficiency')
        gam_cut_config = f'gam_dynamic{gamma_efficiency}'

    else:
        raise KeyError(f'Unknown type of the gammaness cut "{gam_cut_type}". ' \
                       'Select "global" or "dynamic".')

    # Apply the theta cuts:
    theta_cut_type = config_irf['theta']['cut_type']

    if theta_cut_type == 'global':

        logger.info('Applying the global theta cut...')
        global_theta_cut = u.Quantity(config_irf['theta']['global_cut_value'], u.deg)

        event_table_gamma = event_table_gamma[event_table_gamma['theta'] < global_theta_cut]

        extra_headers['RAD_MAX'] = (global_theta_cut.value, 'deg')
        theta_cut_config = f'theta_global{global_theta_cut.value}'

    elif theta_cut_type == 'dynamic':

        logger.info('Applying the dynamic theta cuts...')
        gamma_efficiency = config_irf['theta']['gamma_efficiency']

        event_table_gamma, cut_table_theta = apply_dynamic_theta_cuts(event_table_gamma, energy_bins, gamma_efficiency)

        extra_headers['TH_EFF'] = (gamma_efficiency, 'gamma efficiency')
        theta_cut_config = f'theta_dynamic{gamma_efficiency}'

    else:
        raise KeyError(f'Unknown type of the theta cut "{theta_cut_type}". ' \
                       'Select "global" or "dynamic".')

    # Create an effective area HDU:
    logger.info('\nCreating an effective area HDU...')

    mean_fov_offset = np.round(event_table_gamma['true_source_fov_offset'].mean().to_value(), 1)
    fov_offset_bins = u.Quantity([mean_fov_offset - 0.1, mean_fov_offset + 0.1], u.deg)

    with np.errstate(invalid='ignore', divide='ignore'):

        aeff = effective_area_per_energy(
            selected_events=event_table_gamma,
            simulation_info=sim_info_gamma,
            true_energy_bins=energy_bins,
        )

        hdu_aeff = create_aeff2d_hdu(
            effective_area=aeff[:, np.newaxis],
            true_energy_bins=energy_bins,
            fov_offset_bins=fov_offset_bins,
            point_like=True,
            extname='EFFECTIVE AREA',
            **extra_headers,
        )

    hdus.append(hdu_aeff)

    # Create an energy dispersion HDU:
    logger.info('Creating an energy dispersion HDU...')

    edisp = energy_dispersion(
        selected_events=event_table_gamma,
        true_energy_bins=energy_bins,
        fov_offset_bins=fov_offset_bins,
        migration_bins=migration_bins,
    )

    hdu_edisp = create_energy_dispersion_hdu(
        energy_dispersion=edisp,
        true_energy_bins=energy_bins,
        migration_bins=migration_bins,
        fov_offset_bins=fov_offset_bins,
        point_like=True,
        extname='ENERGY DISPERSION',
        **extra_headers,
    )

    hdus.append(hdu_edisp)

    # Create a background HDU:
    if not only_gamma_mc:

        logger.info('Creating a background HDU...')

        bkg2d = background_2d(
            events=event_table_bkg,
            reco_energy_bins=energy_bins,
            fov_offset_bins=bkg_fov_offset_bins,
            t_obs=obs_time_irf,
        )

        hdu_bkg2d = create_background_2d_hdu(
            background_2d=bkg2d.T,
            reco_energy_bins=energy_bins,
            fov_offset_bins=bkg_fov_offset_bins,
            extname='BACKGROUND',
            **extra_headers,
        )

        hdus.append(hdu_bkg2d)

    # Create a gammaness-cut HDU:
    if gam_cut_type == 'dynamic':

        logger.info('Creating a gammaness-cut HDU...')

        gh_header = fits.Header()
        gh_header['CREATOR'] = f'magicctapipe v{__version__}'
        gh_header['DATE'] = Time.now().utc.iso

        for k, v in extra_headers.items():
            gh_header[k] = v

        hdu_gh = fits.BinTableHDU(cut_table_gh, header=gh_header, name='GH_CUTS')
        hdus.append(hdu_gh)

    # Create a theta-cut HDU:
    if theta_cut_type == 'dynamic':

        logger.info('Creating a rad-max HDU...')

        hdu_rad_max = create_rad_max_hdu(
            rad_max=cut_table_theta['cut'][:, np.newaxis],
            reco_energy_bins=energy_bins,
            fov_offset_bins=fov_offset_bins,
            point_like=True,
            extname='RAD_MAX',
            **extra_headers,
        )

        hdus.append(hdu_rad_max)

    # Save in an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    base_name = Path(input_file_gamma).name
    regex = r'dl2_gamma_(\S+)_run.*'

    parser = re.findall(regex, base_name)[0]

    irf_type = config_irf['irf_type']
    output_file = f'{output_dir}/irf_{parser}_{irf_type}_{gam_cut_config}_{theta_cut_config}.fits.gz'

    fits.HDUList(hdus).writeto(output_file, overwrite=True)

    logger.info('\nOutput file:')
    logger.info(output_file)


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-file-gamma', '-g', dest='input_file_gamma', type=str, required=True,
        help='Path to an input gamma MC DL2 data file.',
    )

    parser.add_argument(
        '--input-file-proton', '-p', dest='input_file_proton', type=str, default=None,
        help='Path to an input proton MC DL2 data file.',
    )

    parser.add_argument(
        '--input-file-electron', '-e', dest='input_file_electron', type=str, default=None,
        help='Path to an input electron MC DL2 data file.',
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
    create_irf(
        args.input_file_gamma,
        args.input_file_proton,
        args.input_file_electron,
        args.output_dir,
        config,
    )

    logger.info('\nDone.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()