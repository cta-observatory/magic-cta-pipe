#!/usr/bin/env python
# coding: utf-8

"""
This script creates the IRFs with input MC DL2 data.
Now it can create only point-like IRFs (effective area, energy migration and background HDUs).
If input data is only gamma MC, it skips the creation of a background HDU.

Usage:
$ python lst1_magic_create_irf.py
--input-file-gamma ./data/dl2_gamma_40deg_90deg_off0.4deg_LST-1_MAGIC_run401_to_1000.h5
--output-dir ./data
--config-file ./config.yaml
(--input-file-proton)
(--input-file-electron)
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

__all__ = [
    'load_dl2_data_file',
    'apply_dynamic_gammaness_cut',
    'apply_dynamic_theta_cut',
    'create_irf',
]


def load_dl2_data_file(input_file, quality_cuts, irf_type, dl2_weight):
    """
    Loads an input MC DL2 data file.

    Parameters
    ----------
    input_file: str
        Path to an input MC DL2 data file
    quality_cuts: str
        Quality cuts applied to the input events
    irf_type: str
        Type of the IRFs which will be created
    dl2_weight: str
        Type of the weight for averaging tel-wise DL2 parameters

    Returns
    -------
    event_table: astropy.table.table.QTable
        Astropy table of MC DL2 events
    sim_info: pyirf.simulations.SimulatedEventsInfo
        Container of simulation information
    """

    df_events = pd.read_hdf(input_file, key='events/parameters')
    df_events.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    df_events.sort_index(inplace=True)

    # Apply the quality cuts:
    if quality_cuts is not None:
        logger.info('\nApplying the quality cuts:')
        logger.info(quality_cuts)

        df_events.query(quality_cuts, inplace=True)
        df_events['multiplicity'] = df_events.groupby(['obs_id', 'event_id']).size()
        df_events.query('multiplicity > 1', inplace=True)

    combo_types = check_tel_combination(df_events)
    df_events.update(combo_types)

    # Select the events of the specified IRF type:
    logger.info(f'\nExtracting the events of the "{irf_type}" type...')

    if irf_type == 'software':
        df_events.query('combo_type == 3', inplace=True)

    elif irf_type == 'software_with_any2':
        df_events.query('(combo_type > 0) & (magic_stereo == True)', inplace=True)

    elif irf_type == 'magic_stereo':
        df_events.query('combo_type == 0', inplace=True)

    elif irf_type != 'hardware':
        raise KeyError(f'Unknown IRF type "{irf_type}".')

    n_events = len(df_events.groupby(['obs_id', 'event_id']).size())
    logger.info(f'--> {n_events} stereo events')

    # Compute the mean of the DL2 parameters:
    logger.info(f'\nDL2 weight type: {dl2_weight}')

    df_dl2_mean = get_dl2_mean(df_events, dl2_weight)
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
    sim_config = pd.read_hdf(input_file, key='simulation/config')

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


def apply_dynamic_gammaness_cut(
    table_gamma,
    table_bkg,
    energy_bins,
    gamma_efficiency,
    min_cut,
    max_cut,
):
    """
    Applies dynamic (energy-dependent) gammaness cuts to input events.
    The cuts are computed in each energy bin so as to keep the specified gamma efficiency.

    Parameters
    ----------
    table_gamma: astropy.table.table.QTable
        Astropy table of gamma MC events
    table_bkg: astropy.table.table.QTable
        Astropy table of background MC events
    energy_bins: astropy.units.quantity.Quantity
        Energy bins where to compute and apply the cuts
    gamma_efficiency: float
        Efficiency of the gamma MC events surviving the cuts
    min_cut: float
        Minimum value of cut - the cuts smaller than this value are replaced with this value
    max_cut: float
        Maximum value of cut - the cuts larger than this value are replaced with this value

    Returns
    -------
    table_gamma: astropy.table.table.QTable
        Astropy table of the gamma MC events surviving the cuts
    table_bkg: astropy.table.table.QTable
        Astropy table of the background MC events surviving the cuts
    cut_table: astropy.table.table.QTable
        Astropy table of the gammaness cuts
    """

    # Compute the cuts satisfying the efficiency:
    percentile = 100 * (1 - gamma_efficiency)

    cut_table = calculate_percentile_cut(
        values=table_gamma['gammaness'],
        bin_values=table_gamma['reco_energy'],
        bins=energy_bins,
        fill_value=min_cut,
        percentile=percentile,
        min_value=min_cut,
        max_value=max_cut,
    )

    # Apply the cuts to the input events:
    mask_gamma = evaluate_binned_cut(
        values=table_gamma['gammaness'],
        bin_values=table_gamma['reco_energy'],
        cut_table=cut_table,
        op=operator.ge,
    )

    table_gamma = table_gamma[mask_gamma]

    if table_bkg is not None:

        mask_bkg = evaluate_binned_cut(
            values=table_bkg['gammaness'],
            bin_values=table_bkg['reco_energy'],
            cut_table=cut_table,
            op=operator.ge,
        )

        table_bkg = table_bkg[mask_bkg]

    return table_gamma, table_bkg, cut_table


def apply_dynamic_theta_cut(
    table_gamma,
    energy_bins,
    gamma_efficiency,
    min_cut,
    max_cut,
):
    """
    Applies dynamic (energy-dependent) theta cuts to input events.
    The cuts are computed in each energy bin so as to keep the specified gamma efficiency.

    Parameters
    ----------
    table_gamma: astropy.table.table.QTable
        Astropy table of gamma MC events
    energy_bins: astropy.units.quantity.Quantity
        Energy bins where to compute and apply the cuts
    gamma_efficiency: float
        Efficiency of the gamma MC events surviving the cuts
    min_cut: astropy.units.quantity.Quantity
        Minimum cut value - the cuts smaller than this value are replaced with it
    max_cut: astropy.units.quantity.Quantity
        Maximum cut value - the cuts larger than this value are replaced with it

    Returns
    -------
    table_gamma: astropy.table.table.QTable
        Astropy table of the gamma MC events surviving the cuts
    cut_table: astropy.table.table.QTable
        Astropy table of the theta cuts
    """

    # Compute the cuts satisfying the efficiency:
    percentile = 100 * gamma_efficiency

    cut_table = calculate_percentile_cut(
        values=table_gamma['theta'],
        bin_values=table_gamma['reco_energy'],
        bins=energy_bins,
        fill_value=max_cut,
        percentile=percentile,
        min_value=min_cut,
        max_value=max_cut,
    )

    # Apply the cuts to the input events:
    mask = evaluate_binned_cut(
        values=table_gamma['theta'],
        bin_values=table_gamma['reco_energy'],
        cut_table=cut_table,
        op=operator.le,
    )

    table_gamma = table_gamma[mask]

    return table_gamma, cut_table


def create_irf(
    input_file_gamma,
    input_file_proton,
    input_file_electron,
    output_dir,
    config,
):
    """
    Creates IRF HDUs with input gamma and background MC DL2 data.

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

    only_gamma_mc = (input_file_proton is None) and (input_file_electron is None)

    config_irf = config['create_irf']

    quality_cuts = config_irf['quality_cuts']
    irf_type = config_irf['irf_type']
    dl2_weight = config_irf['dl2_weight']

    extra_headers = {
        'TELESCOP': 'CTA-N',
        'INSTRUME': 'LST-1_MAGIC',
        'FOVALIGN': 'RADEC',
        'QUAL_CUT': quality_cuts,
        'IRF_TYPE': irf_type,
        'DL2_WEIG': dl2_weight,
    }

    logger.info('\nEnergy bins (log space, TeV):')
    logger.info(config_irf['energy_bins'])

    energy_bins = np.logspace(
        np.log10(config_irf['energy_bins']['start']),
        np.log10(config_irf['energy_bins']['stop']),
        config_irf['energy_bins']['n_bins'] + 1,
    ) * u.TeV

    logger.info('\nMigration bins (geom space):')
    logger.info(config_irf['migration_bins'])

    migration_bins = np.geomspace(
        config_irf['migration_bins']['start'],
        config_irf['migration_bins']['stop'],
        config_irf['migration_bins']['n_bins'] + 1,
    )

    if not only_gamma_mc:
        logger.info('\nBackground FoV offset bins (linear space, deg):')
        logger.info(config_irf['bkg_fov_offset_bins'])

        bkg_fov_offset_bins = np.linspace(
            config_irf['bkg_fov_offset_bins']['start'],
            config_irf['bkg_fov_offset_bins']['stop'],
            config_irf['bkg_fov_offset_bins']['n_bins'] + 1,
        ) * u.deg

    # Load the input gamma MC file:
    logger.info('\nLoading the input gamma MC DL2 data file:')
    logger.info(input_file_gamma)

    table_gamma, sim_info_gamma = load_dl2_data_file(input_file_gamma, quality_cuts, irf_type, dl2_weight)

    if sim_info_gamma.viewcone.value != 0.0:
        logger.info('\nHave not yet implemented functions to create diffuse IRFs. Exiting.')
        sys.exit()

    mean_fov_offset = np.round(table_gamma['true_source_fov_offset'].mean().to_value(), decimals=1)
    fov_offset_bins = u.Quantity([mean_fov_offset - 0.1, mean_fov_offset + 0.1], u.deg)

    logger.info(f'\nMean FoV offset: {mean_fov_offset * u.deg}')
    logger.info(f'--> FoV offset bins: {fov_offset_bins}')

    if not only_gamma_mc:
        # Load the input proton MC file:
        logger.info('\nLoading the input proton MC DL2 data file:')
        logger.info(input_file_proton)

        table_proton, sim_info_proton = load_dl2_data_file(input_file_proton, quality_cuts, irf_type, dl2_weight)
        simulated_spectrum_proton = PowerLaw.from_simulation(sim_info_proton, obs_time_irf)

        table_proton['weight'] = calculate_event_weights(
            true_energy=table_proton['true_energy'],
            target_spectrum=IRFDOC_PROTON_SPECTRUM,
            simulated_spectrum=simulated_spectrum_proton,
        )

        # Load the input electron MC file:
        logger.info('\nLoading the input electron MC DL2 data file:')
        logger.info(input_file_electron)

        table_electron, sim_info_electron = load_dl2_data_file(input_file_electron, quality_cuts, irf_type, dl2_weight)
        simulated_spectrum_electron = PowerLaw.from_simulation(sim_info_electron, obs_time_irf)

        table_electron['weight'] = calculate_event_weights(
            true_energy=table_electron['true_energy'],
            target_spectrum=IRFDOC_ELECTRON_SPECTRUM,
            simulated_spectrum=simulated_spectrum_electron,
        )

        # Combine the proton and electron tables:
        table_bkg = table.vstack([table_proton, table_electron])

    else:
        table_bkg = None

    # Apply the gammaness cut:
    gam_cut_type = config_irf['gammaness']['cut_type']

    if gam_cut_type == 'global':
        logger.info('\nApplying the global gammaness cut:')

        global_gam_cut = config_irf['gammaness']['global_cut_value']
        logger.info(f'Global cut value: {global_gam_cut}')

        table_gamma = table_gamma[table_gamma['gammaness'] > global_gam_cut]

        if not only_gamma_mc:
            table_bkg = table_bkg[table_bkg['gammaness'] > global_gam_cut]

        gam_cut_config = f'gam_glob{global_gam_cut}'
        extra_headers['GH_CUT'] = global_gam_cut

    elif gam_cut_type == 'dynamic':
        logger.info('\nApplying the dynamic gammaness cuts:')

        gamma_efficiency = config_irf['gammaness']['gamma_efficiency']
        min_cut = config_irf['gammaness']['min_cut']
        max_cut = config_irf['gammaness']['max_cut']

        table_gamma, table_bkg, cut_table_gh = apply_dynamic_gammaness_cut(table_gamma, table_bkg, energy_bins,
                                                                           gamma_efficiency, min_cut, max_cut)

        logger.info(f'Gamma efficiency: {gamma_efficiency}')
        logger.info(f'Minimum gammaness cut allowed: {min_cut}')
        logger.info(f'Maximum gammaness cut allowed: {max_cut}')
        logger.info(f'\n{cut_table_gh}')

        gam_cut_config = f'gam_dyn{gamma_efficiency}'
        extra_headers['GH_EFF'] = (gamma_efficiency, 'gamma efficiency')

    else:
        raise KeyError(f'Unknown type of the gammaness cut "{gam_cut_type}".')

    # Apply the theta cut:
    theta_cut_type = config_irf['theta']['cut_type']

    if theta_cut_type == 'global':
        logger.info('\nApplying the global theta cut:')

        global_theta_cut = u.Quantity(config_irf['theta']['global_cut_value'], u.deg)
        logger.info(f'Global cut value: {global_theta_cut}')

        table_gamma = table_gamma[table_gamma['theta'] < global_theta_cut]

        theta_cut_config = f'theta_glob{global_theta_cut.value}'
        extra_headers['RAD_MAX'] = (global_theta_cut.value, 'deg')

    elif theta_cut_type == 'dynamic':
        logger.info('\nApplying the dynamic theta cuts:')

        gamma_efficiency = config_irf['theta']['gamma_efficiency']
        min_cut = u.Quantity(config_irf['theta']['min_cut'], u.deg)
        max_cut = u.Quantity(config_irf['theta']['max_cut'], u.deg)

        table_gamma, cut_table_theta = apply_dynamic_theta_cut(table_gamma, energy_bins,
                                                               gamma_efficiency, min_cut, max_cut)

        logger.info(f'Gamma efficiency: {gamma_efficiency}')
        logger.info(f'Minimum theta cut allowed: {min_cut}')
        logger.info(f'Maximum theta cut allowed: {max_cut}')
        logger.info(f'\n{cut_table_theta}')

        extra_headers['TH_EFF'] = (gamma_efficiency, 'gamma efficiency')
        theta_cut_config = f'theta_dyn{gamma_efficiency}'

    else:
        raise KeyError(f'Unknown type of the theta cut "{theta_cut_type}".')

    # Create an effective area HDU:
    logger.info('\nCreating an effective area HDU...')

    hdus = fits.HDUList([fits.PrimaryHDU(), ])

    with np.errstate(invalid='ignore', divide='ignore'):

        aeff = effective_area_per_energy(
            selected_events=table_gamma,
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
        selected_events=table_gamma,
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
            events=table_bkg,
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

    # Save the data in an output file:
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    regex = r'dl2_gamma_(\S+)_run.*'
    file_name = Path(input_file_gamma).name

    if re.fullmatch(regex, file_name):
        parser = re.findall(regex, file_name)[0]
        output_file = f'{output_dir}/irf_{parser}_{irf_type}_{gam_cut_config}_{theta_cut_config}.fits.gz'

    hdus.writeto(output_file, overwrite=True)

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
