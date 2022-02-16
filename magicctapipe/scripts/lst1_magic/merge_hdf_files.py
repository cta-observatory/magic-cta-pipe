#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script merges HDF files produced by the LST-1 + MAGIC combined analysis pipeline.
It parses information from file names, so they should follow the convention (*_run*.h5 or *_run*.*.h5).
If one gives "--run-wise" or "--subrun-wise" arguments, it merges input files run-wise or subrun-wise respectively.

Usage:
$ python merge_hdf_files.py
--input-dir ./data/dl1
--output-dir ./data/dl1/merged
"""

import re
import glob
import time
import tables
import logging
import argparse
import numpy as np
from pathlib import Path
from ctapipe.instrument import SubarrayDescription

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

__all__ = [
    'merge_hdf_files',
]


class MultiFileTypesError(Exception):
    pass


def write_to_table(input_file_mask, output_file):
    """
    This function creates a new table and writes input data.

    Parameters
    ----------
    input_file_mask: str
        Mask of the paths to input HDF files
    output_file: str
        Path to an output HDF file
    """

    input_files = glob.glob(input_file_mask)
    input_files.sort()

    with tables.open_file(output_file, mode='w') as merged_file:

        # Create a new table and write the data of the first input file:
        logger.info(input_files[0])

        with tables.open_file(input_files[0]) as input_data:

            event_params = input_data.root.events.params
            merged_file.create_table('/events', 'params', createparents=True, obj=event_params.read())

            for attribute in event_params.attrs._f_list():
                merged_file.root.events.params.attrs[attribute] = event_params.attrs[attribute]

            if 'simulation' in input_data.root:
                # Write the simulation configuration of the first input file,
                # assuming that it is consistent with the other input files:
                sim_config = input_data.root.simulation.config
                merged_file.create_table('/simulation', 'config', createparents=True, obj=sim_config.read())

                for attribute in sim_config.attrs._f_list():
                    merged_file.root.simulation.config.attrs[attribute] = sim_config.attrs[attribute]

        # Write the rest of the input files:
        for input_file in input_files[1:]:

            logger.info(input_file)

            with tables.open_file(input_file) as input_data:
                event_params = input_data.root.events.params
                merged_file.root.events.params.append(event_params.read())

    # Save the subarray description of the first input file,
    # assuming that it is consistent with the other input files:
    subarray = SubarrayDescription.from_hdf(input_files[0])
    subarray.to_hdf(output_file)

    logger.info(f'--> {output_file}\n')


def merge_hdf_files(
    input_dir,
    output_dir,
    run_wise=False,
    subrun_wise=False,
):
    """
    This function merges input HDF files.

    Parameters
    ----------
    input_dir: str
        Path to a directory where input HDF files are stored
    output_dir: str
        Path to a directory where to save merged HDF files
    run_wise: bool
        If True, input files are merged run-wise
    subrun_wise: bool
        If True, input files are merged subrun-wise
    """

    logger.info(f'\nInput directory:\n{input_dir}')
    input_file_mask = f'{input_dir}/*.h5'

    input_files = glob.glob(input_file_mask)
    input_files.sort()

    # Parse information from input file names:
    regex_run = r'(\S+)_run(\d+)\.h5'
    regex_subrun = r'(\S+)_run(\d+)\.(\d+)\.h5'

    file_names = np.array([])
    run_ids = np.array([])
    subrun_ids = np.array([])

    for file_path in input_files:

        base_name = Path(file_path).name

        if re.fullmatch(regex_run, base_name):
            parser = re.findall(regex_run, base_name)[0]
            file_names = np.append(file_names, parser[0])
            run_ids = np.append(run_ids, parser[1])

        elif re.fullmatch(regex_subrun, base_name):
            parser = re.findall(regex_subrun, base_name)[0]
            file_names = np.append(file_names, parser[0])
            run_ids = np.append(run_ids, parser[1])
            subrun_ids = np.append(subrun_ids, parser[2])

    run_ids_unique = np.unique(run_ids)
    logger.info(f'\nThe following run IDs are found:\n{run_ids_unique}')

    file_names_unique = np.unique(file_names)

    if len(file_names_unique) > 1:
        if file_names_unique.tolist() == ['dl1_m1', 'dl1_m2']:
            file_name = 'dl1_magic'
        else:
            raise MultiFileTypesError('Multiple types of files exist in the input directory.')

    else:
        file_name = file_names_unique[0]

    # Merge input files:
    logger.info('\nMerging the following files:')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if subrun_wise:
        for run_id in run_ids_unique:
            subrun_ids_unique = np.unique(subrun_ids[run_ids == run_id])

            for subrun_id in subrun_ids_unique:
                file_mask = f'{input_dir}/*_run{run_id}.{subrun_id}.h5'
                output_file = f'{output_dir}/{file_name}_run{run_id}.{subrun_id}.h5'

                write_to_table(file_mask, output_file)

    elif run_wise:
        for run_id in run_ids_unique:
            file_mask = f'{input_dir}/*_run{run_id}.*'
            output_file = f'{output_dir}/{file_name}_run{run_id}.h5'

            write_to_table(file_mask, output_file)

    else:
        file_mask = f'{input_dir}/*.h5'
        output_file = f'{output_dir}/{file_name}_run{run_ids_unique[0]}_to_run{run_ids_unique[-1]}.h5'

        write_to_table(file_mask, output_file)

    logger.info('\nDone.')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-dir', '-i', dest='input_dir', type=str, required=True,
        help='Path to a directory where input HDF files are stored.',
    )

    parser.add_argument(
        '--output-dir', '-o', dest='output_dir', type=str, default='./data',
        help='Path to a directory where to save merged HDF files.',
    )

    parser.add_argument(
        '--run-wise', dest='run_wise', action='store_true',
        help='Merge input files run-wise.',
    )

    parser.add_argument(
        '--subrun-wise', dest='subrun_wise', action='store_true',
        help='Merge input files subrun-wise.',
    )

    args = parser.parse_args()

    merge_hdf_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        run_wise=args.run_wise,
        subrun_wise=args.subrun_wise,
    )

    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
