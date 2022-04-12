#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script merges HDF files produced by the LST-1 + MAGIC combined analysis pipeline.
It parses information from file names, so they should follow the convention (*Run*.h5 or *run*.h5).
It saves merged file(s) in the "merged" directory which will be created under an input directory.

If the "--run-wise" or "--subrun-wise" arguments are given, it merges input files run-wise or subrun-wise.
They are applicable only to real data since MC data are already produced run-wise.
The "subrun-wise" option is used to merge telescope-wise MAGIC subrun files (e.g., dl1_M1_Run05093711.001.h5 + dl1_M2_Run05093711.001.h5).

Usage:
$ python merge_hdf_files.py
--input-dir ./data/dl1
(--run-wise)
(--subrun-wise)
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
    'write_to_table',
    'merge_hdf_files',
]


def write_to_table(input_file_mask, output_file):
    """
    Creates a new table and write input data into it.

    Parameters
    ----------
    input_file_mask: str
        Mask of the paths to input HDF files
    output_file: str
        Path to an output HDF file
    """

    input_files = glob.glob(input_file_mask)
    input_files.sort()

    with tables.open_file(output_file, mode='w') as f_out:

        logger.info(input_files[0])

        # Create a new table with the first input file:
        with tables.open_file(input_files[0]) as f_input:

            event_data = f_input.root.events.parameters
            f_out.create_table('/events', 'parameters', createparents=True, obj=event_data.read())

            for attribute in event_data.attrs._f_list():
                f_out.root.events.parameters.attrs[attribute] = event_data.attrs[attribute]

            if 'simulation' in f_input.root:
                # Write the simulation configuration of the first input file,
                # assuming that it is consistent with the other input files:
                sim_config = f_input.root.simulation.config
                f_out.create_table('/simulation', 'config', createparents=True, obj=sim_config.read())

                for attribute in sim_config.attrs._f_list():
                    f_out.root.simulation.config.attrs[attribute] = sim_config.attrs[attribute]

        # Write the rest of the input files:
        for input_file in input_files[1:]:

            logger.info(input_file)

            with tables.open_file(input_file) as f_input:
                event_data = f_input.root.events.parameters
                f_out.root.events.parameters.append(event_data.read())

    # Save the subarray description of the first input file,
    # assuming that it is consistent with the other input files:
    subarray = SubarrayDescription.from_hdf(input_files[0])
    subarray.to_hdf(output_file)

    logger.info(f'--> {output_file}\n')


def merge_hdf_files(input_dir, run_wise=False, subrun_wise=False):
    """
    Merges input HDF files produced by
    the LST-1 + MAGIC combined analysis pipeline.

    Parameters
    ----------
    input_dir: str
        Path to a directory where input HDF files are stored
    run_wise: bool
        If True, it merges input files run-wise (applicable to real data)
    subrun_wise: bool
        If True, it merges input files subrun-wise (applicable to MAGIC real data)
    """

    input_file_mask = f'{input_dir}/*.h5'

    input_files = glob.glob(input_file_mask)
    input_files.sort()

    # Parse information from the input file names:
    regex_run = re.compile(r'(\S+run)(\d+)\.h5', re.IGNORECASE)
    regex_subrun = re.compile(r'(\S+run)(\d+)\.(\d+)\.h5', re.IGNORECASE)

    file_names = []
    run_ids = []
    subrun_ids = []

    for input_file in input_files:

        file_name = Path(input_file).name

        if re.fullmatch(regex_run, file_name):
            parser = re.findall(regex_run, file_name)[0]
            file_names.append(parser[0])
            run_ids.append(parser[1])

        elif re.fullmatch(regex_subrun, file_name):
            parser = re.findall(regex_subrun, file_name)[0]
            file_names.append(parser[0])
            run_ids.append(parser[1])
            subrun_ids.append(parser[2])

    file_names_unique = np.unique(file_names)
    run_ids_unique = np.unique(run_ids)

    subrun_ids = np.array(subrun_ids)

    if len(file_names_unique) == 1:
        file_name = file_names_unique[0]

    elif file_names_unique.tolist() == ['dl1_M1.Run', 'dl1_M2.Run']:
        file_name = 'dl1_MAGIC.Run'

    else:
        raise RuntimeError('Multiple types of files are found in the input directory.')

    # Merge the input files:
    output_dir = f'{input_dir}/merged'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if subrun_wise:
        logger.info('\nMerging the input files subrun-wise:')

        for run_id in run_ids_unique:
            subrun_ids_unique = np.unique(subrun_ids[run_ids == run_id])

            for subrun_id in subrun_ids_unique:
                file_mask = f'{input_dir}/*Run{run_id}.{subrun_id}.h5'
                output_file = f'{output_dir}/{file_name}{run_id}.{subrun_id}.h5'

                write_to_table(file_mask, output_file)

    elif run_wise:
        logger.info('\nMerging the input files run-wise:')

        for run_id in run_ids_unique:
            file_mask = f'{input_dir}/*Run{run_id}.*h5'
            output_file = f'{output_dir}/{file_name}{run_id}.h5'

            write_to_table(file_mask, output_file)

    else:
        logger.info('\nMerging the input files:')

        file_mask = f'{input_dir}/*.h5'

        if len(run_ids_unique) == 1:
            output_file = f'{output_dir}/{file_name}{run_ids_unique[0]}.h5'

        else:
            run_id_min = np.min(run_ids_unique.astype(int))
            run_id_max = np.max(run_ids_unique.astype(int))

            string_lengths_unique = np.unique([len(x) for x in run_ids_unique])

            if len(string_lengths_unique) == 1:
                # Handle the case that the run_ids are originally zero-padded:
                run_id_min = str(run_id_min).zfill(string_lengths_unique[0])
                run_id_max = str(run_id_max).zfill(string_lengths_unique[0])

            output_file = f'{output_dir}/{file_name}{run_id_min}_to_{run_id_max}.h5'

        write_to_table(file_mask, output_file)


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-dir', '-i', dest='input_dir', type=str, required=True,
        help='Path to a directory where input HDF files are stored.',
    )

    parser.add_argument(
        '--run-wise', dest='run_wise', action='store_true',
        help='Merge input files run-wise (applicable to real data).',
    )

    parser.add_argument(
        '--subrun-wise', dest='subrun_wise', action='store_true',
        help='Merge input files subrun-wise (applicable to MAGIC real data).',
    )

    args = parser.parse_args()

    # Merge the input files:
    merge_hdf_files(args.input_dir, args.run_wise, args.subrun_wise)

    logger.info('Done.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
