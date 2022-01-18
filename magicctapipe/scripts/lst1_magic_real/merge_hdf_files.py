#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp) 

Merge the HDF files produced by the LST-1 + MAGIC combined analysis pipeline.

Usage:
$ python merge_hdf_files.py 
--input-files "./data/dl1/dl1_*.h5"
--output-file "./data/dl1/dl1_magic_run05093711_to_05093714_merged.h5"
"""

import time
import glob
import tables
import logging
import argparse
from pathlib import Path

from ctapipe.instrument import SubarrayDescription

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

__all__ = ['merge_hdf_files']


def merge_hdf_files(input_files, output_file):

    file_paths = glob.glob(input_files)
    file_paths.sort()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with tables.open_file(output_file, mode='w') as merged_file:

        logger.info('\nMerging the following data files:')
        logger.info(file_paths[0])

        with tables.open_file(file_paths[0]) as input_data:
            
            event_params = input_data.root.events.params
            merged_file.create_table('/events', 'params', createparents=True, obj=event_params.read())

            for attribute in event_params.attrs._f_list():
                merged_file.root.events.params.attrs[attribute] = event_params.attrs[attribute]

            if 'simulation' in input_data.root:
                sim_config = input_data.root.simulation.config
                merged_file.create_table('/simulation', 'config', createparents=True, obj=sim_config.read())

                for attribute in sim_config.attrs._f_list():
                    merged_file.root.simulation.config.attrs[attribute] = sim_config.attrs[attribute]

        for path in file_paths[1:]:
            
            logger.info(path)

            with tables.open_file(path) as input_data:
                event_params = input_data.root.events.params
                merged_file.root.events.params.append(event_params.read())

    subarray = SubarrayDescription.from_hdf(file_paths[0])
    subarray.to_hdf(output_file)    

    logger.info(f'\nOutput data file: {output_file}')
    logger.info('\nDone.')


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-files', '-i', dest='input_files', type=str,
        help='Path to input HDF data files.'
    )

    parser.add_argument(
        '--output-file', '-o', dest='output_file', type=str, default='./merged_data.h5',
        help='Path to an output HDF data file.'
    )

    args = parser.parse_args()

    merge_hdf_files(args.input_files, args.output_file) 

    end_time = time.time()
    logger.info(f'\nProcess time: {end_time - start_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
