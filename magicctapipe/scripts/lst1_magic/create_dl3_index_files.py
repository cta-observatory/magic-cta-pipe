#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script creates IRF index files. Output files will be saved in an input directory.

Usage:
$ python create_dl3_index_files.py
--input-dir ./data/dl3
"""

import glob
from pathlib import Path
from lstchain.high_level import (
    create_hdu_index_hdu,
    create_obs_index_hdu,
)

def create_dl3_index_files(input_dir):
    """
    Creates DL3 index files.

    Parameters
    ----------
    input_dir: str
        Path to a directory where input DL3 data files are stored
    """

    file_mask = f'{input_dir}/dl3_*.fits.gz'

    input_files = glob.glob(file_mask)
    input_files.sort()

    file_names = []

    for input_file in input_files:
        base_name = Path(input_file).name
        file_names.append(base_name)

    hdu_index_file = 'hdu-index.fits.gz'
    obs_index_file = 'obs-index.fits.gz'

    # Create a hdu index file:
    create_hdu_index_hdu(
        filename_list=file_names,
        fits_dir=input_dir,
        hdu_index_file=hdu_index_file,
        overwrite=True,
    )

    # Create an observation index file:
    create_obs_index_hdu(
        filename_list=file_names,
        fits_dir=input_dir,
        obs_index_file=obs_index_file,
        overwrite=True,
    )


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-dir', '-i', dest='input_dir', type=str, required=True,
        help='Path to a directory where input DL3 files are stored.',
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    # Create the index files:
    create_dl3_index_files(args.input_dir)

    logger.info('\nDone.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
