#!/usr/bin/env python
# coding: utf-8

"""
Author: Yoshiki Ohtani (ICRR, ohtani@icrr.u-tokyo.ac.jp)

This script creates IRF index files.

Usage:
$ python create_dl3_index_files.py

"""

import glob
from pathlib import Path
from lstchain.high_level import (
    create_hdu_index_hdu,
    create_obs_index_hdu,
)

def create_dl3_index_files(input_dir, overwrite):
    """
    Create DL3 index files.

    Parameters
    ----------
    input_dir: str

    overwrite: bool

    """

    file_paths = glob.glob(f'{input_dir}/dl3_*.fits.gz')
    file_paths.sort()

    file_names = []

    for path in file_paths:
        base_name = Path(path).name
        file_names.append(base_name)

    hdu_index_file = 'hdu-index.fits.gz'
    obs_index_file = 'obs-index.fits.gz'

    # Create a hdu index file:
    create_hdu_index_hdu(
        filename_list=file_names,
        fits_dir=input_dir,
        hdu_index_file=hdu_index_file,
        overwrite=overwrite,
    )

    # Create an observation index file:
    create_obs_index_hdu(
        filename_list=file_names,
        fits_dir=input_dir,
        obs_index_file=hdu_index_file,
        overwrite=overwrite,
    )


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input-dir', '-i', dest='input_dir', type=str, required=True,
        help='Path to a directory where input DL3 files are stored.',
    )

    parser.add_argument(
        '--overwrite', dest='overwrite', action='store_true',
        help='Overwrite output files if they already exist in an input directory.',
    )

    args = parser.parse_args()

    with open(args.config_file, 'rb') as f:
        config = yaml.safe_load(f)

    # Create the index files:
    create_dl3_index_files(args.input_dir, args.overwrite)

    logger.info('\nDone.')

    process_time = time.time() - start_time
    logger.info(f'\nProcess time: {process_time:.0f} [sec]\n')


if __name__ == '__main__':
    main()
