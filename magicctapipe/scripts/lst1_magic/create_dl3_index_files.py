#!/usr/bin/env python
# coding: utf-8

"""
This script creates IRF index files using the modules developed in lstchain.
Output files will be saved in the same directory of input DL3 files.

Usage:
$ python create_dl3_index_files.py
--input-dir ./dl3
"""

import argparse
import glob
import logging
import time
from pathlib import Path

from lstchain.high_level import create_hdu_index_hdu, create_obs_index_hdu

__all__ = ["create_dl3_index_files"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def create_dl3_index_files(input_dir):
    """
    Creates DL3 index files.

    Parameters
    ----------
    input_dir: str
        Path to a directory where input DL3 data files are stored
    """

    file_mask = f"{input_dir}/dl3_*.fits.gz"

    input_files = glob.glob(file_mask)
    input_files.sort()

    file_names = []

    logger.info("\nInput DL3 data files:")

    for input_file in input_files:
        logger.info(input_file)
        file_name = Path(input_file).name
        file_names.append(file_name)

    hdu_index_file = f"{input_dir}/hdu-index.fits.gz"
    obs_index_file = f"{input_dir}/obs-index.fits.gz"

    logger.info("\nCreating DL3 index files...")

    # Create a hdu index file:
    create_hdu_index_hdu(
        filename_list=file_names,
        fits_dir=Path(input_dir),
        hdu_index_file=Path(hdu_index_file),
        overwrite=True,
    )

    # Create an observation index file:
    create_obs_index_hdu(
        filename_list=file_names,
        fits_dir=Path(input_dir),
        obs_index_file=Path(obs_index_file),
        overwrite=True,
    )


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        "-i",
        dest="input_dir",
        type=str,
        required=True,
        help="Path to a directory where input DL3 files are stored.",
    )

    args = parser.parse_args()

    # Create the index files:
    create_dl3_index_files(args.input_dir)

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
