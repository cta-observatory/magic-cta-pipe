#!/usr/bin/env python
# coding: utf-8

"""
This script merges the HDF files produced by the LST-1 + MAGIC combined
analysis pipeline. It parses information from the file names, so they
should follow the convention, i.e., *Run*.*.h5 or *run*.h5.

If no output directory is specified with the `--output-dir` argument,
it saves merged files in the `merged` directory which will be created
under the input directory.

If the `--run-wise` argument is given, it merges input files run-wise.
It is applicable only to real data since MC data are already produced
run-wise. The `--subrun-wise` argument can be also used to merge MAGIC
DL1 real data subrun-wise (for example, dl1_M1.Run05093711.001.h5
+ dl1_M2.Run05093711.001.h5 -> dl1_MAGIC.Run05093711.001.h5).

Usage:
$ python merge_hdf_files.py
--input-dir dl1
(--output-dir dl1_merged)
(--run-wise)
(--subrun-wise)
"""

import argparse
import glob
import logging
import re
import time
from pathlib import Path

import numpy as np
import tables
from ctapipe.instrument import SubarrayDescription

__all__ = ["write_data_to_table", "merge_hdf_files"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def write_data_to_table(input_file_mask, output_file):
    """
    Writes data to a new table.

    Parameters
    ----------
    input_file_mask: str
        Mask of the paths to input HDF files
    output_file: str
        Path to an output HDF file
    """

    # Find the input files
    input_files = glob.glob(input_file_mask)
    input_files.sort()

    with tables.open_file(output_file, mode="w") as f_out:

        logger.info(f"\n{input_files[0]}")

        # Create a new table with the first input file
        with tables.open_file(input_files[0]) as f_input:

            event_data = f_input.root.events.parameters

            f_out.create_table(
                "/events", "parameters", createparents=True, obj=event_data.read()
            )

            for attr in event_data.attrs._f_list():
                f_out.root.events.parameters.attrs[attr] = event_data.attrs[attr]

            if "simulation" in f_input.root:
                # Write the simulation configuration of the first input
                # file, assuming that it is consistent with the others
                sim_config = f_input.root.simulation.config

                f_out.create_table(
                    "/simulation", "config", createparents=True, obj=sim_config.read()
                )

                for attr in sim_config.attrs._f_list():
                    f_out.root.simulation.config.attrs[attr] = sim_config.attrs[attr]

        # Write the rest of the input files
        for input_file in input_files[1:]:

            logger.info(input_file)

            with tables.open_file(input_file) as f_input:
                event_data = f_input.root.events.parameters
                f_out.root.events.parameters.append(event_data.read())

    # Save the subarray description of the first input file, assuming
    # that it is consistent with the others
    subarray = SubarrayDescription.from_hdf(input_files[0])
    subarray.to_hdf(output_file)

    logger.info(f"--> Output file: {output_file}")


def merge_hdf_files(input_dir, output_dir=None, run_wise=False, subrun_wise=False):
    """
    Merges the HDF files produced by the combined analysis pipeline.

    Parameters
    ----------
    input_dir: str
        Path to a directory where input HDF files are stored
    output_dir: str
        Path to a directory where to save output HDF files
    run_wise: bool
        If `True`, it merges the input files run-wise
        (applicable only to real data)
    subrun_wise: bool
        If `True`, it merges the input files subrun-wise
        (applicable only to MAGIC real data)

    Raises
    ------
    FileNotFoundError
        If any HDF files are not found in the input directory
    RuntimeError
        If multiple types of files are found in the input directory
    """

    # Find the input files
    logger.info(f"\nInput directory: {input_dir}")

    input_file_mask = f"{input_dir}/*.h5"

    input_files = glob.glob(input_file_mask)
    input_files.sort()

    if len(input_files) == 0:
        raise FileNotFoundError("Could not find any HDF files in the input directory.")

    # Parse information from the input file names
    regex_run = re.compile(r"(\S+run)(\d+)\.h5", re.IGNORECASE)
    regex_subrun = re.compile(r"(\S+run)(\d+)\.(\d+)\.h5", re.IGNORECASE)
    regex_run_merged = re.compile(r"(\S+run)(\d+)_to_(\d+)\.h5", re.IGNORECASE)

    file_names = []
    run_ids = []
    subrun_ids = []

    for input_file in input_files:

        input_file_name = Path(input_file).name

        if re.fullmatch(regex_run, input_file_name):
            parser = re.findall(regex_run, input_file_name)[0]
            file_names.append(parser[0])
            run_ids.append(parser[1])

        elif re.fullmatch(regex_subrun, input_file_name):
            parser = re.findall(regex_subrun, input_file_name)[0]
            file_names.append(parser[0])
            run_ids.append(parser[1])
            subrun_ids.append(parser[2])

        elif re.fullmatch(regex_run_merged, input_file_name):
            parser = re.findall(regex_run_merged, input_file_name)[0]
            file_names.append(parser[0])
            run_ids.append(parser[1])
            run_ids.append(parser[2])

    file_names_unique = np.unique(file_names)

    if len(file_names_unique) == 1:
        output_file_name = file_names_unique[0]

    elif file_names_unique.tolist() == ["dl1_M1.Run", "dl1_M2.Run"]:
        # Assume that the input files are telescope-wise MAGIC DL1 data
        output_file_name = "dl1_MAGIC.Run"

    else:
        raise RuntimeError("Multiple types of files are found in the input directory.")

    # Create an output directory
    if output_dir is None:
        output_dir = f"{input_dir}/merged"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Merge the input files
    run_ids_unique = np.unique(run_ids)

    if subrun_wise:
        logger.info("\nMerging the input files subrun-wise...")

        run_ids = np.array(run_ids)
        subrun_ids = np.array(subrun_ids)

        for run_id in run_ids_unique:

            subrun_ids_unique, counts = np.unique(
                subrun_ids[run_ids == run_id], return_counts=True
            )

            subrun_ids_unique = subrun_ids_unique[counts > 1]

            for subrun_id in subrun_ids_unique:
                file_mask = f"{input_dir}/*Run{run_id}.{subrun_id}.h5"
                output_file = f"{output_dir}/{output_file_name}{run_id}.{subrun_id}.h5"

                write_data_to_table(file_mask, output_file)

    elif run_wise:
        logger.info("\nMerging the input files run-wise...")

        for run_id in run_ids_unique:
            file_mask = f"{input_dir}/*Run{run_id}.*h5"
            output_file = f"{output_dir}/{output_file_name}{run_id}.h5"

            write_data_to_table(file_mask, output_file)

    else:
        logger.info("\nMerging the input files...")

        file_mask = f"{input_dir}/*.h5"

        if len(run_ids_unique) == 1:
            output_file = f"{output_dir}/{output_file_name}{run_ids_unique[0]}.h5"

        else:
            string_lengths_unique = np.unique([len(x) for x in run_ids_unique])

            # Check the minimum and maximum run IDs with the int type
            run_ids_unique = run_ids_unique.astype(int)

            run_id_min = run_ids_unique.min()
            run_id_max = run_ids_unique.max()

            if len(string_lengths_unique) == 1:
                # Handle the case when the run IDs are zero-padded
                run_id_min = str(run_id_min).zfill(string_lengths_unique[0])
                run_id_max = str(run_id_max).zfill(string_lengths_unique[0])

            output_file = (
                f"{output_dir}/{output_file_name}{run_id_min}_to_{run_id_max}.h5"
            )

        write_data_to_table(file_mask, output_file)


def main():

    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        "-i",
        dest="input_dir",
        type=str,
        required=True,
        help="Path to a directory where input HDF files are stored",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        help="Path to a directory where to save output HDF files",
    )

    parser.add_argument(
        "--run-wise",
        dest="run_wise",
        action="store_true",
        help="Merge input files run-wise (applicable only to real data)",
    )

    parser.add_argument(
        "--subrun-wise",
        dest="subrun_wise",
        action="store_true",
        help="Merge input files subrun-wise (applicable only to MAGIC real data)",
    )

    args = parser.parse_args()

    # Merge the input files
    merge_hdf_files(args.input_dir, args.output_dir, args.run_wise, args.subrun_wise)

    logger.info("\nDone.")

    process_time = time.time() - start_time
    logger.info(f"\nProcess time: {process_time:.0f} [sec]\n")


if __name__ == "__main__":
    main()
