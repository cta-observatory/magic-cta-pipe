#!/usr/bin/env python
# coding: utf-8

"""
This script processes LIDAR report files obtained with MARS-based software quate to extract cloud-reveant parameters present during the observations with MAGIC telescopes (if any), which can then be used to correct event images affected by the clouds using lst1_magic_cloud_correction.py script. 

Usage:
$ python lst1_magic_lidar_report_to_yaml.py
--input-file "magic_lidar.*.results"
--output-file "magic_lidar_report_clouds_CrabNebula.yaml"
"""

import os
import re
import glob
import argparse
from datetime import datetime
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import SingleQuotedScalarString as SQS
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def extract_data_from_file(input_filename):
    """
    Extracts cloud-relevant data from a LIDAR report file (if any). 
    
    Parameters
    ----------
    input_filename : str
        The name of an input LIDAR  report file to process.

    Returns
    -------
    dict or None
        A dictionary with extracted data if valid cloud data is found, or None otherwise.
    """
    
    try:
        with open(input_filename, "r") as file:
            content = file.read()
            logger.info(f"\n----------------------------------------------------"
                        f"\nProcessing file: {input_filename}")

            if not re.search(r"ERROR_CODE:\s*4", content):
                logger.info("No valid LIDAR report found (ERROR_CODE != 4). \nSkipping file.")
                return None

            zenith_match = re.search(r"ZENITH_deg:\s*([\d.]+)", content)
            clouds_match = re.search(r"CLOUDS:\s*(\d+)", content)

            if not zenith_match or not clouds_match:
                logger.warning("Required data fields missing (ZENITH and/or CLOUDS). \nSkipping file.")
                return None

            zenith = float(zenith_match.group(1))
            num_clouds = int(clouds_match.group(1))

            if num_clouds == 0:
                logger.info("No cloud layers found in the report. \nSkipping file.")
                return None

            layers = extract_cloud_layers(content, num_clouds)
            timestamp = extract_timestamp_from_filename(input_filename)

            if timestamp is None:
                return None

            extracted_data = {
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "lidar_zenith": zenith,
                "layers": layers
            }
            
            return extracted_data
            
    except Exception as e:
        logger.error(f"Error processing file '{input_filename}': {e}")
        return None

def extract_cloud_layers(content, num_clouds):
    """
    Extracts cloud layer data from the input file containig LIDAR quate reports.

    Parameters
    ----------
    content : str
        The content of the LIDAR report file.
    num_clouds : int
        The number of cloud layers.

    Returns
    -------
    list
        A list of dictionaries containing cloud layer parameters.
    """
    
    layers = []
    for i in range(num_clouds):
        cloud_pattern = rf"CLOUD{i}\s*:\s*MEAN_HEIGHT:\s*([\d.]+)\s*\+\-\s*[\d.]+\s*BASE:\s*([\d.]+)\s*TOP:\s*([\d.]+)\s*FWHM:\s*[\d.]+\s*TRANS:\s*([\d.]+)"
        cloud_match = re.search(cloud_pattern, content)
        if cloud_match:
            mean_height, base, top, trans = cloud_match.groups()
            layers.append({
                "base_height": float(base),
                "top_height": float(top),
                "transmission": float(trans)
            })
        else:
            logger.warning(f"Cloud data not found.")
    return layers

def extract_timestamp_from_filename(input_filename):
    """
    Extracts the timestamp from the input filename.

    Parameters
    ----------
    input_filename : str
        The name of the input file.

    Returns
    -------
    datetime or None
        The extracted timestamp as a datetime object, or None if the format is incorrect.
    """

    # Extract only the base name for filename validation
    basename = os.path.basename(input_filename)
    pattern = r"^(magic_lidar\.)?(\d{8})\.(\d{6})\.results$"
    match = re.match(pattern, basename)

    if not match:
        logger.error(f"Unexpected filename format for timestamp in file: '{input_filename}'")
        logger.error("Expected format: 'magic_lidar.YYYYMMDD.HHMMSS.results'.")
        return None

    date_str = match.group(2)  # YYYYMMDD
    time_str = match.group(3)  # HHMMSS

    timestamp_str = f"{date_str}.{time_str}"
    return datetime.strptime(timestamp_str, "%Y%m%d.%H%M%S")
    

def write_to_yaml(data, output_filename):
    """
    Writes extracted cloud parameters to a YAML file, including units metadata.

    Parameters
    ----------
    data : list
        A list of dictionaries, each containing cloud data for a specific timestamp.
    output_filename : str
        The name of the output YAML file.

    Returns
    -------
    None
    """
    if not data:
        logger.info("No valid data found to write.")
        return

    output_data = {
        "units": {
            "timestamp": SQS("iso"),
            "lidar_zenith": SQS("deg"),
            "base_height": SQS("m"),
            "top_height": SQS("m"),
            "transmission": SQS("dimensionless"),
        },
        "data": data
    }

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.sort_keys = False

    try:
        with open(output_filename, 'w') as yaml_file:
            yaml.dump(output_data, yaml_file)
            logger.info(f"\nData written to output file '{output_filename}'.")
    except IOError as e:
        logger.error(f"Failed to write data to '{output_filename}': {e}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Input file name or full path to input file. Wildcards allowed."
    )

    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        required=True,
        help="Output YAML file name. One may provide full path where the output file should be created."
    )
    
    args = parser.parse_args()
    results = []

    files = glob.glob(args.input_file)
    if not files:
        logger.error(f"No file matching pattern '{args.input_file}'. Exiting.")
        return

    for filepath in files:
        basename = os.path.basename(filepath)
        if not re.match(r"^(magic_lidar\.)?(\d{8})\.(\d{6})\.results$", basename):
            logger.error(f"Invalid filename format: '{basename}'. Expected format:"
                         f"'magic_lidar.YYYYMMDD.HHMMSS.results'.")
            continue
            
        data = extract_data_from_file(filepath)
        if data:
            results.append(data)

    if results:
        results = sorted(results, key=lambda x: x["timestamp"])
        write_to_yaml(results, args.output_file)
        logger.info("All data processed and written successfully.")
    else:
        logger.info("No valid LIDAR data extracted from input files.")

if __name__ == "__main__":
    main()
