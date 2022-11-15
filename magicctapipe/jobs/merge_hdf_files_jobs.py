import os
import argparse

import subprocess as sp

from jinja2 import Template
from pathlib import Path


__all__ = ["merge_hdf_files_jobs"]


template = """#!/bin/bash
#SBATCH -A aswg
#SBATCH -p short,long,xxl
#SBATCH -J {{ job_name }}_%A
#SBATCH -o {{ output_dir }}/{{ job_name }}_%A_%a.out
#SBATCH -e {{ output_dir }}/{{ job_name }}_%A_%a.err

source {{ home }}/.bashrc
conda activate magic-lst1

merge_hdf_files -i {{ input_dir }} -o {{ output_dir }} {% if run_wise -%}--run-wise{% endif %} {% if subrun_wise -%}--subrun-wise{% endif %}

"""


def merge_hdf_files_jobs(input_dir, output_dir, run_wise, subrun_wise, submit):

    if output_dir is None:
        output_dir = f"{str(Path(input_dir).resolve())}/merged"

    job_dict = {
        "job_name": "merge_hdf_files",
        "home": os.environ["HOME"],
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "run_wise": run_wise,
        "subrun_wise": subrun_wise,
    }

    j2_template = Template(template)
    print(j2_template.render(job_dict))
    job_submit_filename = "merge_hdf_files.slurm"
    j2_template.stream(job_dict).dump(job_submit_filename)

    if submit:
        commandargs = ["sbatch", job_submit_filename]
        sp.check_output(commandargs, shell=False)


def main():

    parser = argparse.ArgumentParser(
        description="Script to create and optionally submit jobs for merge_hdf_files script.",
        prefix_chars="-",
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        dest="input_dir",
        type=str,
        required=True,
        help="Path to DL1 files",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default=None,
        help="Path to a directory where to save the output data files and job log files",
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

    parser.add_argument(
        "--submit",
        dest="submit",
        action="store_true",
        help="Submit the job via SLURM sbatch",
    )

    flags = parser.parse_args()

    input_dir = flags.input_dir
    output_dir = flags.output_dir
    run_wise = flags.run_wise
    subrun_wise = flags.subrun_wise
    submit = flags.submit

    if run_wise and subrun_wise:
        print(
            "run_wise and subrun_wise options are not compatible. If both are selected, files will be merged subrun-wise."
        )

    merge_hdf_files_jobs(input_dir, output_dir, run_wise, subrun_wise, submit)


if __name__ == "__main__":
    main()
