import os
import argparse
import glob

import subprocess as sp

from jinja2 import Template
from pathlib import Path


__all__ = ["create_dl3_index_files_jobs"]


template = """#!/bin/bash
#SBATCH -A aswg
#SBATCH -p short,long,xxl
#SBATCH -J {{ job_name }}_%A
#SBATCH -o {{ output_dir }}/{{ job_name }}_%A.out
#SBATCH -e {{ output_dir }}/{{ job_name }}_%A.err

source {{ home }}/.bashrc
conda activate magic-lst1

create_dl3_index_files -i {{ input_dir }}  
"""


def create_dl3_index_files_jobs(input_dir, submit):

    job_dict = {
        "job_name": "create_dl3_index_files",
        "home": os.environ["HOME"],
        "input_dir": str(Path(input_dir).resolve()),
    }

    j2_template = Template(template)
    print(j2_template.render(job_dict))
    job_submit_filename = "create_dl3_index_files.slurm"
    j2_template.stream(job_dict).dump(job_submit_filename)

    if submit:
        commandargs = ["sbatch", job_submit_filename]
        sp.check_output(commandargs, shell=False)


def main():

    parser = argparse.ArgumentParser(
        description="Script to create and optionally submit jobs for create_dl3_index_files script.",
        prefix_chars="-",
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        dest="input_dir",
        type=str,
        required=True,
        help="Path to a directory where input DL3 data files are stored",
    )

    parser.add_argument(
        "--submit",
        dest="submit",
        action="store_true",
        help="Submit the job via SLURM sbatch",
    )

    flags = parser.parse_args()

    input_dir = flags.input_dir
    submit = flags.submit

    create_dl3_index_files_jobs(input_dir, submit)


if __name__ == "__main__":
    main()
