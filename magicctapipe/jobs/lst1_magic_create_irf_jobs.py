import os
import argparse
import glob

import subprocess as sp

from jinja2 import Template
from pathlib import Path

__all__ = ["lst1_magic_create_irf_jobs"]

template = """#!/bin/bash
#SBATCH -A aswg
#SBATCH -p short,long,xxl
#SBATCH -J {{ job_name }}_%A
#SBATCH -o {{ output_dir }}/{{ job_name }}_%A_%a.out
#SBATCH -e {{ output_dir }}/{{ job_name }}_%A_%a.err
#SBATCH --array=0-{{ stop_job }}

source {{ home }}/.bashrc
conda activate magic-lst1

case $SLURM_ARRAY_TASK_ID in
{% for infile in file_list %}
    {{ loop.index0 }}) INPUT_FILE={{ infile }};;
{% endfor %}
esac

lst1_magic_create_irf -g {{ input_file_gamma }} -p {{ input_file_proton }} -e {{ input_file_electron }} -o {{ output_dir }} -c {{ config_file }}

"""

def lst1_magic_create_irf_jobs(
	input_file_gamma, input_file_proton, input_file_electron, output_dir, config_file, submit
):
	file_list_gamma = [
        f"{str(Path(filename).parent.resolve())}/{Path(filename).name}"
        for filename in sorted(glob.glob(input_file_gamma))
    ]

    file_list_proton = [
        f"{str(Path(filename).parent.resolve())}/{Path(filename).name}"
        for filename in sorted(glob.glob(input_file_proton))
    ]

    file_list_electron = [
        f"{str(Path(filename).parent.resolve())}/{Path(filename).name}"
        for filename in sorted(glob.glob(input_file_electron))
    ]

    n_jobs = len(file_list)

    job_dict = {
        "job_name": "lst1_magic_create_irf",
        "stop_job": n_jobs - 1,
        "home": os.environ["HOME"],
        "file_list_gamma": file_list_gamma,
        "file_list_proton": file_list_proton,
        "file_list_electron": file_list_electron,
        "output_dir": str(Path(output_dir).resolve()),
        "config_file": f"{str(Path(config_file).parent.resolve())}/{Path(config_file).name}",
    }

    j2_template = Template(template)
    print(j2_template.render(job_dict))
    job_submit_filename = "lst1_magic_create_irf.slurm"
    j2_template.stream(job_dict).dump(job_submit_filename)

    if submit:
        commandargs = ["sbatch", job_submit_filename]
        sp.check_output(commandargs, shell=False)

def main():

    parser = argparse.ArgumentParser(
        description="Script to create and optionally submit jobs for lst1_magic_create_irf script.",
        prefix_chars="-",
    )

    
    parser.add_argument(
    "--input-file-gamma",
        "-g",
        dest="input_file_gamma",
        type=str,
        required=True,
        help="Path to an input gamma MC DL2 data file",
    )

    parser.add_argument(
        "--input-file-proton",
        "-p",
        dest="input_file_proton",
        type=str,
        help="Path to an input proton MC DL2 data file",
    )

    parser.add_argument(
        "--input-file-electron",
        "-e",
        dest="input_file_electron",
        type=str,
        help="Path to an input electron MC DL2 data file",
    )
    parser.add_argument(
        "--output-dir",
        "-d",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output DL3 data file",
    )

    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config.yaml",
        help="Path to a configuration file",
    )

    parser.add_argument(
        "--submit",
        dest="submit",
        action="store_true",
        help="Submit the job via SLURM sbatch",
    )
    
    flags = parser.parse_args()

    input_file_gamma = flags.input_file_gamma
    input_file_electron = flags.input_file_electron
    input_file_proton = flags.input_file_proton
    output_dir = flags.output_dir
    config_file = flags.config_file
    submit = flags.submit

    lst1_magic_create_irf_jobs(
        input_file_gamma, input_file_electron, input_file_proton, output_dir, config_file, submit
    )


if __name__ == "__main__":
    main()