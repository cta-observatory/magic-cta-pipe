import os
import argparse
import glob

import subprocess as sp

from jinja2 import Template
from pathlib import Path

__all__ = ["lst1_magic_dl2_to_dl3_jobs"]

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

lst1_magic_dl2_to_dl3 -d $INPUT_FILE -o {{ output_dir }} -i {{ input_dir_irf }} -c {{ config_file }}

"""

def lst1_magic_dl2_to_dl3_jobs(
	input_file_dl2, output_dir, input_dir_irf, config_file, submit
):
	file_list = [
        f"{str(Path(filename).parent.resolve())}/{Path(filename).name}"
        for filename in sorted(glob.glob(input_file_dl2))
    ]

    n_jobs = len(file_list)

    job_dict = {
        "job_name": "lst1_magic_dl2_to_dl3",
        "stop_job": n_jobs - 1,
        "home": os.environ["HOME"],
        "file_list": file_list,
        "output_dir": str(Path(output_dir).resolve()),
        "input_dir_irf": str(Path(input_dir_irf).resolve()),
        "config_file": f"{str(Path(config_file).parent.resolve())}/{Path(config_file).name}",
    }

    j2_template = Template(template)
    print(j2_template.render(job_dict))
    job_submit_filename = "lst1_magic_dl2_to_dl3.slurm"
    j2_template.stream(job_dict).dump(job_submit_filename)

    if submit:
        commandargs = ["sbatch", job_submit_filename]
        sp.check_output(commandargs, shell=False)

def main():

    parser = argparse.ArgumentParser(
        description="Script to create and optionally submit jobs for lst1_magic_dl2_to_dl3 script.",
        prefix_chars="-",
    )

    parser.add_argument(
        "--input-file_dl2",
        "-i",
        dest="input_file_dl2",
        type=str,
        required=True,
        help="Path to input DL2 data files",
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
    	"--input-dir-irf",
        "-i",
        dest="input_dir_irf",
        type=str,
        required=True,
        default="./rfs",
        help="Path to a directory that contains the IRF files",
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

    input_file_dl2 = flags.input_file_dl2
    output_dir = flags.output_dir
    input_dir_irf = flags.input_dir_irf
    config_file = flags.config_file
    submit = flags.submit

    lst1_magic_dl2_to_dl3_jobs(
        input_file_dl2, output_dir,input_dir_irf, config_file, submit
    )


if __name__ == "__main__":
    main()