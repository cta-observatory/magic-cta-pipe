import os
import argparse
import glob

import subprocess as sp

from jinja2 import Template
from pathlib import Path


__all__ = ["lst1_magic_stereo_reco_jobs"]


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

lst1_magic_stereo_reco -i $INPUT_FILE -o {{ output_dir }} -c {{ config_file }} {% if magic_only -%}--magic-only{% endif %}

"""


def lst1_magic_stereo_reco_jobs(
    input_files, output_dir, config_file, magic_only, submit
):

    file_list = [
        f"{str(Path(filename).parent.resolve())}/{Path(filename).name}"
        for filename in sorted(glob.glob(input_files))
    ]

    n_jobs = len(file_list)

    job_dict = {
        "job_name": "lst1_magic_stereo_reco",
        "stop_job": n_jobs - 1,
        "home": os.environ["HOME"],
        "file_list": file_list,
        "output_dir": str(Path(output_dir).resolve()),
        "config_file": f"{str(Path(config_file).parent.resolve())}/{Path(config_file).name}",
        "magic_only": magic_only,
    }

    j2_template = Template(template)
    print(j2_template.render(job_dict))
    job_submit_filename = "lst1_magic_stereo_reco.slurm"
    j2_template.stream(job_dict).dump(job_submit_filename)

    if submit:
        commandargs = ["sbatch", job_submit_filename]
        sp.check_output(commandargs, shell=False)


def main():

    parser = argparse.ArgumentParser(
        description="Script to create and optionally submit jobs for lst1_magic_stereo_reco script.",
        prefix_chars="-",
    )

    parser.add_argument(
        "--input-files",
        "-i",
        dest="input_files",
        type=str,
        required=True,
        help="Path to input DL1 data files",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output DL1-stereo data file",
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
        "--magic-only",
        dest="magic_only",
        action="store_true",
        help="Reconstruct the stereo parameters using only MAGIC events",
    )

    parser.add_argument(
        "--submit",
        dest="submit",
        action="store_true",
        help="Submit the job via SLURM sbatch",
    )

    flags = parser.parse_args()

    input_files = flags.input_files
    output_dir = flags.output_dir
    config_file = flags.config_file
    magic_only = flags.magic_only
    submit = flags.submit

    lst1_magic_stereo_reco_jobs(
        input_files, output_dir, config_file, magic_only, submit
    )


if __name__ == "__main__":
    main()
