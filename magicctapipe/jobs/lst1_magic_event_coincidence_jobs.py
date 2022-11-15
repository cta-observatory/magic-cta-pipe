import os
import argparse
import glob

import subprocess as sp

from jinja2 import Template
from pathlib import Path


__all__ = ["lst1_magic_event_coincidence_jobs"]


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
{% for infile in file_list_lst %}
    {{ loop.index0 }}) INPUT_FILE={{ infile }};;
{% endfor %}
esac

lst1_magic_event_coincidence -l $INPUT_FILE -m {{ input_dir_magic }} -o {{ output_dir}} -c {{ config_file }}

"""


def lst1_magic_event_coincidence_jobs(
    input_dir_lst, input_dir_magic, output_dir, config_file, submit
):

    file_list_lst = [
        f"{str(Path(filename).parent.resolve())}/{Path(filename).name}"
        for filename in sorted(
            glob.glob(f"{str(Path(input_dir_lst).resolve())}/dl1_LST-1*.h5")
        )
    ]

    n_jobs = len(file_list_lst)

    job_dict = {
        "job_name": "magic_calib_to_dl1",
        "stop_job": n_jobs - 1,
        "home": os.environ["HOME"],
        "file_list_lst": file_list_lst,
        "input_dir_magic": str(Path(input_dir_magic).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "config_file": f"{str(Path(config_file).parent.resolve())}/{Path(config_file).name}",
    }

    j2_template = Template(template)
    print(j2_template.render(job_dict))
    job_submit_filename = "lst1_magic_event_coincidence.slurm"
    j2_template.stream(job_dict).dump(job_submit_filename)

    if submit:
        commandargs = ["sbatch", job_submit_filename]
        sp.check_output(commandargs, shell=False)


def main():

    parser = argparse.ArgumentParser(
        description="Script to create and optionally submit jobs for lst1_magic_event_coincidence script.",
        prefix_chars="-",
    )

    parser.add_argument(
        "--input-dir-lst",
        "-l",
        dest="input_dir_lst",
        type=str,
        required=True,
        help="Path to directory with LST-1 DL1 data files",
    )

    parser.add_argument(
        "--input-dir-magic",
        "-m",
        dest="input_dir_magic",
        type=str,
        required=True,
        help="Path to a directory where input MAGIC DL1 data files are stored",
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

    input_dir_lst = flags.input_dir_lst
    input_dir_magic = flags.input_dir_magic
    output_dir = flags.output_dir
    config_file = flags.config_file
    submit = flags.submit

    lst1_magic_event_coincidence_jobs(
        input_dir_lst, input_dir_magic, output_dir, config_file, submit
    )


if __name__ == "__main__":
    main()
