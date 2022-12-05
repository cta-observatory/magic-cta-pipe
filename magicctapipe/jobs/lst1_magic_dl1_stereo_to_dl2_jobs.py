import os
import argparse
import glob

import subprocess as sp

from jinja2 import Template
from pathlib import Path

__all__ = ["lst1_magic_dl1_stereo_to_dl2_jobs"]

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

lst1_magic_dl1_stereo_to_dl2 -d $INPUT_FILE -o {{ output_dir }} -r {{ input_dir_rfs }}

"""

def lst1_magic_dl1_stereo_to_dl2_jobs(
	input_files, output_dir, input_dir_rfs, submit
):
	file_list = [
        f"{str(Path(filename).parent.resolve())}/{Path(filename).name}"
        for filename in sorted(glob.glob(input_files))
    ]

    n_jobs = len(file_list)

    job_dict = {
        "job_name": "lst1_magic_dl1_stereo_to_dl2",
        "stop_job": n_jobs - 1,
        "home": os.environ["HOME"],
        "file_list": file_list,
        "output_dir": str(Path(output_dir).resolve()),
        "input_dir_rfs": str(Path(input_dir_rfs).resolve()),
    }

    j2_template = Template(template)
    print(j2_template.render(job_dict))
    job_submit_filename = "lst1_magic_dl1_stereo_to_dl2.slurm"
    j2_template.stream(job_dict).dump(job_submit_filename)

    if submit:
        commandargs = ["sbatch", job_submit_filename]
        sp.check_output(commandargs, shell=False)

def main():

    parser = argparse.ArgumentParser(
        description="Script to create and optionally submit jobs for lst1_magic_dl1_stereo_to_dl2 script.",
        prefix_chars="-",
    )

    parser.add_argument(
        "--input-file_dl1",
        "-d",
        dest="input_file_dl1",
        type=str,
        required=True,
        help="Path to input DL1-stereo data files",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save an output DL2 data file",
    )

    parser.add_argument(
    	"--input-dir-rfs",
        "-r",
        dest="input_dir_rfs",
        type=str,
        required=True,
        default="./rfs",
        help="Path to a directory that contains the RF files",
    )

    parser.add_argument(
        "--submit",
        dest="submit",
        action="store_true",
        help="Submit the job via SLURM sbatch",
    )

    flags = parser.parse_args()

    input_file_dl1 = flags.input_file_dl1
    output_dir = flags.output_dir
    input_dir_rfs = flags.input_dir_rfs
    submit = flags.submit

    lst1_magic_dl1_stereo_to_dl2_jobs(
        input_file_dl1, output_dir,input_dir_rfs, submit
    )


if __name__ == "__main__":
    main()