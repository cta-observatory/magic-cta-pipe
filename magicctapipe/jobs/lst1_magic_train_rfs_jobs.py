import os
import argparse
import glob

import subprocess as sp

from jinja2 import Template
from pathlib import Path

__all__ = ["lst1_magic_train_rfs_jobs"]

template = """#!/bin/bash
#SBATCH -A aswg
#SBATCH -p short,long,xxl
#SBATCH -J {{ job_name }}_%A
#SBATCH -o {{ output_dir }}/{{ job_name }}_%A_%a.out
#SBATCH -e {{ output_dir }}/{{ job_name }}_%A_%a.err
#SBATCH --array=0-{{ stop_job }}

source {{ home }}/.bashrc
conda activate magic-lst1

lst1_magic_train_rfs -g {{ input_dir_gamma }} -p {{ input_dir_proton }} -o {{ output_dir }} -c {{ config_file }} \
{% if train_energy -%}--train-energy{% endif %} {% if train_disp -%}--train-disp{% endif %} \
{% if train_classifier -%}--train-classifier{% endif %} {% if use_unsigned -%}--use-unsigned{% endif %}

"""

def lst1_magic_train_rfs_jobs(
	input_dir_gamma, input_dir_proton, output_dir, train_energy, train_disp, train_classifier, \
    use_unisgned, config_file, submit
):
    job_dict = {
        "job_name": "lst1_magic_train_rfs",
        "home": os.environ["HOME"],
        "output_dir": str(Path(output_dir).resolve()),
        "input_dir_gamma": str(Path(input_dir_gamma).resolve()),
        "input_dir_proton": str(Path(input_dir_proton).resolve()),
        "train_energy": train_energy
        "train_disp": train_disp
        "train_classifier": train_classifier
        "use_unsigned": use_unisgned
        "config_file": f"{str(Path(config_file).parent.resolve())}/{Path(config_file).name}",
    }

    j2_template = Template(template)
    print(j2_template.render(job_dict))
    job_submit_filename = "lst1_magic_train_rfs.slurm"
    j2_template.stream(job_dict).dump(job_submit_filename)

    if submit:
        commandargs = ["sbatch", job_submit_filename]
        sp.check_output(commandargs, shell=False)

def main():

    parser = argparse.ArgumentParser(
        description="Script to create and optionally submit jobs for lst1_magic_train_rfs script.",
        prefix_chars="-",
    )

     parser.add_argument(
        "--input-dir-gamma",
        "-g",
        dest="input_dir_gamma",
        type=str,
        required=True,
        help="Path to a directory where input gamma MC data files are stored",
    )

    parser.add_argument(
        "--input-dir-proton",
        "-p",
        dest="input_dir_proton",
        type=str,
        help="Path to a directory where input proton MC data files are stored",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
        help="Path to a directory where to save trained RFs",
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
        "--train-energy",
        dest="train_energy",
        action="store_true",
        help="Train energy regressors",
    )

    parser.add_argument(
        "--train-disp",
        dest="train_disp",
        action="store_true",
        help="Train DISP regressors",
    )

    parser.add_argument(
        "--train-classifier",
        dest="train_classifier",
        action="store_true",
        help="Train event classifiers",
    )

    parser.add_argument(
        "--use-unsigned",
        dest="use_unsigned",
        action="store_true",
        help="Use unsigned features for training RFs",
    )

    parser.add_argument(
        "--submit",
        dest="submit",
        action="store_true",
        help="Submit the job via SLURM sbatch",
    )

    flags = parser.parse_args()

    input_dir_gamma = flags.input_dir_gamma
    input_dir_proton = flags.input_dir_proton
    output_dir = flags.output_dir
    config_file = flags.config_file
    train_energy = flags.train_energy
    train_disp = flags.train_disp
    train_classifier = flags.train_classifier
    use_unsigned = flags.use_unsigned
    submit = flags.submit

    lst1_magic_train_rfs_jobs(
        input_dir_proton, input_dir_gamma, output_dir,input_dir_irf, config_file, train_energy, train_classifier, train_disp, submit
    )


if __name__ == "__main__":
    main()