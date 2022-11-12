import os
import sys
import argparse
import glob

from jinja2 import Template
from pathlib import Path


template = """
#!/bin/bash
#SBATCH -A aswg
#SBATCH -p short,long,xxl
#SBATCH -J {{ job_name }}
#SBATCH -o {{ output_dir }}
#SBATCH --array=0-{{ stop_job }}

source {{ home }}/.bashrc
conda activate magic-lst1

case $SLURM_ARRAY_TASK_ID in
{% for infile in file_list %}
    {{ loop.index0 }}) INPUT_FILE={{ infile }};;
{% endfor %}
esac

magic_calib_to_dl1 -i $INPUT_FILE -o {{ output_dir }} -c {{ config_file }} {% if process_run -%}--process-run{% endif %}
"""


def parse_args(args):
    """
    Parse command line options and arguments.
    """

    parser = argparse.ArgumentParser(
        description="Check evolution of survival fraction of pedestal events.",
        prefix_chars="-",
    )
    parser.add_argument(
        "--input-files",
        "-i",
        dest="input_files",
        type=str,
        required=True,
        help="Path to MAGIC calibrated files",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        type=str,
        default="./data",
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
        "--process-run",
        dest="process_run",
        action="store_true",
        help="Process the events of all the subrun files at once",
    )

    return parser.parse_args(args)


def main(*args):

    flags = parse_args(args)

    input_files = flags.input_files
    output_dir = flags.output_dir
    config_file = flags.config_file
    process_run = flags.process_run

    file_list = [
        f"{str(Path(filename).parent.resolve())}/{Path(filename).name}"
        for filename in sorted(glob.glob(input_files))
    ]

    n_jobs = len(file_list)

    job_dict = {
        "job_name": "magic_calib_to_dl1",
        "stop_job": n_jobs - 1,
        "home": os.environ["HOME"],
        "file_list": file_list,
        "output_dir": str(Path(output_dir).resolve()),
        "config_file": f"{str(Path(config_file).parent.resolve())}/{Path(config_file).name}",
        "process_run": process_run,
    }

    j2_template = Template(template)
    print(j2_template.render(job_dict))


if __name__ == "__main__":
    main(*sys.argv[1:])
