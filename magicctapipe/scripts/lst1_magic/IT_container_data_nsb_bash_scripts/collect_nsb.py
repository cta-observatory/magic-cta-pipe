"""
Loop on the files produced by the parallel jobs to create a list of LST runs for each NSB level
"""
import argparse
import glob

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        "-c",
        dest="config_file",
        type=str,
        default="./config_general.yaml",
        help="Path to a configuration file",
    )

    args = parser.parse_args()
    with open(
        args.config_file, "rb"
    ) as f:  # "rb" mode opens the file in binary format for reading
        config = yaml.safe_load(f)
    source = config["directories"]["target_name"]

    nsb = config["general"]["nsb"]
    for nsblvl in nsb:
        allfile = np.sort(glob.glob(f"{source}_LST_{nsblvl}_*.txt"))
        if len(allfile) == 0:
            continue
        for j in allfile:
            with open(j) as ff:
                line = ff.readline()
                with open(f"{source}_LST_{nsblvl}_.txt", "a+") as f:
                    f.write(f"{line.rstrip()}\n")


if __name__ == "__main__":
    main()
