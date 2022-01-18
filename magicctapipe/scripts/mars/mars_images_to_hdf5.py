"""
This script takes as an input superstar files which have containers with the
calibrated and cleaned images inside (i.e. star was run with the -saveimages,
-saveimagesclean and -savecerevt flags), and saves them in HDF5 format.

This can be later used to compare the cleaned images as produced by MARS and
by magic-cta-pipe.
"""

import sys
import re
import argparse
from pathlib import Path

import numpy as np
import uproot
import tables

from ctapipe.core.container import Container, Field
from ctapipe.io import HDF5TableWriter, HDF5TableReader

from ctapipe_io_magic import MAGICEventSource


def parse_args(args):
    """
    Parse command line options and arguments.
    """

    parser = argparse.ArgumentParser(description="", prefix_chars='-')
    parser.add_argument(
        "--calibrated",
        action='store_true',
        help="Save also calibrated images."
    )
    parser.add_argument(
        "-in",
        "--input_mask",
        nargs='?',
        help='Mask for input files e.g. "20*_S_*.root" (NOTE: the double quotes should be there).'
    )

    return parser.parse_args(args)


class ImageContainerCalibrated(Container):
    obs_id = Field(-1, "Observation ID")
    event_id = Field(-1, "Event ID")
    tel_id = Field(-1, "Telescope ID")
    image_calibrated = Field(
        None,
        "Numpy array of pixels before cleaning, after calibration." "Shape: (n_pixel)",
        dtype=">f8",
        ndim=1,
    )
    image_cleaned = Field(
        None,
        "Numpy array of pixels after cleaning." "Shape: (n_pixel)",
        dtype=">f8",
        ndim=1,
    )


class ImageContainerCleaned(Container):
    obs_id = Field(-1, "Observation ID")
    event_id = Field(-1, "Event ID")
    tel_id = Field(-1, "Telescope ID")
    image_cleaned = Field(
        None,
        "Numpy array of pixels after cleaning." "Shape: (n_pixel)", dtype=">f8", ndim=1,
    )


def get_run_info_from_name(file_name):
    file_name = Path(file_name)
    file_name = file_name.name
    mask_data = r".*\d+_(\d+)_S_.*"
    mask_mc = r".*_M\d_za\d+to\d+_\d_(\d+)_Y_.*"
    mask_mc_alt = r".*_M\d_\d_(\d+)_.*"
    if re.findall(mask_data, file_name):
        parsed_info = re.findall(mask_data, file_name)
        is_mc = False
    elif re.findall(mask_mc, file_name):
        parsed_info = re.findall(mask_mc, file_name)
        is_mc = True
    else:
        parsed_info = re.findall(mask_mc_alt, file_name)
        is_mc = True

    try:
        run_number = int(parsed_info[0])
    except IndexError:
        raise IndexError(
            'Can not identify the run number and type (data/MC) of the file '
            '{:s}'.format(file_name))

    return run_number, is_mc


def build_image_container_calibrated(run_number, event_id, tel, image_calibrated, image_cleaned):
    """
    Builds a ImageContainerCalibrated

    Parameters
    ----------
    run_number : int
        Run number
    event_id : int
        Description
    tel : int
        Description
    image_calibrated : np.array
        Calibrated image (all pixels)
    image_cleaned : np.array
        Cleaned image (pixels not belonging to image set to 0)

    Returns
    -------
    ImageContainerCalibrated
        Container containing both calibrated and cleaned images
    """

    return ImageContainerCalibrated(
        obs_id=run_number,
        event_id=event_id,
        tel_id=tel,
        image_calibrated=image_calibrated,
        image_cleaned=image_cleaned,
    )


def build_image_container_cleaned(run_number, event_id, tel, image_cleaned):
    """
    Builds a ImageContainerCleaned

    Parameters
    ----------
    run_number : int
        Run number
    event_id : int
        Description
    tel : int
        Description
    image_cleaned : np.array
        Cleaned image (pixels not belonging to image set to 0)

    Returns
    -------
    ImageContainerCleaned
        Container containing cleaned image
    """

    return ImageContainerCleaned(
        obs_id=run_number,
        event_id=event_id,
        tel_id=tel,
        image_cleaned=image_cleaned,
    )


def read_images(hdf5_files_mask, read_calibrated=False):
    """
    Reads images from a HDF5 file.

    Parameters
    ----------
    hdf5_files_mask : str
        Mask for HDF5 files
    read_calibrated : bool, optional
        Flag to read also calibrated images

    Yields
    ------
    ImageContainerCalibrated or ImageContainerCleaned
        Container with cleaned images (and possibly calibrated)
    """

    with HDF5TableReader(
        filename=hdf5_files_mask,
    ) as reader:

        if read_calibrated:
            for image_container in reader.read(
                table_name="/dl1/event/telescope/image/MAGIC/M1",
                containers=ImageContainerCalibrated()
            ):
                yield image_container
            for image_container in reader.read(
                table_name="/dl1/event/telescope/image/MAGIC/M2",
                containers=ImageContainerCalibrated()
            ):
                yield image_container
        else:
            for image_container in reader.read(
                table_name="/dl1/event/telescope/image/MAGIC/M1",
                containers=ImageContainerCleaned()
            ):
                yield image_container
            for image_container in reader.read(
                table_name="/dl1/event/telescope/image/MAGIC/M2",
                containers=ImageContainerCleaned()
            ):
                yield image_container


def save_images(mars_files_mask, save_calibrated=False):
    """
    Saves cleaned images (and possibly calibrated) in a HDF5 file.

    Parameters
    ----------
    mars_files_mask : str
        Mask for input MARS files
    save_calibrated : bool, optional
        Flag to save also calibrated images
    """

    mars_files = Path(mars_files_mask)
    mars_filelist = sorted(Path(mars_files.parent).expanduser().glob(mars_files.name))

    for mars_file in mars_filelist:

        HDF5_ZSTD_FILTERS = tables.Filters(
            complevel=5,            # enable compression, 5 is a good tradeoff between compression and speed
            complib='blosc:zstd',   # compression using blosc/zstd
            fletcher32=True,        # attach a checksum to each chunk for error correction
            bitshuffle=False,       # for BLOSC, shuffle bits for better compression
        )

        output_filename = Path(mars_file).name.replace(".root", ".h5")

        with HDF5TableWriter(
            filename=output_filename,
            group_name='dl1/event',
            mode='a',
            filters=HDF5_ZSTD_FILTERS,
            add_prefix=False,
            # overwrite=True,
        ) as writer:

            with MAGICEventSource(input_url=mars_file) as source:
                run_info = source.get_run_info_from_name(mars_file)

            run_number = run_info[0]
            telescope = run_info[2]
            datalevel = run_info[3]
            print(f"Opening {mars_file} ...")

            with uproot.open(mars_file) as sstar:

                events = sstar["Events"]

                print(f"Writing output in {output_filename}")

                batch_no = 0
                if save_calibrated:
                    if datalevel == 1:
                        branches = [
                            "UprootImageOrig",
                            "UprootImageOrigClean",
                            "MRawEvtHeader.fStereoEvtNumber"
                        ]
                    else:
                        branches = [
                            "UprootImageOrig_1",
                            "UprootImageOrigClean_1",
                            "MRawEvtHeader_1.fStereoEvtNumber",
                            "UprootImageOrig_2",
                            "UprootImageOrigClean_2",
                            "MRawEvtHeader_2.fStereoEvtNumber"
                        ]
                else:
                    if datalevel == 1:
                        branches = [
                            "UprootImageOrigClean",
                            "MRawEvtHeader.fStereoEvtNumber"
                        ]
                    else:
                        branches = [
                            "UprootImageOrigClean_1",
                            "MRawEvtHeader_1.fStereoEvtNumber",
                            "UprootImageOrigClean_2",
                            "MRawEvtHeader_2.fStereoEvtNumber"
                        ]
                for batch in events.iterate(
                        step_size="10 MB",
                        expressions=branches,
                        library="np"
                ):
                    print(f"Writing batch of events {batch_no+1}")
                    if datalevel == 1:
                        for j in range(len(batch["MRawEvtHeader.fStereoEvtNumber"])):
                            if save_calibrated:
                                image_container = build_image_container_calibrated(
                                    run_number,
                                    batch["MRawEvtHeader.fStereoEvtNumber"][j],
                                    1,
                                    np.array(batch["UprootImageOrig"][j]),
                                    np.array(batch["UprootImageOrigClean"][j])
                                    )
                                # write image first, so we are sure nothing here modifies it
                                writer.write(
                                    table_name=f'telescope/image/MAGIC/M{telescope}',
                                    containers=image_container
                                )
                            else:
                                image_container = build_image_container_cleaned(
                                        run_number,
                                        batch["MRawEvtHeader.fStereoEvtNumber"][j],
                                        1,
                                        np.array(batch["UprootImageOrigClean"][j])
                                    )
                                # write image first, so we are sure nothing here modifies it
                                writer.write(
                                    table_name=f'telescope/image/MAGIC/M{telescope}',
                                    containers=image_container
                                )
                    else:
                        for j in range(len(batch["MRawEvtHeader_1.fStereoEvtNumber"])):
                            if save_calibrated:
                                image_container = build_image_container_calibrated(
                                        run_number,
                                        batch["MRawEvtHeader_1.fStereoEvtNumber"][j],
                                        1,
                                        np.array(batch["UprootImageOrig_1"][j]),
                                        np.array(batch["UprootImageOrigClean_1"][j])
                                    )
                                # write image first, so we are sure nothing here modifies it
                                writer.write(
                                    table_name='telescope/image/MAGIC/M1',
                                    containers=image_container
                                )
                                image_container = build_image_container_calibrated(
                                        run_number,
                                        batch["MRawEvtHeader_2.fStereoEvtNumber"][j],
                                        2,
                                        np.array(batch["UprootImageOrig_2"][j]),
                                        np.array(batch["UprootImageOrigClean_2"][j])
                                    )
                                # write image first, so we are sure nothing here modifies it
                                writer.write(
                                    table_name='telescope/image/MAGIC/M2',
                                    containers=image_container
                                )
                            else:
                                image_container = build_image_container_cleaned(
                                        run_number,
                                        batch["MRawEvtHeader_1.fStereoEvtNumber"][j],
                                        1,
                                        np.array(batch["UprootImageOrigClean_1"][j])
                                    )
                                # write image first, so we are sure nothing here modifies it
                                writer.write(
                                    table_name='telescope/image/MAGIC/M1',
                                    containers=image_container
                                )
                                image_container = build_image_container_cleaned(
                                        run_number,
                                        batch["MRawEvtHeader_2.fStereoEvtNumber"][j],
                                        2,
                                        np.array(batch["UprootImageOrigClean_2"][j])
                                    )
                                # write image first, so we are sure nothing here modifies it
                                writer.write(
                                    table_name='telescope/image/MAGIC/M2',
                                    containers=image_container
                                )
                    batch_no += 1


def main(*args):
    flags = parse_args(args)

    save_calibrated = flags.calibrated
    input_mask = flags.input_mask

    save_images(input_mask, save_calibrated)


if __name__ == '__main__':
    main(*sys.argv[1:])
