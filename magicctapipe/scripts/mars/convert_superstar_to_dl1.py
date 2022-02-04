#!/usr/bin/env python

import re
import sys
import argparse
from pathlib import Path

import uproot
import pandas as pd
import numpy as np

from astropy.table import QTable, vstack
import astropy.units as u

from ctapipe.containers import (
    CameraHillasParametersContainer,
    LeakageContainer,
    CameraTimingParametersContainer,
    ReconstructedGeometryContainer,
)
from ctapipe.io import HDF5TableWriter
from ctapipe.core.container import Container, Field
from astropy.coordinates import AltAz, SkyCoord
from ctapipe.coordinates import CameraFrame, TelescopeFrame
from ctapipe.instrument import (
    CameraDescription,
    TelescopeDescription,
    OpticsDescription,
)

magic_optics = OpticsDescription.from_name('MAGIC')
magic_cam = CameraDescription.from_name('MAGICCam')
magic_tel_description = TelescopeDescription(name='MAGIC',
                                             tel_type='MAGIC',
                                             optics=magic_optics,
                                             camera=magic_cam)
magic_tel_descriptions = {1: magic_tel_description,
                          2: magic_tel_description}

magic_bdec = u.Quantity(-7.0, u.deg).to(u.rad)

columns_mc = {
        'event_id': ('MRawEvtHeader_1.fStereoEvtNumber', dict(dtype=int)),
        'true_energy': ('MMcEvt_1.fEnergy', dict(unit=u.GeV)),
        'true_pointing_zd': ('MMcEvt_1.fTelescopeTheta', dict(unit=u.rad)),
        'true_pointing_az': ('MMcEvt_1.fTelescopePhi', dict(unit=u.rad)),
        'true_zd': ('MMcEvt_1.fTheta', dict(unit=u.rad)),
        'true_az': ('MMcEvt_1.fPhi', dict(unit=u.rad)),
        'true_core_x': ('MMcEvt_1.fCoreX', dict(unit=u.cm)),
        'true_core_y': ('MMcEvt_1.fCoreY', dict(unit=u.cm)),
        'particle_id': ('MMcEvt_1.fPartId', dict()),
        'length1': ('MHillas_1.fLength', dict(unit=u.mm)),
        'length2': ('MHillas_2.fLength', dict(unit=u.mm)),
        'psi1': ('MHillas_1.fDelta', dict(unit=u.rad)),
        'psi2': ('MHillas_2.fDelta', dict(unit=u.rad)),
        'width1': ('MHillas_1.fWidth', dict(unit=u.mm)),
        'width2': ('MHillas_2.fWidth', dict(unit=u.mm)),
        'size1': ('MHillas_1.fSize', dict()),
        'size2': ('MHillas_2.fSize', dict()),
        'leakage1_1': ('MNewImagePar_1.fLeakage1', dict()),
        'leakage2_1': ('MNewImagePar_1.fLeakage2', dict()),
        'leakage1_2': ('MNewImagePar_2.fLeakage1', dict()),
        'leakage2_2': ('MNewImagePar_2.fLeakage2', dict()),
        'x1': ('MHillas_1.fMeanX', dict(unit=u.mm)),
        'y1': ('MHillas_1.fMeanY', dict(unit=u.mm)),
        'x2': ('MHillas_2.fMeanX', dict(unit=u.mm)),
        'y2': ('MHillas_2.fMeanY', dict(unit=u.mm)),
        'slope1': ('MHillasTimeFit_1.fP1Grad', dict(unit=1/u.mm)),
        'slope2': ('MHillasTimeFit_2.fP1Grad', dict(unit=1/u.mm)),
        'reco_zd': ('MStereoPar.fDirectionZd', dict(unit=u.deg)),
        'reco_az': ('MStereoPar.fDirectionAz', dict(unit=u.deg)),
        'reco_source_x': ('MStereoPar.fDirectionX', dict(unit=u.deg)),
        'reco_source_y': ('MStereoPar.fDirectionY', dict(unit=u.deg)),
        'reco_core_x': ('MStereoPar.fCoreX', dict(unit=u.cm)),
        'reco_core_y': ('MStereoPar.fCoreY', dict(unit=u.cm)),
        'hmax': ('MStereoPar.fMaxHeight', dict(unit=u.cm)),
        'impact1': ('MStereoPar.fM1Impact', dict(unit=u.cm)),
        'impact2': ('MStereoPar.fM2Impact', dict(unit=u.cm)),
        'theta2': ('MStereoPar.fTheta2', dict(unit=u.deg**2)),
        'is_valid': ('MStereoPar.fValid', dict(dtype=int)),
}

columns_mc_orig = {
        'true_energy_1': ('MMcEvtBasic_1.fEnergy', dict(unit=u.GeV)),
        'tel_az_1': ('MMcEvtBasic_1.fTelescopePhi', dict(unit=u.rad)),
        'tel_zd_1': ('MMcEvtBasic_1.fTelescopeTheta', dict(unit=u.rad)),
        'cam_x_1': ('MSrcPosCam_1.fX', dict(unit=u.mm)),
        'cam_y_1': ('MSrcPosCam_1.fY', dict(unit=u.mm)),
        'true_energy_2': ('MMcEvtBasic_2.fEnergy', dict(unit=u.GeV)),
        'tel_az_2': ('MMcEvtBasic_2.fTelescopePhi', dict(unit=u.rad)),
        'tel_zd_2': ('MMcEvtBasic_2.fTelescopeTheta', dict(unit=u.rad)),
        'cam_x_2': ('MSrcPosCam_2.fX', dict(unit=u.mm)),
        'cam_y_2': ('MSrcPosCam_2.fY', dict(unit=u.mm)),
}

columns_data = {
        'event_id': ('MRawEvtHeader_1.fStereoEvtNumber', dict(dtype=int)),
        'pointing_zen': ('MPointingPos_1.fZd', dict(unit=u.deg)),
        'pointing_az': ('MPointingPos_1.fAz', dict(unit=u.deg)),
        'mjd1': ('MTime_1.fMjd', dict(dtype=float)),
        'mjd2': ('MTime_2.fMjd', dict(dtype=float)),
        'millisec1': ('MTime_1.fTime.fMilliSec', dict(dtype=float)),
        'millisec2': ('MTime_2.fTime.fMilliSec', dict(dtype=float)),
        'nanosec1': ('MTime_1.fNanoSec', dict(dtype=float)),
        'nanosec2': ('MTime_2.fNanoSec', dict(dtype=float)),
        'length1': ('MHillas_1.fLength', dict(unit=u.mm)),
        'length2': ('MHillas_2.fLength', dict(unit=u.mm)),
        'psi1': ('MHillas_1.fDelta', dict(unit=u.rad)),
        'psi2': ('MHillas_2.fDelta', dict(unit=u.rad)),
        'width1': ('MHillas_1.fWidth', dict(unit=u.mm)),
        'width2': ('MHillas_2.fWidth', dict(unit=u.mm)),
        'size1': ('MHillas_1.fSize', dict()),
        'size2': ('MHillas_2.fSize', dict()),
        'leakage1_1': ('MNewImagePar_1.fLeakage1', dict()),
        'leakage2_1': ('MNewImagePar_1.fLeakage2', dict()),
        'leakage1_2': ('MNewImagePar_2.fLeakage1', dict()),
        'leakage2_2': ('MNewImagePar_2.fLeakage2', dict()),
        'x1': ('MHillas_1.fMeanX', dict(unit=u.mm)),
        'y1': ('MHillas_1.fMeanY', dict(unit=u.mm)),
        'x2': ('MHillas_2.fMeanX', dict(unit=u.mm)),
        'y2': ('MHillas_2.fMeanY', dict(unit=u.mm)),
        'slope1': ('MHillasTimeFit_1.fP1Grad', dict(unit=1/u.mm)),
        'slope2': ('MHillasTimeFit_2.fP1Grad', dict(unit=1/u.mm)),
        'reco_zd': ('MStereoPar.fDirectionZd', dict(unit=u.deg)),
        'reco_az': ('MStereoPar.fDirectionAz', dict(unit=u.deg)),
        'reco_core_x': ('MStereoPar.fCoreX', dict(unit=u.cm)),
        'reco_core_y': ('MStereoPar.fCoreY', dict(unit=u.cm)),
        'hmax': ('MStereoPar.fMaxHeight', dict(unit=u.cm)),
        'impact1': ('MStereoPar.fM1Impact', dict(unit=u.cm)),
        'impact2': ('MStereoPar.fM2Impact', dict(unit=u.cm)),
        'theta2': ('MStereoPar.fTheta2', dict(unit=u.deg**2)),
        'is_valid': ('MStereoPar.fValid', dict(dtype=int)),
}


class InfoContainerMC(Container):
    obs_id = Field(-1, "Observation ID")
    event_id = Field(-1, "Event ID")
    tel_id = Field(-1, "Telescope ID")
    true_energy = Field(-1 * u.TeV, "MC event energy", unit=u.TeV)
    true_alt = Field(-1 * u.rad, "MC event altitude", unit=u.rad)
    true_az = Field(-1 * u.rad, "MC event azimuth", unit=u.rad)
    true_core_x = Field(-1 * u.m, "MC event x-core position", unit=u.m)
    true_core_y = Field(-1 * u.m, "MC event y-core position", unit=u.m)
    tel_alt = Field(-1 * u.rad, "MC telescope altitude", unit=u.rad)
    tel_az = Field(-1 * u.rad, "MC telescope azimuth", unit=u.rad)


class InfoContainerData(Container):
    obs_id = Field(-1, "Observation ID")
    event_id = Field(-1, "Event ID")
    tel_id = Field(-1, "Telescope ID")
    mjd = Field(-1, "Event MJD", dtype=np.float64)
    tel_alt = Field(-1, "Telescope altitude", unit=u.rad)
    tel_az = Field(-1, "Telescope azimuth", unit=u.rad)


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


def parse_args(args):
    """
    Parse command line options and arguments.
    """

    parser = argparse.ArgumentParser(description="", prefix_chars='-')
    parser.add_argument("--use_mc", action='store_true', help="Read MC data if flag is specified.")
    parser.add_argument("-in", "--input_mask", nargs='?', help='Mask for input files e.g. "20*_S_*.root" (NOTE: the double quotes should be there).')

    return parser.parse_args(args)


def write_hdf5_mc(filelist):
    """
    Writes an HDF5 file for each superstar file in
    filelist. Specific for MC files.

    Parameters
    ----------
    filelist : list
        A list of files to be opened.
    """

    obs_id = 0
    columns = columns_mc

    for path in filelist:
        print(f'Opening {path}')
        with uproot.open(path) as f:

            events_tree = f['Events']
            events = QTable()

            for column, (branch, kwargs) in columns.items():
                events[column] = u.Quantity(events_tree[branch].array(library="np"), copy=False, **kwargs)

            events = vstack(events)
            outfile = str(path).replace(".root", ".h5")

            with HDF5TableWriter(filename=outfile, group_name='dl1', overwrite=True) as writer:
                print(f'Writing in {outfile}')
                event_info = dict()
                hillas_params = dict()
                timing_params = dict()
                leakage_params = dict()
                id_prev = events[0]["event_id"]
                for event in events:
                    id_current = event["event_id"]
                    if id_current < id_prev:
                        obs_id += 1
                    # correction needed for true_az since it is in the corsika
                    # reference frame. Also, wrapped in [-180, 180] interval
                    # see e.g. MHMcEnergyMigration.cc lines 567-570
                    true_az = np.pi - event["true_az"].value + magic_bdec.value
                    if true_az > np.pi:
                        true_az -= 2*np.pi
                    # correction needed for tel_az since it is in the corsika
                    # reference frame. Also, wrapped in [0, 360] interval
                    # see e.g. MPointingPosCalc.cc lines 149-153
                    tel_az = np.pi - event["true_pointing_az"].value + magic_bdec.value
                    if tel_az < 0:
                        tel_az += 2*np.pi
                    if tel_az > 2*np.pi:
                        tel_az -= 2*np.pi
                    if event["is_valid"] > 0:
                        is_valid = True
                    else:
                        is_valid = False
                    event_info[1] = InfoContainerMC(
                            obs_id=obs_id,
                            event_id=event["event_id"],
                            tel_id=1,
                            true_energy=event["true_energy"].to(u.TeV),
                            true_alt=(90. * u.deg).to(u.rad) - event["true_zd"],
                            true_az=u.Quantity(true_az, u.rad),
                            true_core_x=event["true_core_x"].to(u.m),
                            true_core_y=event["true_core_y"].to(u.m),
                            tel_alt=(90. * u.deg).to(u.rad) - event["true_pointing_zd"],
                            tel_az=u.Quantity(tel_az, u.rad),)
                    event_info[2] = InfoContainerMC(
                            obs_id=obs_id,
                            event_id=event["event_id"],
                            tel_id=2,
                            true_energy=event["true_energy"].to(u.TeV),
                            true_alt=(90. * u.deg).to(u.rad) - event["true_zd"],
                            true_core_x=event["true_core_x"].to(u.m),
                            true_core_y=event["true_core_y"].to(u.m),
                            true_az=u.Quantity(true_az, u.rad),
                            tel_alt=(90. * u.deg).to(u.rad) - event["true_pointing_zd"],
                            tel_az=u.Quantity(tel_az, u.rad),)
                    hillas_params[1] = CameraHillasParametersContainer(
                            x=event["x1"].to(u.m),
                            y=event["y1"].to(u.m),
                            intensity=event["size1"],
                            length=event["length1"].to(u.m),
                            width=event["width1"].to(u.m),
                            psi=event["psi1"].to(u.deg),)
                    hillas_params[2] = CameraHillasParametersContainer(
                            x=event["x2"].to(u.m),
                            y=event["y2"].to(u.m),
                            intensity=event["size2"],
                            length=event["length2"].to(u.m),
                            width=event["width2"].to(u.m),
                            psi=event["psi2"].to(u.deg),)
                    timing_params[1] = CameraTimingParametersContainer(
                            slope=event["slope1"].to(1/u.m))
                    timing_params[2] = CameraTimingParametersContainer(
                            slope=event["slope2"].to(1/u.m))
                    leakage_params[1] = LeakageContainer(
                            intensity_width_1=event["leakage1_1"],
                            intensity_width_2=event["leakage2_1"],)
                    leakage_params[2] = LeakageContainer(
                            intensity_width_1=event["leakage1_2"],
                            intensity_width_2=event["leakage2_2"],)
                    stereo_params = ReconstructedGeometryContainer(
                            alt=(90. * u.deg) - event["reco_zd"],
                            az=event["reco_az"],
                            core_x=event["reco_core_x"].to(u.m),
                            core_y=event["reco_core_y"].to(u.m),
                            tel_ids=[h for h in hillas_params.keys()],
                            average_intensity=np.mean([h.intensity for h in hillas_params.values()]),
                            is_valid=is_valid,
                            h_max=event["hmax"].to(u.m),)
                    for tel_id in list(event_info.keys()):
                        writer.write("hillas_params", (event_info[tel_id], hillas_params[tel_id], leakage_params[tel_id], timing_params[tel_id]))
                    event_info[list(event_info.keys())[0]].tel_id = -1
                    # Storing the result
                    writer.write("stereo_params", (event_info[list(event_info.keys())[0]], stereo_params))
                    id_prev = event["event_id"]

            originalmc_tree = f['OriginalMC']
            originalmc = QTable()

            for column, (branch, kwargs) in columns_mc_orig.items():
                originalmc[column] = u.Quantity(originalmc_tree[branch].array(library="np"), copy=False, **kwargs)

            originalmc = vstack(originalmc)

            shower_data = pd.DataFrame()

            for telescope in [1, 2]:

                true_energy = originalmc[f'true_energy_{telescope}'].to(u.TeV)
                tel_az = originalmc[f'tel_az_{telescope}']
                tel_alt = (90. * u.deg).to(u.rad) - originalmc[f'tel_zd_{telescope}']

                # # Transformation from Monte Carlo to usual azimuth
                # tel_az = -1 * (tel_az - np.pi + np.radians(7))

                cam_x = originalmc[f'cam_x_{telescope}']
                cam_y = originalmc[f'cam_y_{telescope}']

                tel_pointing = AltAz(alt=tel_alt, az=tel_az)

                optics = magic_tel_descriptions[telescope].optics
                camera = magic_tel_descriptions[telescope].camera.geometry

                camera_frame = CameraFrame(focal_length=optics.equivalent_focal_length, rotation=camera.cam_rotation)

                telescope_frame = TelescopeFrame(telescope_pointing=tel_pointing)

                camera_coord = SkyCoord(-cam_y, cam_x, frame=camera_frame)
                shower_coord_in_telescope = camera_coord.transform_to(telescope_frame)

                true_az = shower_coord_in_telescope.altaz.az.to(u.rad)
                true_alt = shower_coord_in_telescope.altaz.alt.to(u.rad)

                evt_id = np.arange(len(tel_az))
                run_id = np.arange(len(tel_az))
                tel_id = np.repeat(telescope, len(tel_az))

                data_ = {
                    'obs_id': run_id,
                    'tel_id': tel_id,
                    'event_id': evt_id,
                    'tel_az': tel_az,
                    'tel_alt': tel_alt,
                    'true_az': true_az,
                    'true_alt': true_alt,
                    'true_energy': true_energy
                }

                df_ = pd.DataFrame(data=data_)
                shower_data = shower_data.append(df_)

            shower_data.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
            shower_data.to_hdf(outfile, key='dl1/original_mc', mode='a')

            obs_id += 1


def write_hdf5_data(filelist):
    """
    Writes an HDF5 file for each superstar file in
    filelist. Specific for real data files.

    Parameters
    ----------
    filelist : list
        A list of files to be opened.
    """

    columns = columns_data

    for path in filelist:
        print(f'Opening {path}')
        with uproot.open(path) as f:

            events_tree = f['Events']
            events = QTable()

            for column, (branch, kwargs) in columns.items():
                events[column] = u.Quantity(events_tree[branch].array(library="np"), copy=False, **kwargs)

            events = vstack(events)
            outfile = str(path).replace(".root", ".h5")
            run_info = get_run_info_from_name(path)
            run_number = run_info[0]

            with HDF5TableWriter(filename=outfile, group_name='dl1', overwrite=True) as writer:
                print(f'Writing in {outfile}')
                event_info = dict()
                hillas_params = dict()
                timing_params = dict()
                leakage_params = dict()
                for event in events:
                    event_mjd = event["mjd1"] + (event["millisec1"] / 1.0e3 + event["nanosec1"] / 1.0e9) / 86400.0
                    if event["is_valid"] > 0:
                        is_valid = True
                    else:
                        is_valid = False
                    event_info[1] = InfoContainerData(
                            obs_id=run_number,
                            event_id=event["event_id"],
                            tel_id=1,
                            mjd=event_mjd.value,
                            tel_alt=(90. * u.deg).to(u.rad) - event["pointing_zen"].to(u.rad),
                            tel_az=event["pointing_az"].to(u.rad),)
                    event_info[2] = InfoContainerData(
                            obs_id=run_number,
                            event_id=event["event_id"],
                            tel_id=2,
                            mjd=event_mjd.value,
                            tel_alt=(90. * u.deg).to(u.rad) - event["pointing_zen"].to(u.rad),
                            tel_az=event["pointing_az"].to(u.rad),)
                    hillas_params[1] = CameraHillasParametersContainer(
                            x=event["x1"].to(u.m),
                            y=event["y1"].to(u.m),
                            intensity=event["size1"],
                            length=event["length1"].to(u.m),
                            width=event["width1"].to(u.m),
                            psi=event["psi1"].to(u.deg),)
                    hillas_params[2] = CameraHillasParametersContainer(
                            x=event["x2"].to(u.m),
                            y=event["y2"].to(u.m),
                            intensity=event["size2"],
                            length=event["length2"].to(u.m),
                            width=event["width2"].to(u.m),
                            psi=event["psi2"].to(u.deg),)
                    timing_params[1] = CameraTimingParametersContainer(
                            slope=event["slope1"].to(1/u.m))
                    timing_params[2] = CameraTimingParametersContainer(
                            slope=event["slope2"].to(1/u.m))
                    leakage_params[1] = LeakageContainer(
                            intensity_width_1=event["leakage1_1"],
                            intensity_width_2=event["leakage2_1"],)
                    leakage_params[2] = LeakageContainer(
                            intensity_width_1=event["leakage1_2"],
                            intensity_width_2=event["leakage2_2"],)
                    stereo_params = ReconstructedGeometryContainer(
                            alt=(90. * u.deg) - event["reco_zd"],
                            az=event["reco_az"],
                            core_x=event["reco_core_x"].to(u.m),
                            core_y=event["reco_core_y"].to(u.m),
                            tel_ids=[h for h in hillas_params.keys()],
                            average_intensity=np.mean([h.intensity for h in hillas_params.values()]),
                            is_valid=is_valid,
                            h_max=event["hmax"].to(u.m),)
                    for tel_id in list(event_info.keys()):
                        writer.write("hillas_params", (event_info[tel_id], hillas_params[tel_id], leakage_params[tel_id], timing_params[tel_id]))
                    event_info[list(event_info.keys())[0]].tel_id = -1
                    # Storing the result
                    writer.write("stereo_params", (event_info[list(event_info.keys())[0]], stereo_params))


def convert_superstar_to_dl1(input_files_mask, is_mc):
    """
    Takes superstar files as input and converts them in HDF5
    format. Real and MC data are treated differently.

    Parameters
    ----------
    input_files_mask : str
        Mask for the superstar input files.
    is_mc : bool
        Flag to tell if real or MC data.
    """

    input_files = Path(input_files_mask)
    filelist = sorted(Path(input_files.parent).expanduser().glob(input_files.name))

    if is_mc:
        write_hdf5_mc(filelist)
    else:
        write_hdf5_data(filelist)


def main(*args):
    flags = parse_args(args)

    is_mc      = flags.use_mc
    input_mask = flags.input_mask

    convert_superstar_to_dl1(input_mask, is_mc)


if __name__ == '__main__':
    main(*sys.argv[1:])
