import numpy as np
import astropy.units as u

from ctapipe.core.container import Container, Field


def write_hillas(writer, event_info, hillas_p, leakage_p, timing_p):
    """Write

    Parameters
    ----------
    writer : ctapipe.io.hdf5tableio.HDF5TableWriter
        HDF5TableWriter object
    event_info : magicctapipe.reco.stereo.StereoInfoContainer
        event info, StereoInfoContainer object
    hillas_p : ctapipe.containers.HillasParametersContainer
        hillas parameters
    leakage_p : ctapipe.containers.HillasParametersContainer
        leakage parameters
    timing_p : ctapipe.containers.TimingParametersContainer
        timing parameters
    """
    writer.write("hillas_params", (event_info, hillas_p, leakage_p, timing_p))


def check_write_stereo(event, tel_id, hillas_p, hillas_reco, subarray,
                       array_pointing, telescope_pointings, event_info, writer):
    """Check hillas parameters and write stero parameters

    Parameters
    ----------
    event : ctapipe.containers.EventAndMonDataContainer
        event
    tel_id : int
        telescope id
    hillas_p : dict
        computed hillas parameters
    hillas_reco : ctapipe.reco.HillasReconstructor.HillasReconstructor
        HillasReconstructor
    subarray : ctapipe.instrument.subarray.SubarrayDescription
        source.subarray
    array_pointing : astropy.coordinates.sky_coordinate.SkyCoord
        array_pointing
    telescope_pointings : dict
        telescope_pointings
    event_info : __main__.StereoInfoContainer
        StereoInfoContainer object
    writer : ctapipe.io.hdf5tableio.HDF5TableWriter
        HDF5TableWriter object

    Returns
    -------
    ctapipe.containers.ReconstructedShowerContainer
        stereo_params
    """
    if len(hillas_p.keys()) > 1:
        err_str = ("Event ID %d  (obs ID: %d) has an ellipse with width = %s: "
                   "stereo parameters calculation skipped.")
        stereo_params = None
        if any([hillas_p[tel_id]["width"].value == 0 for tel_id in hillas_p]):
            print(err_str % (event.index.event_id, event.index.obs_id, '0'))
        elif any([np.isnan(hillas_p[tel_id]["width"].value) for
                  tel_id in hillas_p]):
            print(err_str % (event.index.event_id, event.index.obs_id, 'NaN'))
        else:
            # Reconstruct stereo event. From ctapipe
            stereo_params = hillas_reco.predict(
                hillas_dict=hillas_p,
                subarray=subarray,
                array_pointing=array_pointing,
                telescopes_pointings=telescope_pointings
            )
            event_info.tel_id = -1
            stereo_params.tel_ids = \
                sum([2**a_ for a_ in stereo_params.tel_ids])
            # How to go back
            # n = stereo_params.tel_ids
            # np.where(np.array(list(bin(n)[2:][::-1]))=='1')[0]
            writer.write("stereo_params", (event_info, stereo_params))
    return stereo_params


class StereoInfoContainer(Container):
    """"InfoContainer fro stereo"""
    obs_id = Field(-1, "Observation ID")
    event_id = Field(-1, "Event ID")
    tel_id = Field(-1, "Telescope ID")
    true_energy = Field(-1, "MC event energy", unit=u.TeV)
    true_alt = Field(-1, "MC event altitude", unit=u.rad)
    true_az = Field(-1, "MC event azimuth", unit=u.rad)
    tel_alt = Field(-1, "MC telescope altitude", unit=u.rad)
    tel_az = Field(-1, "MC telescope azimuth", unit=u.rad)
    num_islands = Field(-1, "Number of image islands")
