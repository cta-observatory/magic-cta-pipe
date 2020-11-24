import numpy as np

def check_write_stereo(event, tel_id, computed_hillas_params, hillas_reco,
                        subarray, array_pointing, telescope_pointings,
                        event_info, writer):
    """Check hillas parameters and write stero parameters

    Parameters
    ---------- 
    event : ctapipe.containers.EventAndMonDataContainer
        event
    tel_id : int
        telescope id
    computed_hillas_params : dict
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
    if len(computed_hillas_params.keys()) > 1:
        err_str = ("Event ID %d  (obs ID: %d) has an ellipse with width = %s: "
                   "stereo parameters calculation skipped.")
        if any([computed_hillas_params[tel_id]["width"].value == 0
                for tel_id in computed_hillas_params]):
            print(err_str % (vent.index.event_id, event.index.obs_id, '0'))
        elif any([np.isnan(computed_hillas_params[tel_id]["width"].value) for
                  tel_id in computed_hillas_params]):
            print(err_str % (vent.index.event_id, event.index.obs_id, 'NaN'))
        else:
            # Reconstruct stereo event. From ctapipe
            stereo_params = hillas_reco.predict(
                hillas_dict=computed_hillas_params,
                subarray=subarray,
                array_pointing=array_pointing,
                telescopes_pointings=telescope_pointings
            )
            event_info.tel_id = -1
            # Storing the result
            writer.write("stereo_params", (event_info, stereo_params))
    return stereo_params
