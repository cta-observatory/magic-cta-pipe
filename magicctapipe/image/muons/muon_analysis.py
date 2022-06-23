import numpy as np
from lstchain.image.muon import tag_pix_thr, fill_muon_event, analyze_muon_event

__all__ = [
    'perform_muon_analysis',
]


def perform_muon_analysis(muon_parameters, event, telescope_id, image, subarray,
                          r1_dl1_calibrator_for_muon_rings, good_ring_config, event_time=np.nan,
                          min_pe_for_muon_t_calc=10., data_type='mc'):
    """

    Parameters
    ----------
    muon_parameters: dict
        Container for the parameters of all muon rings
    event: ctapipe event container
    telescope_id: int
        Id of the telescope
    image:  `np.ndarray`
        Number of photoelectrons in each pixel
    subarray: `ctapipe.instrument.subarray.SubarrayDescription`
    r1_dl1_calibrator_for_muon_rings: `ctapipe.calib.camera.CameraCalibrator`
    good_ring_config: dict
        Set of parameters used to perform the muon ring analysis and select good rings
    event_time: float
    min_pe_for_muon_t_calc: float
        Minimum pixel brightness used to search for the waveform maximum time
    data_type: string
        'obs' or 'mc'

    """
    if data_type == 'obs':
        try:
            bad_pixels = event.mon.tel[telescope_id].calibration.unusable_pixels[0]
            # Set to 0 unreliable pixels:
            image = image * (~bad_pixels)
        except TypeError:
            pass
    # process only promising events, in terms of # of pixels with large signals:
    thr_low = good_ring_config['thr_low'] if 'thr_low' in good_ring_config else 50
    if tag_pix_thr(image, thr_low=thr_low):
        if data_type == 'obs':
            try:
                bad_pixels_hg = event.mon.tel[telescope_id].calibration.unusable_pixels[0]
                bad_pixels_lg = event.mon.tel[telescope_id].calibration.unusable_pixels[1]
                bad_pixels = bad_pixels_hg | bad_pixels_lg
                image = image * (~bad_pixels)
            except TypeError:
                pass
        if r1_dl1_calibrator_for_muon_rings is not None:
            # re-calibrate r1 to obtain new dl1, using a more adequate pulse integrator for muon rings
            numsamples = event.r1.tel[telescope_id].waveform.shape[1]  # not necessarily the same as in r0!
            if data_type == 'obs':
                bad_waveform = np.transpose(np.array(numsamples * [bad_pixels]))
                event.r1.tel[telescope_id].waveform *= ~bad_waveform
            r1_dl1_calibrator_for_muon_rings(event)

        # Check again: with the extractor for muon rings (most likely GlobalPeakWindowSum)
        # perhaps the event is no longer promising (e.g. if it has a large time evolution)
        if not tag_pix_thr(image, thr_low=thr_low):
            good_ring = False
        else:
            event_id = event.index.event_id
            muonintensityparam, dist_mask, \
            ring_size, size_outside_ring, muonringparam, \
            good_ring, radial_distribution, \
            mean_pixel_charge_around_ring, \
            muonpars = \
                analyze_muon_event(subarray, telescope_id, event_id,
                                   image, good_ring_config,
                                   plot_rings=False, plots_path='')
                                  #plot_rings=True, plots_path='../data/real'+telescope_name+'/')
            #           (test) plot muon rings as png files

            if r1_dl1_calibrator_for_muon_rings is not None:
                # Now we want to obtain the waveform sample (in HG & LG) at which the ring light peaks:
                bright_pixels = image > min_pe_for_muon_t_calc
                selected_gain = event.r1.tel[telescope_id].selected_gain_channel
                mask_hg = bright_pixels & (selected_gain == 0)
                mask_lg = bright_pixels & (selected_gain == 1)

                bright_pixels_waveforms_hg = event.r1.tel[telescope_id].waveform[mask_hg, :]
                bright_pixels_waveforms_lg = event.r1.tel[telescope_id].waveform[mask_lg, :]
                stacked_waveforms_hg = np.sum(bright_pixels_waveforms_hg, axis=0)
                stacked_waveforms_lg = np.sum(bright_pixels_waveforms_lg, axis=0)

                # stacked waveforms from all bright pixels; shape (ngains, nsamples)
                hg_peak_sample = np.argmax(stacked_waveforms_hg, axis=-1)
                lg_peak_sample = np.argmax(stacked_waveforms_lg, axis=-1)
            else:
                hg_peak_sample, lg_peak_sample = -1, -1

        mc_energy = event.simulation.shower.energy if data_type == 'mc' else -1

        if good_ring:
            fill_muon_event(mc_energy,
                            muon_parameters,
                            good_ring,
                            event.index.event_id,
                            event_time,
                            muonintensityparam,
                            dist_mask,
                            muonringparam,
                            radial_distribution,
                            ring_size,
                            size_outside_ring,
                            mean_pixel_charge_around_ring,
                            muonpars,
                            hg_peak_sample, lg_peak_sample)
