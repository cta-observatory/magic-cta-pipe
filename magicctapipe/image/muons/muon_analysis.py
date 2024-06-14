"""Script for the muon analysis."""

import numpy as np
from lstchain.image.muon import (
    analyze_muon_event,
    fill_muon_event,
    tag_pix_thr,
    pixel_coords_to_telescope,
)
import astropy.units as u
from iminuit import Minuit

from scipy.stats import poisson

__all__ = [
    "perform_muon_analysis",
]


def MARS_radial_light_distribution(
    center_x, center_y, pixel_x, pixel_y, image, seed_radius, seed_width
):
    """
    Calculate the radial distribution of the muon ring

    Parameters
    ----------
    center_x : `astropy.Quantity`
        Center of muon ring in the field of view from circle fitting
    center_y : `astropy.Quantity`
        Center of muon ring in the field of view from circle fitting
    pixel_x : `ndarray`
        X position of pixels in image
    pixel_y : `ndarray`
        Y position of pixel in image
    image : `ndarray`
        Amplitude of image pixels
    seed_radius: `float`
        Muon ring radius from ctapipe
    seed_width: `float`
        Muon ring width from ctapipe

    Returns
    -------
    standard_dev, skewness

    """

    if np.sum(image) == 0:
        return {
            "standard_dev": np.nan * u.deg,
        }

    # Convert everything to degrees:
    x0 = center_x.to_value(u.deg)
    y0 = center_y.to_value(u.deg)
    pix_x = pixel_x.to_value(u.deg)
    pix_y = pixel_y.to_value(u.deg)

    pix_r = np.sqrt((pix_x - x0) ** 2 + (pix_y - y0) ** 2)
    r_edges = np.linspace(
        seed_radius - 3 * seed_width,
        seed_radius + 3 * seed_width,
        int(6 * seed_width / 0.05) + 1,
    )
    r = (r_edges[1:] + r_edges[:-1]) / 2
    n_pix_hist = np.histogram(pix_r, bins=r_edges)[0]
    width_hist = np.histogram(pix_r, weights=image, bins=r_edges)[0] / n_pix_hist
    inte_seed = np.sum(width_hist) * seed_width
    r0_seed = seed_radius
    standard_dev_seed = seed_width

    def f(inte, r0, mu):
        gauss = inte * np.exp(-((r - r0) ** 2 / mu**2) / 2) / (np.sqrt(2 * np.pi) * mu)
        return -np.sum(poisson._logpmf(width_hist, gauss))

    m = Minuit(f, inte=inte_seed, r0=r0_seed, mu=standard_dev_seed)
    m.errordef = Minuit.LIKELIHOOD
    m.simplex().migrad()
    return {"standard_dev": m.values["mu"] * u.deg}


def perform_muon_analysis(
    muon_parameters,
    event,
    telescope_id,
    telescope_name,
    image,
    subarray,
    r1_dl1_calibrator_for_muon_rings,
    good_ring_config,
    event_time=np.nan,
    min_pe_for_muon_t_calc=10.0,
    data_type="mc",
    plot_rings=False,
    plots_path="./",
):
    """
    Performs the muon analysis.

    Parameters
    ----------
    muon_parameters : dict
        Container for the parameters of all muon rings
    event : ctapipe.containers.ArrayEventContainer
        Event container.
    telescope_id : int
        Id of the telescope
    telescope_name : str
        Name of the telescope
    image : np.ndarray
        Number of photoelectrons in each pixel
    subarray : ctapipe.instrument.subarray.SubarrayDescription
        Subarray
    r1_dl1_calibrator_for_muon_rings : ctapipe.calib.camera.CameraCalibrator
        Camera calibrator.
    good_ring_config : dict
        Set of parameters used to perform the muon ring analysis and select good rings
    event_time : float
        Event time.
    min_pe_for_muon_t_calc : float
        Minimum pixel brightness used to search for the waveform maximum time
    data_type : str
        'obs' or 'mc'
    plot_rings : `bool`
        If True, muon ring plots are produced
    plots_path : str
        Destination of plotted muon rings
    """
    if data_type == "obs":
        try:
            bad_pixels = event.mon.tel[telescope_id].calibration.unusable_pixels[0]
            # Set to 0 unreliable pixels:
            image = image * (~bad_pixels)
        except TypeError:
            pass
    # process only promising events, in terms of # of pixels with large signals:
    thr_low = good_ring_config["thr_low"] if "thr_low" in good_ring_config else 50
    if tag_pix_thr(image, thr_low=thr_low):
        if data_type == "obs":
            try:
                bad_pixels_hg = event.mon.tel[telescope_id].calibration.unusable_pixels[
                    0
                ]
                bad_pixels_lg = event.mon.tel[telescope_id].calibration.unusable_pixels[
                    1
                ]
                bad_pixels = bad_pixels_hg | bad_pixels_lg
                image = image * (~bad_pixels)
            except TypeError:
                pass
        if r1_dl1_calibrator_for_muon_rings is not None:
            # re-calibrate r1 to obtain new dl1, using a more adequate pulse integrator for muon rings
            numsamples = event.r1.tel[telescope_id].waveform.shape[
                1
            ]  # not necessarily the same as in r0!
            if data_type == "obs":
                bad_waveform = np.transpose(np.array(numsamples * [bad_pixels]))
                event.r1.tel[telescope_id].waveform *= ~bad_waveform
            r1_dl1_calibrator_for_muon_rings(event)

        # Check again: with the extractor for muon rings (most likely GlobalPeakWindowSum)
        # perhaps the event is no longer promising (e.g. if it has a large time evolution)
        if not tag_pix_thr(image, thr_low=thr_low):
            good_ring = False
        else:
            event_id = event.index.event_id
            (
                muonintensityparam,
                dist_mask,
                ring_size,
                size_outside_ring,
                muonringparam,
                good_ring,
                radial_distribution,
                mean_pixel_charge_around_ring,
                muonpars,
            ) = analyze_muon_event(
                subarray,
                telescope_id,
                event_id,
                image,
                good_ring_config,
                plot_rings=plot_rings,
                plots_path=plots_path,
            )
            if good_ring and "MAGIC" in telescope_name:
                # Perform MARS-like width extraction

                tel_description = subarray.tels[telescope_id]
                geom = tel_description.camera.geometry
                equivalent_focal_length = tel_description.optics.equivalent_focal_length

                x, y = pixel_coords_to_telescope(geom, equivalent_focal_length)
                MARS_radial_distribution = MARS_radial_light_distribution(
                    muonringparam.center_fov_lon,
                    muonringparam.center_fov_lat,
                    x,
                    y,
                    image,
                    muonringparam.radius.value,
                    radial_distribution["standard_dev"].value,
                )

            if r1_dl1_calibrator_for_muon_rings is not None:
                # Now we want to obtain the waveform sample (in HG & LG) at which the ring light peaks:
                bright_pixels = image > min_pe_for_muon_t_calc
                selected_gain = event.r1.tel[telescope_id].selected_gain_channel
                mask_hg = bright_pixels & (selected_gain == 0)
                mask_lg = bright_pixels & (selected_gain == 1)

                bright_pixels_waveforms_hg = event.r1.tel[telescope_id].waveform[
                    mask_hg, :
                ]
                bright_pixels_waveforms_lg = event.r1.tel[telescope_id].waveform[
                    mask_lg, :
                ]
                stacked_waveforms_hg = np.sum(bright_pixels_waveforms_hg, axis=0)
                stacked_waveforms_lg = np.sum(bright_pixels_waveforms_lg, axis=0)

                # stacked waveforms from all bright pixels; shape (ngains, nsamples)
                hg_peak_sample = np.argmax(stacked_waveforms_hg, axis=-1)
                lg_peak_sample = np.argmax(stacked_waveforms_lg, axis=-1)
            else:
                hg_peak_sample, lg_peak_sample = -1, -1

        mc_energy = event.simulation.shower.energy if data_type == "mc" else -1

        if good_ring:
            fill_muon_event(
                mc_energy,
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
                hg_peak_sample,
                lg_peak_sample,
            )
            muon_parameters["telescope_name"].append(telescope_name)
            if "MAGIC" in telescope_name:
                muon_parameters["MARS_radial_stdev"].append(
                    MARS_radial_distribution["standard_dev"].value
                )
            else:
                muon_parameters["MARS_radial_stdev"].append(np.nan * u.deg)
