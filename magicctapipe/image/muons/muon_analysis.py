import numpy as np
from lstchain.image.muon import tag_pix_thr, fill_muon_event

__all__ = [
    'perform_muon_analysis',
]

def perform_muon_analysis(muon_parameters, event, telescope_id, telescope_name, image, subarray,
                          r1_dl1_calibrator_for_muon_rings, good_ring_config, event_time=np.nan,
                          min_pe_for_muon_t_calc=10., data_type='mc'):
    if data_type == 'obs':
        bad_pixels = event.mon.tel[telescope_id].calibration.unusable_pixels[0]
        # Set to 0 unreliable pixels:
        image = image * (~bad_pixels)
    # process only promising events, in terms of # of pixels with large signals:
    thr_low = good_ring_config['thr_low'] if 'thr_low' in good_ring_config else 50
    if tag_pix_thr(image, thr_low=thr_low):
        # re-calibrate r1 to obtain new dl1, using a more adequate pulse integrator for muon rings
        numsamples = event.r1.tel[telescope_id].waveform.shape[1]  # not necessarily the same as in r0!
        if data_type == 'obs':
            bad_pixels_hg = event.mon.tel[telescope_id].calibration.unusable_pixels[0]
            bad_pixels_lg = event.mon.tel[telescope_id].calibration.unusable_pixels[1]
            # Now set to 0 all samples in unreliable pixels. Important for global peak
            # integrator in case of crazy pixels!  TBD: can this be done in a simpler
            # way?
            bad_pixels = bad_pixels_hg | bad_pixels_lg
            bad_waveform = np.transpose(np.array(numsamples * [bad_pixels]))

            # print('hg bad pixels:',np.where(bad_pixels_hg))
            # print('lg bad pixels:',np.where(bad_pixels_lg))

            event.r1.tel[telescope_id].waveform *= ~bad_waveform
            image = image * (~bad_pixels)
        r1_dl1_calibrator_for_muon_rings(event)

        focal_length = subarray.tel[telescope_id].optics.equivalent_focal_length

        # Check again: with the extractor for muon rings (most likely GlobalPeakWindowSum)
        # perhaps the event is no longer promising (e.g. if it has a large time evolution)
        if not tag_pix_thr(image, thr_low=thr_low):
            good_ring = False
        else:
            # read geometry from event.inst. But not needed for every event. FIXME?
            geom = subarray.tel[telescope_id]. \
                camera.geometry
            mirror_area = subarray.tel[telescope_id].optics.mirror_area
            event_id = event.index.event_id
            muonintensityparam, dist_mask, \
            ring_size, size_outside_ring, muonringparam, \
            good_ring, radial_distribution, \
            mean_pixel_charge_around_ring, \
            muonpars = \
                analyze_muon_event(subarray, telescope_id, event_id,
                                   image, good_ring_config,
            #                       plot_rings=False, plots_path='')
                                  plot_rings=True, plots_path='../../../../data/'+telescope_name+'/')
            #           (test) plot muon rings as png files

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

        if good_ring:
            fill_muon_event(-1,
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
            muon_parameters['telescope_name'].append(telescope_name)


# TODO replace this copy of dev version of lstchain (last release tag v0.9.4) by an import once v0.9.5+ is out
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from ctapipe.containers import (
    MuonEfficiencyContainer,
    MuonParametersContainer,
)
from ctapipe.coordinates import (
    CameraFrame,
    TelescopeFrame,
)
from ctapipe.image.muon import (
    MuonIntensityFitter
)
from ctapipe.image.muon.features import (
    ring_completeness,
    ring_containment,
)
from lstchain.image.muon.muon_analysis import (
    fit_muon,
    radial_light_distribution
)
from lstchain.image.muon import pixel_coords_to_telescope, plot_muon_event


def update_parameters(config, n_pixels):
    """
    Create the parameters used to select good muon rings and perform the muon analysis.
    Parameters
    ----------
    config: `dict` or None
        Subset of parameters to be updated
    n_pixels: `int`
        Number of pixels of the camera
    Returns
    -------
    params: `dict`
        Dictionary of parameters used for the muon analysis
    """
    params = {
        'tailcuts': [10, 5],  # Thresholds used for the tail_cut cleaning
        'min_pix': 0.08,  # minimum fraction of the number of pixels in the ring with >0 signal
        'min_pix_fraction_after_cleaning': 0.1,  # minimum fraction of the ring pixels that must be above tailcuts[0]
        'min_ring_radius': 0.8 * u.deg,  # minimum ring radius
        'max_ring_radius': 1.5 * u.deg,  # maximum ring radius
        'max_radial_stdev': 0.1 * u.deg,  # maximum standard deviation of the light distribution along ring radius
        'max_radial_excess_kurtosis': 1.,  # maximum excess kurtosis
        'min_impact_parameter': 0.2,  # in fraction of mirror radius
        'max_impact_parameter': 0.9,  # in fraction of mirror radius
        'ring_integration_width': 0.25,  # +/- integration range along ring radius,
                                         # in fraction of ring radius (was 0.4 until 20200326)
        'outer_ring_width': 0.2,  # in fraction of ring radius, width of ring just outside the
                                  # integrated muon ring, used to check pedestal bias
        'ring_completeness_threshold': 30,  # Threshold in p.e. for pixels used in the ring completeness estimation
    }
    if config is not None:
        for key in config.keys():
            params[key] = config[key]
    params['min_pix'] = int(n_pixels * params['min_pix'])

    return params


def analyze_muon_event(subarray, tel_id, event_id, image, good_ring_config, plot_rings, plots_path):
    """
    Analyze an event to fit a muon ring
    Parameters
    ----------
    subarray: `ctapipe.instrument.subarray.SubarrayDescription`
        Telescopes subarray
    tel_id : `int`
        Id of the telescope used
    event_id : `int`
        Id of the analyzed event
    image : `np.ndarray`
        Number of photoelectrons in each pixel
    good_ring_config : `dict` or None
        Set of parameters used to identify good muon rings to update LST-1 defaults
    plot_rings : `bool`
        Plot the muon ring
    plots_path : `string`
        Path to store the figures
    Returns
    -------
    muonintensityoutput : `MuonEfficiencyContainer`
    dist_mask : `ndarray`
        Pixels used in ring intensity likelihood fit
    ring_size : `float`
        Total intensity in ring in photoelectrons
    size_outside_ring : `float`
        Intensity outside the muon ting in photoelectrons
        to check for "shower contamination"
    muonringparam : `MuonParametersContainer`
    good_ring : `bool`
        It determines whether the ring can be used for analysis or not
    radial_distribution : `dict`
        Return of function radial_light_distribution
    mean_pixel_charge_around_ring : float
        Charge "just outside" ring, to check the possible signal extractor bias
    muonparameters : `MuonParametersContainer`
    """

    tel_description = subarray.tels[tel_id]

    cam_rad = (
                      tel_description.camera.geometry.guess_radius() / tel_description.optics.equivalent_focal_length
              ) * u.rad
    geom = tel_description.camera.geometry
    equivalent_focal_length = tel_description.optics.equivalent_focal_length
    mirror_area = tel_description.optics.mirror_area

    # some parameters for analysis and cuts for good ring selection:
    params = update_parameters(good_ring_config, geom.n_pixels)

    x, y = pixel_coords_to_telescope(geom, equivalent_focal_length)
    muonringparam, clean_mask, dist, image_clean = fit_muon(x, y, image, geom,
                                                            params['tailcuts'])

    mirror_radius = np.sqrt(mirror_area / np.pi)  # meters
    dist_mask = np.abs(dist - muonringparam.radius
                       ) < muonringparam.radius * params['ring_integration_width']
    pix_ring = image * dist_mask
    pix_outside_ring = image * ~dist_mask

    # mask to select pixels just outside the ring that will be integrated to obtain the ring's intensity:
    dist_mask_2 = np.logical_and(~dist_mask,
                                 np.abs(dist - muonringparam.radius) <
                                 muonringparam.radius *
                                 (params['ring_integration_width'] + params['outer_ring_width']))
    pix_ring_2 = image[dist_mask_2]

    #    nom_dist = np.sqrt(np.power(muonringparam.center_x,2)
    #                    + np.power(muonringparam.center_y, 2))

    muonparameters = MuonParametersContainer()
    muonparameters.containment = ring_containment(
        muonringparam.radius,
        muonringparam.center_x, muonringparam.center_y, cam_rad)

    radial_distribution = radial_light_distribution(
        muonringparam.center_x,
        muonringparam.center_y,
        x[clean_mask], y[clean_mask],
        image[clean_mask])

    # Do complicated calculations (minuit-based max likelihood ring fit) only for selected rings:
    candidate_clean_ring = all(
        [radial_distribution['standard_dev'] < params['max_radial_stdev'],
         radial_distribution['excess_kurtosis'] < params['max_radial_excess_kurtosis'],
         (pix_ring > params['tailcuts'][0]).sum() >
         params['min_pix_fraction_after_cleaning'] * params['min_pix'],
         np.count_nonzero(pix_ring) > params['min_pix'],
         muonringparam.radius < params['max_ring_radius'],
         muonringparam.radius > params['min_ring_radius']
         ])

    if candidate_clean_ring:
        intensity_fitter = MuonIntensityFitter(subarray, hole_radius_m=0.308)

        # Use same hard-coded value for pedestal fluctuations as the previous
        # version of ctapipe:
        pedestal_stddev = 1.1 * np.ones(len(image))

        muonintensityoutput = \
            intensity_fitter(tel_id,
                             muonringparam.center_x,
                             muonringparam.center_y,
                             muonringparam.radius,
                             image,
                             pedestal_stddev,
                             dist_mask)

        dist_ringwidth_mask = np.abs(dist - muonringparam.radius) < \
                              muonintensityoutput.width

        # We do the calculation of the ring completeness (i.e. fraction of whole circle) using the pixels
        # within the "width" fitted using MuonIntensityFitter
        muonparameters.completeness = ring_completeness(
            x[dist_ringwidth_mask], y[dist_ringwidth_mask],
            image[dist_ringwidth_mask],
            muonringparam.radius,
            muonringparam.center_x,
            muonringparam.center_y,
            threshold=params['ring_completeness_threshold'],
            bins=30)

        # No longer existing in ctapipe 0.8:
        # pix_ringwidth_im = image[dist_ringwidth_mask]
        # muonintensityoutput.ring_pix_completeness =  \
        #     (pix_ringwidth_im > tailcuts[0]).sum() / len(pix_ringwidth_im)

    else:
        # just to have the default values with units:
        muonintensityoutput = MuonEfficiencyContainer()
        muonintensityoutput.width = u.Quantity(np.nan, u.deg)
        muonintensityoutput.impact = u.Quantity(np.nan, u.m)
        muonintensityoutput.impact_x = u.Quantity(np.nan, u.m)
        muonintensityoutput.impact_y = u.Quantity(np.nan, u.m)

    # muonintensityoutput.mask = dist_mask # no longer there in ctapipe 0.8
    ring_size = np.sum(pix_ring)
    size_outside_ring = np.sum(pix_outside_ring * clean_mask)

    # This is just mean charge per pixel in pixels just around the ring
    # (on the outer side):
    mean_pixel_charge_around_ring = np.sum(pix_ring_2) / len(pix_ring_2)

    if candidate_clean_ring:
        print("Impact parameter={:.3f}, ring_width={:.3f}, ring radius={:.3f}, "
              "ring completeness={:.3f}".format(
            muonintensityoutput.impact,
            muonintensityoutput.width,
            muonringparam.radius,
            muonparameters.completeness, ))
    # Now add the conditions based on the detailed muon ring fit:
    conditions = [
        candidate_clean_ring,
        muonintensityoutput.impact < params['max_impact_parameter'] * mirror_radius,
        muonintensityoutput.impact > params['min_impact_parameter'] * mirror_radius,

        # TODO: To be applied when we have decent optics.
        # muonintensityoutput.width
        # < 0.08,
        # NOTE: inside "candidate_clean_ring" cuts there is already a cut in
        # the std dev of light distribution along ring radius, which is also
        # a measure of the ring width

        # muonintensityoutput.width
        # > 0.04
    ]

    if all(conditions):
        good_ring = True
    else:
        good_ring = False

    if (plot_rings and plots_path and good_ring):
        focal_length = equivalent_focal_length
        ring_telescope = SkyCoord(muonringparam.center_x,
                                  muonringparam.center_y,
                                  TelescopeFrame())

        ring_camcoord = ring_telescope.transform_to(CameraFrame(
            focal_length=focal_length,
            rotation=geom.cam_rotation,
        ))
        centroid = (ring_camcoord.x.value, ring_camcoord.y.value)
        radius = muonringparam.radius
        width = muonintensityoutput.width
        ringrad_camcoord = 2 * radius.to(u.rad) * focal_length
        ringwidthfrac = width / radius
        ringrad_inner = ringrad_camcoord * (1. - ringwidthfrac)
        ringrad_outer = ringrad_camcoord * (1. + ringwidthfrac)

        fig, ax = plt.subplots(figsize=(10, 10))
        plot_muon_event(ax, geom, image * clean_mask, centroid,
                        ringrad_camcoord, ringrad_inner, ringrad_outer,
                        event_id)

        plt.figtext(0.15, 0.20, 'radial std dev: {0:.3f}'. \
                    format(radial_distribution['standard_dev']))
        plt.figtext(0.15, 0.18, 'radial excess kurtosis: {0:.3f}'. \
                    format(radial_distribution['excess_kurtosis']))
        plt.figtext(0.15, 0.16, 'fitted ring width: {0:.3f}'.format(width))
        plt.figtext(0.15, 0.14, 'ring completeness: {0:.3f}'. \
                    format(muonparameters.completeness))

        fig.savefig('{}/Event_{}_fitted.png'.format(plots_path, event_id))

    if (plot_rings and not plots_path):
        print("You are trying to plot without giving a path!")

    return muonintensityoutput, dist_mask, ring_size, size_outside_ring, \
           muonringparam, good_ring, radial_distribution, \
           mean_pixel_charge_around_ring, muonparameters
