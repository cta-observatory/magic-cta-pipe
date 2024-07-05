#!/usr/bin/env python
# coding: utf-8

# script to tune NSB level of MAGIC part of sim_telarray MCs
# inspired by calculate_noise_parameters in lstchain/image/modifier.py
# but adapted for the specifics of the MAGIC data calibration

"""
Usage:

$> python tune_magic_nsb.py
--config  config_file.json     (MCP generic config file to take extractor parameters)
--input-mc  simtel_file.simtel.gz    simulation simtel file (for the same period as the data!)
--input-data 2..._Y_.root          real data DL1 file

Calculates the parameters needed to tune the NSB in MC DL1 files
to the level of NSB in a given data file
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from ctapipe.calib.camera import CameraCalibrator
from ctapipe.io import EventSource
from ctapipe_io_magic import MAGICEventSource
from scipy.interpolate import interp1d
from traitlets.config import Config

log = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Tune MAGIC NSB")

# Required arguments
parser.add_argument(
    "--config",
    "-c",
    type=Path,
    help="Path to the configuration file for the production (MCP general config)",
    required=True,
)

parser.add_argument(
    "--input-mc",
    "-m",
    type=Path,
    help="Path to a simtel file of the production (must include the true "
    "p.e. images)",
    required=True,
)

parser.add_argument(
    "--input-data",
    "-d",
    type=Path,
    help="Path to a data _Y_ MARS file",
    required=True,
)

parser.add_argument(
    "--output-file",
    "-o",
    type=Path,
    help="Path to a output file where to dump the update config",
)


def calculate_MAGIC_noise(simtel_filename, magic_filename, config_filename):
    """
    Calculates the parameters needed to increase the noise in an MC DL1 file
    to match the noise in a real data calibrated file, using add_noise_in_pixels
    The returned parameters are those needed by the function add_noise_in_pixels.

    Parameters
    ----------
    simtel_filename : str
        A simtel file containing showers, from the same
        production (same NSB and telescope settings) as the data file below. It
        must contain pixel-wise info on true number of p.e.'s from C-photons (
        will be used to identify pixels which only contain noise).

    magic_filename : str
        A real calibrated data file (processed with calibration settings
        corresponding to those with which the MC is to be processed). This file
        has the "target" noise which we want to have in the MC files, for better
        agreement of data and simulations.

    config_filename : str
        Configuration file containing the calibration
        settings used for processing both the data and the MC files above

    Returns
    -------
    extra_noise_in_dim_pixels: `float`
        Extra noise of dim pixels (number of NSB photoelectrons).
    extra_bias_in_dim_pixels: `float`
        Extra bias of dim pixels  (direct shift in photoelectrons).
    extra_noise_in_bright_pixels: `float`
        Extra noise of bright pixels  (number of NSB photoelectrons).
    """
    event_source = MAGICEventSource(magic_filename, process_run=False)
    event_source.use_pedestals = True

    tel_id = event_source.telescope
    all_peds = []
    for event in event_source:
        bad_pixels = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[0]
        bad_rms = event_source._get_badrmspixel_mask(event)[0]
        bad_pixels = np.logical_or(bad_pixels, bad_rms)
        good_pixels = ~bad_pixels
        if event.count % 100 == 0:
            log.info(
                f"{event.count} events, {event.trigger.event_type}, {sum(bad_pixels)} bad pixels"
            )
            all_peds += list(event.dl1.tel[tel_id].image[good_pixels])

    log.info(f"{len(all_peds)} pedestal entries (pixels x events)")

    qbins = 100
    qrange = (-10, 15)
    dataq = np.histogram(all_peds, bins=qbins, range=qrange, density=True)

    # Find the peak of the pedestal biased charge distribution of real data.
    # Use an interpolated version of the histogram, for robustness:
    func = interp1d(
        0.5 * (dataq[1][1:] + dataq[1][:-1]),
        dataq[0],
        kind="quadratic",
        fill_value="extrapolate",
    )
    xx = np.linspace(qrange[0], qrange[1], 100 * qbins)
    mode_data = xx[np.argmax(func(xx))]
    log.info(
        f"peak (bias) of the biased extractor recomputed from the data as the mode: {mode_data} p.e."
    )

    # 2 is from unbiased extractor and we take the last one
    data_HG_ped_std_pe = event.mon.tel[tel_id].pedestal.charge_std[2][-1][good_pixels]
    data_median_std_ped_pe = np.median(data_HG_ped_std_pe)
    log.info(
        f"standard deviation of unbiased extractor computed in MARS (with Gaussian fits): {data_median_std_ped_pe} p.e."
    )

    # since we start with MAGIC calibrated files, we have only biased extractor applied to pedestals
    # information about the standard deviation of the unbiased extractor is also saved in the data
    # but it comes from a different definition used in MARS (Gaussian fits) than what we will apply to MC
    # therefore we check for biased extractor how much the two numbers differ and apply the same scaling
    # to unbiased extractor
    mars_mean = np.mean(event.mon.tel[tel_id].pedestal.charge_mean[1][-1][good_pixels])
    mcp_mean = np.mean(all_peds)
    mars_std = np.median(event.mon.tel[tel_id].pedestal.charge_std[1][-1][good_pixels])
    mcp_std = np.std(all_peds)

    factor = mcp_std / mars_std
    data_median_std_ped_pe *= factor
    log.info(f"MARS mean: {mars_mean:.3f} p.e., recomputed mean: {mcp_mean:.3f} p.e.")
    log.info(f"MARS std: {mars_std:.3f} p.e., recomputed std: {mcp_std:.3f} p.e.")
    log.info(
        f"correcting std of unbiased extractor by {factor:.3f} to {data_median_std_ped_pe:.3f} p.e."
    )

    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)

    mc_tel_id = config["mc_tel_ids"]["MAGIC-" + tel_id * "I"]
    log.info(
        f"MAGIC {tel_id} expected at the position {mc_tel_id} in sim_telarray file"
    )
    mc_reader = EventSource(
        input_url=simtel_filename, max_events=150, allowed_tels=[mc_tel_id]
    )
    # hardcoded, MARS is using "SlidingWindowMaxSum", and "FixedWindowSum" is its "fixed" version
    pedestal_extractor_type = "FixedWindowSum"

    shower_extractor_type = config["MAGIC"]["image_extractor"].pop("type")
    config_extractor_magic = {shower_extractor_type: config["MAGIC"]["image_extractor"]}
    upscale = config["MAGIC"]["charge_correction"]["factor"]
    window = config["MAGIC"]["image_extractor"]["window_width"]
    ped_config = {
        "FixedWindowSum": {
            "window_shift": window // 2,
            "window_width": window,
            "peak_index": 18,
            "apply_integration_correction": False,
        }
    }
    pedestal_calibrator = CameraCalibrator(
        image_extractor_type=pedestal_extractor_type,
        subarray=mc_reader.subarray,
        config=Config(ped_config),
    )

    shower_calibrator = CameraCalibrator(
        image_extractor_type=shower_extractor_type,
        config=Config(config_extractor_magic),
        subarray=mc_reader.subarray,
    )

    # MC pedestals integrated with the unbiased pedestal extractor
    mc_ped_charges = []
    # MC pedestals integrated with the biased shower extractor
    mc_ped_charges_biased = []

    for event in mc_reader:
        pedestal_calibrator(event)
        charges = event.dl1.tel[mc_tel_id].image * upscale

        # True number of pe's from Cherenkov photons (to identify noise-only pixels)
        true_image = event.simulation.tel[mc_tel_id].true_image
        mc_ped_charges.append(charges[true_image == 0])

        # Now extract the signal as we would do for shower events (usually
        # with a biased extractor):
        shower_calibrator(event)
        charges_biased = event.dl1.tel[mc_tel_id].image * upscale
        mc_ped_charges_biased.append(charges_biased[true_image == 0])

    # All pixels behave (for now) in the same way in MC, just put them together
    mc_ped_charges = np.concatenate(mc_ped_charges)
    mc_ped_charges_biased = np.concatenate(mc_ped_charges_biased)

    mcq = np.histogram(mc_ped_charges_biased, bins=qbins, range=qrange, density=True)

    # Find the peak of the pedestal biased charge distribution of MC. Use
    # an interpolated version of the histogram, for robustness:
    func = interp1d(
        0.5 * (mcq[1][1:] + mcq[1][:-1]),
        mcq[0],
        kind="quadratic",
        fill_value="extrapolate",
    )
    xx = np.linspace(qrange[0], qrange[1], 100 * qbins)
    mode_mc = xx[np.argmax(func(xx))]
    log.info(f"peak (bias) of the biased extractor computed from MC {mode_mc:.3f} p.e.")

    mc_unbiased_std_ped_pe = np.std(mc_ped_charges)
    log.info(
        f"std of the unbiased extractor computed from MC {mc_unbiased_std_ped_pe:.3f} p.e."
    )

    # Find the additional noise (in data w.r.t. MC) for the unbiased extractor.
    # The idea is that when a strong signal is present, the biased extractor
    # will integrate around it, and the additional noise is unbiased because
    # it won't modify the integration range.
    # The noise is defined as the number of NSB photoelectrons, i.e. the extra
    # variance, rather than standard deviation, of the distribution
    extra_noise_in_bright_pixels = (
        data_median_std_ped_pe**2 - mc_unbiased_std_ped_pe**2
    )

    # Just in case, makes sure we just add noise if the MC noise is smaller
    # than the real data's:
    extra_noise_in_bright_pixels = max(0.0, extra_noise_in_bright_pixels)

    bias = mode_data - mode_mc
    extra_bias_in_dim_pixels = max(bias, 0)

    # differences of values to peak charge:
    dq = np.array(all_peds) - mode_data
    dqmc = mc_ped_charges_biased - mode_mc
    # maximum distance (in pe) from peak, to avoid strong impact of outliers:
    maxq = 10
    # calculate widening of the noise bump:
    added_noise = np.sum(dq[dq < maxq] ** 2) / len(dq[dq < maxq]) - np.sum(
        dqmc[dqmc < maxq] ** 2
    ) / len(dqmc[dqmc < maxq])
    extra_noise_in_dim_pixels = max(0.0, added_noise)

    log.info("extra noise in bright, extra bias in dim, extra noise in dim")
    log.info(
        f"{extra_noise_in_bright_pixels:.3f}, {extra_bias_in_dim_pixels:.3f}, {extra_noise_in_dim_pixels:.3f}"
    )

    return (
        extra_noise_in_dim_pixels,
        extra_bias_in_dim_pixels,
        extra_noise_in_bright_pixels,
    )


def main():
    """Main function for getting the MAGIC added NSB noise parameters."""
    args = parser.parse_args()

    if not args.config.is_file():
        log.error("Config file does not exist or is not a file")
        sys.exit(1)
    if not args.input_mc.is_file():
        log.error("MC simtel file does not exist or is not a file")
        sys.exit(1)
    if not args.input_data.is_file():
        log.error("DL1 data file does not exist or is not a file")
        sys.exit(1)

    log.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logging.getLogger().addHandler(handler)

    a, b, c = calculate_MAGIC_noise(args.input_mc, args.input_data, args.config)
    if a is None:
        logging.error("Could not compute NSB tuning parameters. Exiting!")
        sys.exit(1)

    dict_nsb = {
        "use": True,
        "extra_noise_in_dim_pixels": round(a, 3),
        "extra_bias_in_dim_pixels": round(b, 3),
        "transition_charge": 8,
        "extra_noise_in_bright_pixels": round(c, 3),
    }

    log.info("\n")
    log.info(json.dumps(dict_nsb, indent=2))
    log.info("\n")

    if args.output_file:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

        if "increase_nsb" in cfg["MAGIC"]:
            cfg["MAGIC"]["increase_nsb"].update(dict_nsb)
        else:
            cfg["MAGIC"]["increase_nsb"] = dict_nsb

        with open(args.output_file, "w") as out_file:
            out_file.write(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
