from math import isclose

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

from magicctapipe import __version__
from magicctapipe.io.gadf import (
    create_event_hdu,
    create_gh_cuts_hdu,
    create_gti_hdu,
    create_pointing_hdu,
)


@pytest.fixture(scope="session")
def gammaness_cut():
    return np.array([0.7, 0.8, 0.9])


@pytest.fixture(scope="session")
def energy_bins():
    return np.logspace(-1, 2, 4) * u.TeV


@pytest.fixture(scope="session")
def fov_bins():
    return np.array([0, 1]) * u.deg


@pytest.fixture(scope="session")
def header():
    head = {
        "TELESCOP": "CTA-N",
        "INSTRUME": "LST-1_MAGIC",
    }
    return head


@pytest.fixture(scope="session")
def event():
    evt = QTable()
    evt["timestamp"] = np.array([1.608070e9, 1.608071e9, 1.608072e9]) * u.s
    evt["pointing_ra"] = np.array([84.0, 84.1, 83.9]) * u.deg
    evt["pointing_dec"] = np.repeat(22.246, (3)) * u.deg
    evt["pointing_alt"] = np.array([0.82143, 0.82144, 0.82143]) * u.rad
    evt["pointing_az"] = np.array([1.52638, 1.52639, 1.52638]) * u.rad
    evt["obs_id"] = np.repeat(3267, (3))
    evt["event_id"] = np.array([1, 2, 3])
    evt["reco_ra"] = np.array([84.2, 83.8, 84.0]) * u.deg
    evt["reco_dec"] = np.array([22.25, 22.27, 22.22]) * u.deg
    evt["reco_alt"] = np.array([0.82142, 0.82143, 0.82144]) * u.rad
    evt["reco_az"] = np.array([1.52637, 1.52640, 1.52639]) * u.rad
    evt["reco_energy"] = np.array([0.97, 1.2, 2.3]) * u.TeV
    evt["gammaness"] = np.array([0.79, 0.85, 0.93])
    evt["multiplicity"] = np.array([2, 3, 3])
    evt["combo_type"] = np.array([1, 3, 3])
    return evt


def test_create_gh_cuts_hdu(gammaness_cut, energy_bins, fov_bins, header):
    g_cuts_fits = create_gh_cuts_hdu(
        gammaness_cut, energy_bins, fov_bins, True, **header
    )
    g_cuts = g_cuts_fits.data
    assert np.allclose(g_cuts["ENERG_LO"][0], np.array([0.1, 1.0, 10.0]))
    assert np.allclose(g_cuts["ENERG_HI"][0], np.array([1.0, 10.0, 100.0]))
    assert g_cuts["THETA_LO"] == 0
    assert np.allclose(g_cuts["GH_CUTS"][0], np.array([0.7, 0.8, 0.9]))
    g_head = g_cuts_fits.header
    assert g_head["TELESCOP"] == "CTA-N"
    assert g_head["INSTRUME"] == "LST-1_MAGIC"
    assert g_head["CREATOR"] == f"magicctapipe v{__version__}"


def test_create_pointing_hdu(event):
    point_fits = create_pointing_hdu(event)
    point = point_fits.data
    assert point["TIME"][0] == 1.608070e9
    assert point["DEC_PNT"][0] == 22.246
    assert isclose(
        point["ALT_PNT"][0], event["pointing_alt"][0].to("deg").value, abs_tol=1e-8
    )
    point_head = point_fits.header
    assert point_head["OBS_ID"] == 3267
    assert point_head["TIMEUNIT"] == "s"


def test_create_gti_hdu(event):
    gti_fits = create_gti_hdu(event)
    gti = gti_fits.data
    assert gti["START"] == 1.608070e9
    assert gti["STOP"] == 1.608072e9
    gti_head = gti_fits.header
    gti_head["OBS_ID"] == 3267
    assert gti_head["TIMESYS"] == "UTC"


def test_create_event_hdu(event):
    evt_fits = create_event_hdu(event, 200 * u.s, 0.97, "Crab")
    evt = evt_fits.data
    assert np.array_equal(evt["EVENT_ID"], np.array([1, 2, 3]))
    assert np.array_equal(evt["RA"], np.array([84.2, 83.8, 84.0]))
    assert np.array_equal(evt["GAMMANESS"], np.array([0.79, 0.85, 0.93]))
    assert np.allclose(evt["ALT"], event["reco_alt"].to("deg").value)
    evt_head = evt_fits.header
    assert evt_head["OBS_ID"] == 3267
    assert evt_head["TIMEREF"] == "TOPOCENTER"
    assert evt_head["N_TELS"] == 3
    assert isclose(evt_head["LIVETIME"], 200 * 0.97, abs_tol=1e-8)
    assert evt_head["OBJECT"] == "Crab"
    assert evt_head["RA_PNT"] == 84.0
    assert isclose(
        evt_head["AZ_PNT"], event["pointing_az"][0].to("deg").value, abs_tol=1e-8
    )


def test_create_event_hdu_exc(event):
    with pytest.raises(
        ValueError,
        match="The input RA/Dec coordinate is set to `None`.",
    ):
        _ = create_event_hdu(event, 200 * u.s, 0.97, "abc")
