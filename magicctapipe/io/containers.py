#!/usr/bin/env python
# coding: utf-8

import numpy as np
from astropy import units as u
from ctapipe.core import Container, Field

__all__ = ["BaseEventInfoContainer", "RealEventInfoContainer", "SimEventInfoContainer"]


class BaseEventInfoContainer(Container):
    """Base container to store event information"""

    obs_id = Field(-1, "Observation ID")
    event_id = Field(-1, "Event ID")
    tel_id = Field(-1, "Telescope ID")
    pointing_alt = Field(np.nan * u.rad, "Altitude of the telescope pointing", u.rad)
    pointing_az = Field(np.nan * u.rad, "Azimuth of the telescope pointing", u.rad)
    n_pixels = Field(-1, "Number of pixels of a cleaned image")
    n_islands = Field(-1, "Number of islands of a cleaned image")


class RealEventInfoContainer(BaseEventInfoContainer):
    """Container to store real event information"""

    time_sec = Field(np.nan * u.s, "Seconds of an event trigger time", u.s)
    time_nanosec = Field(np.nan * u.ns, "Nanoseconds of an event trigger time", u.ns)
    time_diff = Field(np.nan * u.s, "Time difference from the previous event", u.s)


class SimEventInfoContainer(BaseEventInfoContainer):
    """Container to store simulated event information"""

    true_energy = Field(np.nan * u.TeV, "True energy of a simulated event", u.TeV)
    true_alt = Field(np.nan * u.deg, "True altitude of a simulated event", u.deg)
    true_az = Field(np.nan * u.deg, "True azimuth of a simulated event", u.deg)
    true_disp = Field(np.nan * u.deg, "True DISP parameter of a simulated event", u.deg)
    true_core_x = Field(np.nan * u.m, "True core X position of a simulated event", u.m)
    true_core_y = Field(np.nan * u.m, "True core Y position of a simulated event", u.m)
    true_impact = Field(np.nan * u.m, "True impact parameter of a simulated event", u.m)
    off_axis = Field(np.nan * u.deg, "Off-axis angle of a simulated event", u.deg)
    magic_stereo = Field(None, "Boolean where `True` means M1 and M2 are triggered")
