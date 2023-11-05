"""
Custom containers
"""
import numpy as np
from astropy import units as u
from ctapipe.core import Container, Field

__all__ = ["BaseEventInfoContainer", "RealEventInfoContainer", "SimEventInfoContainer"]

DEFAULT_VALUE = -1
DEFAULT_RAD = np.nan * u.rad
DEFAULT_DEG = np.nan * u.deg
DEFAULT_TEV = np.nan * u.TeV
DEFAULT_M = np.nan * u.m
DEFAULT_S = np.nan * u.s
DEFAULT_NS = np.nan * u.ns


class BaseEventInfoContainer(Container):
    """Base container to store event information"""

    obs_id = Field(DEFAULT_VALUE, "Observation ID")
    event_id = Field(DEFAULT_VALUE, "Event ID")
    tel_id = Field(DEFAULT_VALUE, "Telescope ID")
    pointing_alt = Field(DEFAULT_RAD, "Altitude of the pointing direction", unit="rad")
    pointing_az = Field(DEFAULT_RAD, "Azimuth of the pointing direction", unit="rad")
    n_pixels = Field(DEFAULT_VALUE, "Number of pixels after the image cleaning")
    n_islands = Field(DEFAULT_VALUE, "Number of islands after the image cleaning")


class RealEventInfoContainer(BaseEventInfoContainer):
    """Container to store real event information"""

    time_sec = Field(DEFAULT_S, "Seconds of the event trigger time", unit="s")
    time_nanosec = Field(DEFAULT_NS, "Nanoseconds of the event trigger time", unit="ns")
    time_diff = Field(DEFAULT_S, "Time difference from the previous event", unit="s")


class SimEventInfoContainer(BaseEventInfoContainer):
    """Container to store simulated event information"""

    true_energy = Field(DEFAULT_TEV, "True energy", unit="TeV")
    true_alt = Field(DEFAULT_DEG, "True altitude", unit="deg")
    true_az = Field(DEFAULT_DEG, "True azimuth", unit="deg")
    true_disp = Field(DEFAULT_DEG, "True DISP parameter", unit="deg")
    true_core_x = Field(DEFAULT_M, "True core X position", unit="m")
    true_core_y = Field(DEFAULT_M, "True core Y position", unit="m")
    true_impact = Field(DEFAULT_M, "True impact parameter", unit="m")
    off_axis = Field(DEFAULT_DEG, "Off-axis angle", unit="deg")
    magic_stereo = Field(None, "Whether both M1 and M2 are triggered or not")
