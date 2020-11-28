import os
import datetime
import pandas as pd
from astropy import units as u

import ctapipe
from ctapipe.instrument import CameraGeometry
from ctapipe.instrument import TelescopeDescription
from ctapipe.instrument import OpticsDescription
from ctapipe.instrument import SubarrayDescription


def get_tel_descriptions(name, cam, tel_ids):
    """Get telescopes description

    Parameters
    ----------
    name : str
        telescope name
    cam : str
        camera name
    tel_ids : list
        telescope ids

    Returns
    -------
    dict
        tel_descriptions
    """
    optics = OpticsDescription.from_name(name)
    cam = CameraGeometry.from_name(cam)
    tel_description = TelescopeDescription(name=name,
                                           tel_type=name,
                                           optics=optics,
                                           camera=cam)
    tel_descriptions = {}
    for tel_id in tel_ids:
        tel_descriptions = {
            **tel_descriptions,
            **{tel_id: tel_description}
        }
    return tel_descriptions


def get_tel_ids(df):
    """Return telescope ids from loaded dl1 pandas dataframe

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        pandas dataframe

    Returns
    -------
    list
        telescope ids
    """
    return list(df.index.levels[2])


def convert_positions_dict(positions_dict):
    """Convert telescope positions loaded from config.yaml file from 
    adimensional numbers to u.m (astropy units)

    Parameters
    ----------
    positions_dict : dict
        telescopes positions

    Returns
    -------
    dict
        telescopes positions
    """
    for k_ in positions_dict.keys():
        positions_dict[k_] *= u.m
    return positions_dict
