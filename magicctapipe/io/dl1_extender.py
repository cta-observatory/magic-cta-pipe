from astropy.coordinates import AltAz, SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
from ctapipe.coordinates import CameraFrame
from ctapipe.core import Component
from ctapipe.core.traits import Bool
from ctapipe.instrument import SubarrayDescription
import pandas as pd
import numpy as np

class Dl1Extender(Component):
    add_log_intensity = Bool(False, help='If True, a the log intensity.',
                             allow_none=False).tag(config=True)
    add_zd_indep_log_intensity = Bool(False, help='If True, a new parameter representing the log intensity corrected '
                                                  'for the dependence on the telescope poining zenith is added.',
                                      allow_none=False).tag(config=True)
    add_zd_indep_intensity = Bool(False, help='If True, a new parameter representing the intensity corrected '
                                               'for the dependence on the telescope poining zenith is added.',
                                  allow_none=False).tag(config=True)
    add_altaz_cog = Bool(False, help='If True, a new parameter representing the AltAz position of the shower center'
                                     'of gravity is added.',
                         allow_none=False).tag(config=True)
    # TODO update when ctapipe version increases
    # focal_length_choice = CaselessStrEnum()

    def __init__(self, config=None, parent=None, **kwargs):
        super().__init__(config=config, parent=parent, **kwargs)
        self.slope = 0

    def dl1_zd_indep_intensity(self, df):
        if 'log_intensity' in df.keys():
            log_intensity = df['log_intensity']
        else:
            log_intensity = np.log(df['intensity'])
            if self.add_log_intensity:
                df['log_intensity'] = log_intensity
        zd_indep_log_intensity = log_intensity - self.slope*(90 - np.rad2deg(df['pointing_alt']))
        if self.add_zd_indep_log_intensity:
            df['zd_indep_log_intensity'] = zd_indep_log_intensity
        if self.add_zd_indep_intensity:
            df['zd_indep_intensity'] = np.power(zd_indep_log_intensity, 10)

    def dl1_altaz_cog(self, df, subarray):
        df['cog_alt'] = np.zeros(len(df))
        df['cog_az'] = np.zeros(len(df))
        for tel in subarray.tel:
            location = EarthLocation.from_geodetic(-17.89139 * u.deg, 28.76139 * u.deg, 2184 * u.m)
            obstime = Time("2018-11-01T02:00")
            horizon_frame = AltAz(location=location, obstime=obstime)
            mask_tel = df['tel_id'] == tel
            pointing_direction = SkyCoord(
                alt=df['pointing_alt']*u.rad,
                az=df['pointing_az']*u.rad, frame=horizon_frame
            )
            # TODO update when ctapipe version increases
            camera_frame = CameraFrame(
                focal_length=subarray.tel[tel].optics.equivalent_focal_length,
                telescope_pointing=pointing_direction
            )
            camera_coord = SkyCoord(df['x']*u.m, df['y']*u.m, frame=camera_frame)
            horizon = camera_coord.transform_to(horizon_frame)
            df['cog_alt'] += horizon.alt.rad * mask_tel
            df['cog_az'] += horizon.az.rad * mask_tel

    def __call__(self, file, base_reader=lambda x: pd.read_hdf(x, key="events/parameters")):
        # TODO update when ctapipe version increases
        full_subarray = SubarrayDescription.from_hdf(
            file
        )
        df = base_reader(file)
        if self.add_altaz_cog:
            self.dl1_altaz_cog(df, full_subarray)
        if self.add_zd_indep_intensity or self.add_zd_indep_log_intensity or self.add_log_intensity:
            self.dl1_zd_indep_intensity(df)
        return df
