#!/usr/bin/env python
# coding: utf-8

import joblib
import logging
import itertools
import numpy as np
import pandas as pd
import sklearn.ensemble
from astropy import units as u
from astropy.coordinates import (
    AltAz,
    SkyCoord,
    angular_separation,
)
from ctapipe.coordinates import (
    CameraFrame,
    TelescopeFrame,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

__all__ = [
    'EnergyRegressor',
    'DirectionRegressor',
    'EventClassifier',
]


class EnergyRegressor:
    """
    RF regressors to reconstruct the energy of primary particles.
    The RFs are trained per telescope.
    """

    def __init__(self, features=[], rf_settings={}):
        """
        Constructor of the class.
        Set settings for training the RFs.

        Parameters
        ----------
        features: list
            Parameters used for training the RFs
        rf_settings: dict
            Settings for the RF regressors
        """

        self.features = features
        self.rf_settings = rf_settings
        self.telescope_rfs = {}

    def fit(self, input_data):
        """
        Train the RFs per telescope.

        Parameters
        ----------
        input_data: pandas.core.frame.DataFrame
            Pandas data frame containing training samples
        """

        self.telescope_rfs.clear()

        param_names = self.features + ['event_weight', 'true_energy']
        telescope_ids = np.unique(input_data.index.get_level_values('tel_id'))

        # Train the RFs per telescope:
        for tel_id in telescope_ids:

            df_tel = input_data.loc[(slice(None), slice(None), tel_id), param_names]

            x_train = df_tel[self.features].values
            y_train = np.log10(df_tel['true_energy'].values)
            weights = df_tel['event_weight'].values

            regressor = sklearn.ensemble.RandomForestRegressor(**self.rf_settings)

            logger.info(f'Telescope {tel_id}...')
            regressor.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = regressor

    def predict(self, input_data):
        """
        Reconstruct the energy of input events.

        Parameters
        ----------
        input_data: pandas.core.frame.DataFrame
            Pandas data frame containing shower events
        """

        reco_params = pd.DataFrame()
        telescope_ids = np.unique(input_data.index.get_level_values('tel_id'))

        # Apply the RFs per telescope:
        for tel_id in telescope_ids:

            df_tel = input_data.loc[(slice(None), slice(None), tel_id), self.features]

            reco_energy = 10 ** self.telescope_rfs[tel_id].predict(df_tel.values)

            responces_per_estimator = []
            for estimator in self.telescope_rfs[tel_id].estimators_:
                responces_per_estimator.append(10 ** estimator.predict(df_tel.values))

            reco_energy_err = np.std(responces_per_estimator, axis=0)

            df_reco_energy = pd.DataFrame(
                data={'reco_energy': reco_energy, 'reco_energy_err': reco_energy_err},
                index=df_tel.index,
            )

            reco_params = reco_params.append(df_reco_energy)

        reco_params.sort_index(inplace=True)

        return reco_params

    def save(self, file_name):
        """
        Save the settings and the trained RFs to a joblib file.

        Parameters
        ----------
        file_name: str
            Path to an output joblib file
        """

        output_data = {}

        output_data['features'] = self.features
        output_data['rf_settings'] = self.rf_settings
        output_data['telescope_rfs'] = self.telescope_rfs

        joblib.dump(output_data, file_name)

    def load(self, file_name):
        """
        Load settings and trained RFs from a joblib file.

        Parameters
        ----------
        file_name: str
            Path to an input joblib file
        """

        input_data = joblib.load(file_name)

        self.features = input_data['features']
        self.rf_settings = input_data['rf_settings']
        self.telescope_rfs = input_data['telescope_rfs']


class DirectionRegressor:
    """
    RF regressors to reconstruct the arrival directions of primary particles.
    The RFs are trained per telescope.
    """

    def __init__(self, features=[], rf_settings={}, tel_descriptions={}):
        """
        Constructor of the class.
        Set settings for training the RFs.

        Parameters
        ----------
        features: list
            Parameters used for training the RFs
        rf_settings: dict
            Settings for the RF regressors
        tel_descriptions: dict
            Telescope descriptions
        """

        self.features = features
        self.rf_settings = rf_settings
        self.tel_descriptions = tel_descriptions
        self.telescope_rfs = {}

    def fit(self, input_data):
        """
        Train the RFs per telescope.

        Parameters
        ----------
        input_data: pandas.core.frame.DataFrame
            Pandas data frame containing training samples
        """

        self.telescope_rfs.clear()

        param_names = self.features + ['event_weight', 'true_disp']
        telescope_ids = np.unique(input_data.index.get_level_values('tel_id'))

        # Train the RFs per telescope:
        for tel_id in telescope_ids:

            df_tel = input_data.loc[(slice(None), slice(None), tel_id), param_names]

            x_train = df_tel[self.features].values
            y_train = df_tel['true_disp'].values
            weights = df_tel['event_weight'].values

            regressor = sklearn.ensemble.RandomForestRegressor(**self.rf_settings)

            logger.info(f'Telescope {tel_id}...')
            regressor.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = regressor

    def predict(self, input_data):
        """
        Reconstruct the arrival directions of input events.

        Parameters
        ----------
        input_data: pandas.core.frame.DataFrame
            Pandas data frame containing shower events
        """

        reco_params = pd.DataFrame()
        param_names = self.features + ['pointing_alt', 'pointing_az', 'x', 'y', 'psi']

        telescope_ids = np.unique(input_data.index.get_level_values('tel_id'))

        # Apply the RFs per telescope:
        for tel_id in telescope_ids:

            df_tel = input_data.loc[(slice(None), slice(None), tel_id), param_names]

            reco_disp = self.telescope_rfs[tel_id].predict(df_tel[self.features].values)

            responces_per_estimator = []
            for estimator in self.telescope_rfs[tel_id].estimators_:
                responces_per_estimator.append(estimator.predict(df_tel[self.features].values))

            reco_disp_err = np.std(responces_per_estimator, axis=0)

            # Reconstruct the Alt/Az directions per flip:
            tel_pointing = AltAz(
                alt=u.Quantity(df_tel['pointing_alt'].values, u.rad),
                az=u.Quantity(df_tel['pointing_az'].values, u.rad),
            )

            telescope_frame = TelescopeFrame(telescope_pointing=tel_pointing)

            camera_frame = CameraFrame(
                focal_length=self.tel_descriptions[tel_id].optics.equivalent_focal_length,
                rotation=self.tel_descriptions[tel_id].camera.geometry.cam_rotation,
            )

            event_coord = SkyCoord(
                u.Quantity(df_tel['x'].values, u.m),
                u.Quantity(df_tel['y'].values, u.m),
                frame=camera_frame,
            )

            event_coord = event_coord.transform_to(telescope_frame)

            for flip in [0, 1]:

                psi_per_flip = u.Quantity(df_tel['psi'].values, u.deg) + u.Quantity(180 * flip, u.deg)
                event_coord_per_flip = event_coord.directional_offset_by(psi_per_flip, u.Quantity(reco_disp, u.deg))

                df_per_flip = pd.DataFrame(
                    data={'reco_disp': reco_disp,
                          'reco_disp_err': reco_disp_err,
                          'reco_alt': event_coord_per_flip.altaz.alt.to(u.deg).value,
                          'reco_az': event_coord_per_flip.altaz.az.to(u.deg).value,
                          'flip': np.repeat(flip, len(df_tel))},
                    index=df_tel.index
                )

                df_per_flip.set_index('flip', append=True, inplace=True)
                reco_params = reco_params.append(df_per_flip)

        reco_params.sort_index(inplace=True)

        # Get the flip combinations minimizing the sum of the angular distances:
        flip_combinations = np.array(list(itertools.product([0, 1], repeat=len(telescope_ids))))
        telescope_combinations = list(itertools.combinations(telescope_ids, 2))

        n_events = len(reco_params.groupby(['obs_id', 'event_id']).size())

        distances = []

        for flip_combo in flip_combinations:

            container = {}

            for tel_id, flip in zip(telescope_ids, flip_combo):
                container[tel_id] = reco_params.query(f'(tel_id == {tel_id}) & (flip == {flip})')

            dists_per_combo = np.zeros(n_events)

            for tel_combo in telescope_combinations:

                tel_id_1 = tel_combo[0]
                tel_id_2 = tel_combo[1]

                theta = angular_separation(
                    lon1=u.Quantity(container[tel_id_1]['reco_az'].values, u.deg),
                    lat1=u.Quantity(container[tel_id_1]['reco_alt'].values, u.deg),
                    lon2=u.Quantity(container[tel_id_2]['reco_az'].values, u.deg),
                    lat2=u.Quantity(container[tel_id_2]['reco_alt'].values, u.deg),
                )

                dists_per_combo += np.array(theta.to(u.deg).value)

            distances.append(dists_per_combo.tolist())

        distances = np.array(distances)
        distances_min = distances.min(axis=0)

        condition = (distances == distances_min)
        indices = np.where(condition.transpose())[1]

        flips = flip_combinations[indices].ravel()

        obs_ids = input_data.index.get_level_values('obs_id')
        event_ids = input_data.index.get_level_values('event_id')
        tel_ids = input_data.index.get_level_values('tel_id')

        multi_indices = pd.MultiIndex.from_arrays(
            [obs_ids, event_ids, tel_ids, flips], names=['obs_id', 'event_id', 'tel_id', 'flip'],
        )

        reco_params = reco_params.loc[multi_indices]
        reco_params.reset_index(level='flip', inplace=True)
        reco_params.drop('flip', axis=1, inplace=True)
        reco_params.sort_index(inplace=True)

        df_dist = pd.DataFrame(
            data={'dist_sum': distances_min},
            index=reco_params.groupby(['obs_id', 'event_id']).size().index,
        )

        reco_params = reco_params.join(df_dist)

        return reco_params

    def save(self, file_name):
        """
        Save the settings and the trained RFs to a joblib file.

        Parameters
        ----------
        file_name: str
            Path to an output joblib file
        """

        output_data = {}

        output_data['features'] = self.features
        output_data['rf_settings'] = self.rf_settings
        output_data['tel_descriptions'] = self.tel_descriptions
        output_data['telescope_rfs'] = self.telescope_rfs

        joblib.dump(output_data, file_name)

    def load(self, file_name):
        """
        Load settings and trained RFs from a joblib file.

        Parameters
        ----------
        file_name: str
            Path to an input joblib file
        """

        input_data = joblib.load(file_name)

        self.features = input_data['features']
        self.rf_settings = input_data['rf_settings']
        self.tel_descriptions = input_data['tel_descriptions']
        self.telescope_rfs = input_data['telescope_rfs']


class EventClassifier:
    """
    RF classifiers to reconstruct the gammaness.
    The RFs are trained per telescope.
    """

    def __init__(self, features=[], rf_settings={}):
        """
        Constructor of the class.
        Set settings for training the RFs.

        Parameters
        ----------
        features: list
            Parameters used for training the RFs
        rf_settings: dict
            Settings for the RF regressors
        """

        self.features = features
        self.rf_settings = rf_settings
        self.telescope_rfs = {}

    def fit(self, input_data):
        """
        Train the RFs per telescope.

        Parameters
        ----------
        input_data: pandas.core.frame.DataFrame
            Pandas data frame containing training samples
        """

        self.telescope_rfs.clear()

        param_names = self.features + ['event_weight', 'true_event_class']
        telescope_ids = np.unique(input_data.index.get_level_values('tel_id'))

        # Train the RFs per telescope:
        for tel_id in telescope_ids:

            df_tel = input_data.loc[(slice(None), slice(None), tel_id), param_names]

            x_train = df_tel[self.features].values
            y_train = df_tel['true_event_class'].values
            weights = df_tel['event_weight'].values

            classifier = sklearn.ensemble.RandomForestClassifier(**self.rf_settings)

            logger.info(f'Telescope {tel_id}...')
            classifier.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = classifier

    def predict(self, input_data):
        """
        Reconstruct the gammaness of input events.

        Parameters
        ----------
        input_data: pandas.core.frame.DataFrame
            Pandas data frame containing shower events
        """
        reco_params = pd.DataFrame()

        # Apply the RFs per telescope:
        telescope_ids = np.unique(input_data.index.get_level_values('tel_id'))

        for tel_id in telescope_ids:

            df_tel = input_data.loc[(slice(None), slice(None), tel_id), self.features]
            responses = self.telescope_rfs[tel_id].predict_proba(df_tel.values)

            df_reco_class = pd.DataFrame({'gammaness': responses[:, 0]}, index=df_tel.index)
            reco_params = reco_params.append(df_reco_class)

        reco_params.sort_index(inplace=True)

        return reco_params

    def save(self, file_name):
        """
        Save the settings and the trained RFs to a joblib file.

        Parameters
        ----------
        file_name: str
            Path to an output joblib file
        """

        output_data = {}

        output_data['features'] = self.features
        output_data['rf_settings'] = self.rf_settings
        output_data['telescope_rfs'] = self.telescope_rfs

        joblib.dump(output_data, file_name)

    def load(self, file_name):
        """
        Load settings and trained RFs from a joblib file.

        Parameters
        ----------
        file_name: str
            Path to an input joblib file
        """

        input_data = joblib.load(file_name)

        self.features = input_data['features']
        self.rf_settings = input_data['rf_settings']
        self.telescope_rfs = input_data['telescope_rfs']

