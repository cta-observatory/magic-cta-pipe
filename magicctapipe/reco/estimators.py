#!/usr/bin/env python
# coding: utf-8

import joblib
import logging
import itertools
import numpy as np
import pandas as pd
import sklearn.ensemble
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.coordinates.angle_utilities import angular_separation
from ctapipe.coordinates import CameraFrame, TelescopeFrame

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
    RF regressors to reconstruct the energies of primary particles.
    The RFs are trained per telescope.
    """

    def __init__(self, features=[], settings={}):
        """
        Constructor of the class.

        Parameters
        ----------
        features: list
            Parameters used for training RFs
        settings: dict
            Settings of RF regressors
        """

        self.features = features
        self.settings = settings
        self.telescope_rfs = {}

    def fit(self, event_data):
        """
        Trains RFs per telescope.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Pandas data frame of shower events
        """

        self.telescope_rfs.clear()

        # Train RFs per telescope:
        tel_ids = np.unique(event_data.index.get_level_values('tel_id'))

        for tel_id in tel_ids:

            df_events = event_data.query(f'tel_id == {tel_id}')

            x_train = df_events[self.features].to_numpy()
            y_train = np.log10(df_events['true_energy'].to_numpy())
            weights = df_events['event_weight'].to_numpy()

            regressor = sklearn.ensemble.RandomForestRegressor(**self.settings)

            logger.info(f'Telescope {tel_id}...')
            regressor.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = regressor

    def predict(self, event_data):
        """
        Reconstructs the energies of input events.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Pandas data frame of shower events

        Returns
        -------
        reco_params: pandas.core.frame.DataFrame
            Pandas data frame of reconstructed energies
        """

        reco_params = pd.DataFrame()

        # Apply trained RFs per telescope:
        tel_ids = np.unique(event_data.index.get_level_values('tel_id'))

        for tel_id in tel_ids:

            df_events = event_data.query(f'tel_id == {tel_id}')

            x_predict = df_events[self.features].to_numpy()
            reco_energy = 10 ** self.telescope_rfs[tel_id].predict(x_predict)

            responces_per_estimator = []
            for estimator in self.telescope_rfs[tel_id].estimators_:
                responces_per_estimator.append(estimator.predict(x_predict))

            reco_energy_err = np.std(responces_per_estimator, axis=0)

            df_reco_energy = pd.DataFrame(
                data={'reco_energy': reco_energy,
                      'reco_energy_err': reco_energy_err},
                index=df_events.index,
            )

            reco_params = reco_params.append(df_reco_energy)

        reco_params.sort_index(inplace=True)

        return reco_params

    def save(self, output_file):
        """
        Saves trained RFs in an output joblib file.

        Parameters
        ----------
        output_file: str
            Path to an output joblib file
        """

        output_data = {
            'features': self.features,
            'settings': self.settings,
            'telescope_rfs': self.telescope_rfs,
        }

        joblib.dump(output_data, output_file)

    def load(self, input_file):
        """
        Loads trained RFs from an input joblib file.

        Parameters
        ----------
        input_file: str
            Path to an input joblib file
        """

        input_data = joblib.load(input_file)

        self.features = input_data['features']
        self.settings = input_data['settings']
        self.telescope_rfs = input_data['telescope_rfs']


class DirectionRegressor:
    """
    RF regressors to reconstruct the arrival directions of primary particles.
    The RFs are trained per telescope.
    """

    def __init__(self, features=[], settings={}):
        """
        Constructor of the class.

        Parameters
        ----------
        features: list
            Parameters used for training RFs
        settings: dict
            Settings of RF regressors
        """

        self.features = features
        self.settings = settings
        self.telescope_rfs = {}

    def fit(self, event_data, tel_descriptions):
        """
        Trains RFs per telescope.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Pandas data frame of shower events
        tel_descriptions: dict
            Telescope descriptions
        """

        self.telescope_rfs.clear()
        self.tel_descriptions = tel_descriptions

        # Train RFs per telescope:
        tel_ids = np.unique(event_data.index.get_level_values('tel_id'))

        for tel_id in tel_ids:

            df_events = event_data.query(f'tel_id == {tel_id}')

            x_train = df_events[self.features].to_numpy()
            y_train = df_events['true_disp'].to_numpy()
            weights = df_events['event_weight'].to_numpy()

            regressor = sklearn.ensemble.RandomForestRegressor(**self.settings)

            logger.info(f'Telescope {tel_id}...')
            regressor.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = regressor

    def predict(self, event_data):
        """
        Reconstructs the arrival directions of input events.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Pandas data frame of shower events

        Returns
        -------
        reco_params: pandas.core.frame.DataFrame
            Pandas data frame of reconstructed directions
        """

        reco_params = pd.DataFrame()

        # Apply trained RFs per telescope:
        tel_ids = np.unique(event_data.index.get_level_values('tel_id'))

        for tel_id in tel_ids:

            df_events = event_data.query(f'tel_id == {tel_id}')

            x_predict = df_events[self.features].to_numpy()
            reco_disp = self.telescope_rfs[tel_id].predict(x_predict)

            responces_per_estimator = []
            for estimator in self.telescope_rfs[tel_id].estimators_:
                responces_per_estimator.append(estimator.predict(x_predict))

            reco_disp_err = np.std(responces_per_estimator, axis=0)

            # Reconstruct Alt/Az directions per flip:
            tel_pointing = AltAz(
                alt=u.Quantity(df_events['pointing_alt'].to_numpy(), u.rad),
                az=u.Quantity(df_events['pointing_az'].to_numpy(), u.rad),
            )

            tel_frame = TelescopeFrame(telescope_pointing=tel_pointing)

            camera_frame = CameraFrame(
                focal_length=self.tel_descriptions[tel_id].optics.equivalent_focal_length,
                rotation=self.tel_descriptions[tel_id].camera.geometry.cam_rotation,
            )

            event_coord = SkyCoord(
                u.Quantity(df_events['x'].to_numpy(), u.m),
                u.Quantity(df_events['y'].to_numpy(), u.m),
                frame=camera_frame,
            )

            event_coord = event_coord.transform_to(tel_frame)

            for flip in [0, 1]:

                psi_per_flip = u.Quantity(df_events['psi'].to_numpy(), u.deg) + u.Quantity(180 * flip, u.deg)
                event_coord_per_flip = event_coord.directional_offset_by(psi_per_flip, u.Quantity(reco_disp, u.deg))

                df_per_flip = pd.DataFrame(
                    data={'reco_disp': reco_disp,
                          'reco_disp_err': reco_disp_err,
                          'reco_alt': event_coord_per_flip.altaz.alt.to(u.deg).value,
                          'reco_az': event_coord_per_flip.altaz.az.to(u.deg).value,
                          'flip': np.repeat(flip, len(df_events))},
                    index=df_events.index
                )

                df_per_flip.set_index('flip', append=True, inplace=True)
                reco_params = reco_params.append(df_per_flip)

        reco_params.sort_index(inplace=True)

        # Get the flip combinations minimizing the sum of the angular distances:
        flip_combinations = np.array(list(itertools.product([0, 1], repeat=len(tel_ids))))
        tel_combinations = list(itertools.combinations(tel_ids, 2))

        n_events = len(reco_params.groupby(['obs_id', 'event_id']).size())

        distances = []

        for flip_combo in flip_combinations:

            container = {}

            for tel_id, flip in zip(tel_ids, flip_combo):
                container[tel_id] = reco_params.query(f'(tel_id == {tel_id}) & (flip == {flip})')

            dists_per_combo = np.zeros(n_events)

            for tel_combo in tel_combinations:

                tel_id_1 = tel_combo[0]
                tel_id_2 = tel_combo[1]

                theta = angular_separation(
                    lon1=u.Quantity(container[tel_id_1]['reco_az'].to_numpy(), u.deg),
                    lat1=u.Quantity(container[tel_id_1]['reco_alt'].to_numpy(), u.deg),
                    lon2=u.Quantity(container[tel_id_2]['reco_az'].to_numpy(), u.deg),
                    lat2=u.Quantity(container[tel_id_2]['reco_alt'].to_numpy(), u.deg),
                )

                dists_per_combo += np.array(theta.to(u.deg).value)

            distances.append(dists_per_combo.tolist())

        distances = np.array(distances)
        distances_min = distances.min(axis=0)

        condition = (distances == distances_min)
        indices = np.where(condition.transpose())[1]

        flips = flip_combinations[indices].ravel()

        obs_ids = event_data.index.get_level_values('obs_id')
        event_ids = event_data.index.get_level_values('event_id')
        tel_ids = event_data.index.get_level_values('tel_id')

        multi_indices = pd.MultiIndex.from_arrays([obs_ids, event_ids, tel_ids, flips],
                                                  names=['obs_id', 'event_id', 'tel_id', 'flip'])

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

    def save(self, output_file):
        """
        Saves trained RFs to an output joblib file.

        Parameters
        ----------
        output_file: str
            Path to an output joblib file
        """

        output_data = {
            'features': self.features,
            'settings': self.settings,
            'tel_descriptions': self.tel_descriptions,
            'telescope_rfs': self.telescope_rfs,
        }

        joblib.dump(output_data, output_file)

    def load(self, input_file):
        """
        Loads trained RFs from an input joblib file.

        Parameters
        ----------
        input_file: str
            Path to an input joblib file
        """

        input_data = joblib.load(input_file)

        self.features = input_data['features']
        self.settings = input_data['settings']
        self.tel_descriptions = input_data['tel_descriptions']
        self.telescope_rfs = input_data['telescope_rfs']


class EventClassifier:
    """
    RF classifiers to reconstruct the gammaness.
    The RFs are trained per telescope.
    """

    def __init__(self, features=[], settings={}):
        """
        Constructor of the class.

        Parameters
        ----------
        features: list
            Parameters used for training RFs
        settings: dict
            Settings of RF classifiers
        """

        self.features = features
        self.settings = settings
        self.telescope_rfs = {}

    def fit(self, event_data):
        """
        Trains RFs per telescope.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Pandas data frame of shower events
        """

        self.telescope_rfs.clear()

        # Train RFs per telescope:
        tel_ids = np.unique(event_data.index.get_level_values('tel_id'))

        for tel_id in tel_ids:

            df_events = event_data.query(f'tel_id == {tel_id}')

            x_train = df_events[self.features].to_numpy()
            y_train = df_events['true_event_class'].to_numpy()
            weights = df_events['event_weight'].to_numpy()

            classifier = sklearn.ensemble.RandomForestClassifier(**self.settings)

            logger.info(f'Telescope {tel_id}...')
            classifier.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = classifier

    def predict(self, event_data):
        """
        Reconstructs the gammaness of input events.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Pandas data frame of shower events

        Returns
        -------
        reco_params: pandas.core.frame.DataFrame
            Pandas data frame of reconstructed gammaness
        """

        reco_params = pd.DataFrame()

        # Apply trained RFs per telescope:
        tel_ids = np.unique(event_data.index.get_level_values('tel_id'))

        for tel_id in tel_ids:

            df_events = event_data.query(f'tel_id == {tel_id}')

            x_predict = df_events[self.features].to_numpy()
            responses = self.telescope_rfs[tel_id].predict_proba(x_predict)

            df_reco_class = pd.DataFrame(
                data={'gammaness': responses[:, 0]},
                index=df_events.index,
            )

            reco_params = reco_params.append(df_reco_class)

        reco_params.sort_index(inplace=True)

        return reco_params

    def save(self, output_file):
        """
        Saves trained RFs in an output joblib file.

        Parameters
        ----------
        output_file: str
            Path to an output joblib file
        """

        output_data = {
            'features': self.features,
            'settings': self.settings,
            'telescope_rfs': self.telescope_rfs,
        }

        joblib.dump(output_data, output_file)

    def load(self, input_file):
        """
        Loads trained RFs from an input joblib file.

        Parameters
        ----------
        input_file: str
            Path to an input joblib file
        """

        input_data = joblib.load(input_file)

        self.features = input_data['features']
        self.settings = input_data['settings']
        self.telescope_rfs = input_data['telescope_rfs']


