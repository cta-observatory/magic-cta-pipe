#!/usr/bin/env python
# coding: utf-8

import logging

import joblib
import numpy as np
import pandas as pd
import sklearn.ensemble
from magicctapipe.io import TEL_NAMES

__all__ = ["EnergyRegressor", "DispRegressor", "EventClassifier"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class EnergyRegressor:
    """
    RF regressors to reconstruct the energies of primary particles.

    Attributes
    ----------
    settings: dict
        Settings of RF regressors
    features: list
        Parameters for training RFs
    use_unsigned_features: bool
        If `True`, it trains RFs with unsigned features
    telescope_rfs: dict
        Telescope-wise RFs
    """

    def __init__(self, settings={}, features=[], use_unsigned_features=None):
        """
        Constructor of the class.

        Parameters
        ----------
        settings: dict
            Settings of RF regressors
        features: list
            Parameters for training RFs
        use_unsigned_features: bool
            If `True`, it trains RFs with unsigned features
        """

        self.settings = settings
        self.features = features
        self.use_unsigned_features = use_unsigned_features
        self.telescope_rfs = {}

    def fit(self, event_data):
        """
        Trains RFs per telescope.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Data frame of shower events
        """

        self.telescope_rfs.clear()

        # Train RFs per telescope
        tel_ids = np.unique(event_data["tel_id"])

        for tel_id in tel_ids:

            df_events = event_data.query(f"tel_id == {tel_id}")
            df_events.dropna(subset=self.features, inplace=True)

            if self.use_unsigned_features:
                x_train = np.abs(df_events[self.features].to_numpy())
            else:
                x_train = df_events[self.features].to_numpy()

            # Use logarithmic energy for the target values
            y_train = np.log10(df_events["true_energy"].to_numpy())
            weights = df_events["event_weight"].to_numpy()

            regressor = sklearn.ensemble.RandomForestRegressor(**self.settings)

            logger.info(f"Training {TEL_NAMES[tel_id]} RF...")
            regressor.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = regressor

    def predict(self, event_data):
        """
        Reconstructs the energies of primary particles with trained RFs.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Data frame of shower events

        Returns
        -------
        reco_params: pandas.core.frame.DataFrame
            Data frame of the shower events with reconstructed energies
        """

        reco_params = pd.DataFrame()

        # Apply trained RFs per telescope
        for tel_id, telescope_rf in self.telescope_rfs.items():

            df_events = event_data.query(f"tel_id == {tel_id}")
            df_events.dropna(subset=self.features, inplace=True)

            if len(df_events) > 0:

                if self.use_unsigned_features:
                    x_predict = np.abs(df_events[self.features].to_numpy())
                else:
                    x_predict = df_events[self.features].to_numpy()

                reco_energy = 10 ** telescope_rf.predict(x_predict)

                responses_per_estimator = []
                for estimator in telescope_rf.estimators_:
                    responses_per_estimator.append(estimator.predict(x_predict))

                reco_energy_var = np.var(responses_per_estimator, axis=0)

            else:
                reco_energy = []
                reco_energy_var = []

            df_reco_energy = pd.DataFrame(
                data={"reco_energy": reco_energy, "reco_energy_var": reco_energy_var},
                index=df_events.index,
            )

            reco_params = reco_params.append(df_reco_energy)

        reco_params.sort_index(inplace=True)

        return reco_params

    def save(self, output_file):
        """
        Saves trained RFs in a joblib file.

        Parameters
        ----------
        output_file: str
            Path to an output joblib file
        """

        output_data = {
            "settings": self.settings,
            "features": self.features,
            "use_unsigned_features": self.use_unsigned_features,
            "telescope_rfs": self.telescope_rfs,
        }

        joblib.dump(output_data, output_file)

    def load(self, input_file):
        """
        Loads trained RFs from a joblib file.

        Parameters
        ----------
        input_file: str
            Path to an input joblib file
        """

        input_data = joblib.load(input_file)

        self.settings = input_data["settings"]
        self.features = input_data["features"]
        self.use_unsigned_features = input_data["use_unsigned_features"]
        self.telescope_rfs = input_data["telescope_rfs"]


class DispRegressor:
    """
    RF regressors to reconstruct the DISP parameter.

    Attributes
    ----------
    settings: dict
        Settings of RF regressors
    features: list
        Parameters for training RFs
    use_unsigned_features: bool
        If `True`, it trains RFs with unsigned features
    telescope_rfs: dict
        Telescope-wise RFs
    """

    def __init__(self, settings={}, features=[], use_unsigned_features=None):
        """
        Constructor of the class.

        Parameters
        ----------
        settings: dict
            Settings of RF regressors
        features: list
            Parameters for training RFs
        use_unsigned_features: bool
            If `True`, it trains RFs with unsigned features
        """

        self.settings = settings
        self.features = features
        self.use_unsigned_features = use_unsigned_features
        self.telescope_rfs = {}

    def fit(self, event_data):
        """
        Trains RFs per telescope.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Data frame of shower events
        """

        self.telescope_rfs.clear()

        # Train RFs per telescope
        tel_ids = np.unique(event_data["tel_id"])

        for tel_id in tel_ids:

            df_events = event_data.query(f"tel_id == {tel_id}")
            df_events.dropna(subset=self.features, inplace=True)

            if self.use_unsigned_features:
                x_train = np.abs(df_events[self.features].to_numpy())
            else:
                x_train = df_events[self.features].to_numpy()

            y_train = df_events["true_disp"].to_numpy()
            weights = df_events["event_weight"].to_numpy()

            regressor = sklearn.ensemble.RandomForestRegressor(**self.settings)

            logger.info(f"Training {TEL_NAMES[tel_id]} RF...")
            regressor.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = regressor

    def predict(self, event_data):
        """
        Reconstructs the DISP parameter with trained RFs.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Data frame of shower events

        Returns
        -------
        reco_params: pandas.core.frame.DataFrame
            Data frame of the shower events with the DISP parameter
        """

        reco_params = pd.DataFrame()

        # Apply trained RFs per telescope
        for tel_id, telescope_rf in self.telescope_rfs.items():

            df_events = event_data.query(f"tel_id == {tel_id}")
            df_events.dropna(subset=self.features, inplace=True)

            if len(df_events) > 0:

                if self.use_unsigned_features:
                    x_predict = np.abs(df_events[self.features].to_numpy())
                else:
                    x_predict = df_events[self.features].to_numpy()

                reco_disp = telescope_rf.predict(x_predict)

                responses_per_estimator = []
                for estimator in telescope_rf.estimators_:
                    responses_per_estimator.append(estimator.predict(x_predict))

                reco_disp_var = np.var(responses_per_estimator, axis=0)

            else:
                reco_disp = []
                reco_disp_var = []

            df_reco_disp = pd.DataFrame(
                data={"reco_disp": reco_disp, "reco_disp_var": reco_disp_var},
                index=df_events.index,
            )

            reco_params = reco_params.append(df_reco_disp)

        reco_params.sort_index(inplace=True)

        return reco_params

    def save(self, output_file):
        """
        Saves trained RFs to a joblib file.

        Parameters
        ----------
        output_file: str
            Path to an output joblib file
        """

        output_data = {
            "settings": self.settings,
            "features": self.features,
            "use_unsigned_features": self.use_unsigned_features,
            "telescope_rfs": self.telescope_rfs,
        }

        joblib.dump(output_data, output_file)

    def load(self, input_file):
        """
        Loads trained RFs from a joblib file.

        Parameters
        ----------
        input_file: str
            Path to an input joblib file
        """

        input_data = joblib.load(input_file)

        self.settings = input_data["settings"]
        self.features = input_data["features"]
        self.use_unsigned_features = input_data["use_unsigned_features"]
        self.telescope_rfs = input_data["telescope_rfs"]


class EventClassifier:
    """
    RF classifiers to reconstruct the gammaness.

    Attributes
    ----------
    settings: dict
        Settings of RF classifiers
    features: list
        Parameters for training RFs
    use_unsigned_features: bool
        If `True`, it trains RFs with unsigned features
    telescope_rfs: dict
        Telescope-wise RFs
    """

    def __init__(self, settings={}, features=[], use_unsigned_features=None):
        """
        Constructor of the class.

        Parameters
        ----------
        settings: dict
            Settings of RF classifiers
        features: list
            Parameters for training RFs
        use_unsigned_features: bool
            If `True`, it trains RFs with unsigned features
        """

        self.settings = settings
        self.features = features
        self.use_unsigned_features = use_unsigned_features
        self.telescope_rfs = {}

    def fit(self, event_data):
        """
        Trains RFs per telescope.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Data frame of shower events
        """

        self.telescope_rfs.clear()

        # Train RFs per telescope
        tel_ids = np.unique(event_data["tel_id"])

        for tel_id in tel_ids:

            df_events = event_data.query(f"tel_id == {tel_id}")
            df_events.dropna(subset=self.features, inplace=True)

            if self.use_unsigned_features:
                x_train = np.abs(df_events[self.features].to_numpy())
            else:
                x_train = df_events[self.features].to_numpy()

            y_train = df_events["true_event_class"].to_numpy()
            weights = df_events["event_weight"].to_numpy()

            classifier = sklearn.ensemble.RandomForestClassifier(**self.settings)

            logger.info(f"Training {TEL_NAMES[tel_id]} RF...")
            classifier.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = classifier

    def predict(self, event_data):
        """
        Reconstructs the gammaness.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Data frame of shower events

        Returns
        -------
        reco_params: pandas.core.frame.DataFrame
            Data frame of the shower events with the gammaness
        """

        reco_params = pd.DataFrame()

        # Apply trained RFs per telescope
        for tel_id, telescope_rf in self.telescope_rfs.items():

            df_events = event_data.query(f"tel_id == {tel_id}")
            df_events.dropna(subset=self.features, inplace=True)

            if len(df_events) > 0:

                if self.use_unsigned_features:
                    x_predict = np.abs(df_events[self.features].to_numpy())
                else:
                    x_predict = df_events[self.features].to_numpy()

                gammaness = telescope_rf.predict_proba(x_predict)[:, 0]

                # Calculate the variance of the binomial distribution
                gammaness_var = gammaness * (1 - gammaness)

                # Set the artificial finite value in case the variance is 0
                # to avoid that the inverse value, which may be used for the
                # weights of averaging telescope-wise values, is infinite
                gammaness_var[gammaness == 1] = 0.99 * (1 - 0.99)
                gammaness_var[gammaness == 0] = 0.01 * (1 - 0.01)

            else:
                gammaness = []
                gammaness_var = []

            df_gammaness = pd.DataFrame(
                data={"gammaness": gammaness, "gammaness_var": gammaness_var},
                index=df_events.index,
            )

            reco_params = reco_params.append(df_gammaness)

        reco_params.sort_index(inplace=True)

        return reco_params

    def save(self, output_file):
        """
        Saves trained RFs in a joblib file.

        Parameters
        ----------
        output_file: str
            Path to an output joblib file
        """

        output_data = {
            "settings": self.settings,
            "features": self.features,
            "use_unsigned_features": self.use_unsigned_features,
            "telescope_rfs": self.telescope_rfs,
        }

        joblib.dump(output_data, output_file)

    def load(self, input_file):
        """
        Loads trained RFs from a joblib file.

        Parameters
        ----------
        input_file: str
            Path to an input joblib file
        """

        input_data = joblib.load(input_file)

        self.settings = input_data["settings"]
        self.features = input_data["features"]
        self.use_unsigned_features = input_data["use_unsigned_features"]
        self.telescope_rfs = input_data["telescope_rfs"]
