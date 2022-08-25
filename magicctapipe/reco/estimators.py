#!/usr/bin/env python
# coding: utf-8

import itertools
import logging

import joblib
import numpy as np
import pandas as pd
import sklearn.ensemble
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord, angular_separation
from ctapipe.coordinates import TelescopeFrame

__all__ = ["EnergyRegressor", "DirectionRegressor", "EventClassifier"]

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class EnergyRegressor:
    """
    RF regressors to reconstruct the energies of primary particles.
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
            Pandas data frame of shower events
        """

        self.telescope_rfs.clear()

        # Train RFs per telescope
        tel_ids = np.unique(event_data.index.get_level_values("tel_id"))

        for tel_id in tel_ids:

            df_events = event_data.query(f"tel_id == {tel_id}")

            if self.use_unsigned_features:
                x_train = np.abs(df_events[self.features].to_numpy())
            else:
                x_train = df_events[self.features].to_numpy()

            y_train = np.log10(df_events["true_energy"].to_numpy())
            weights = df_events["event_weight"].to_numpy()

            regressor = sklearn.ensemble.RandomForestRegressor(**self.settings)

            logger.info(f"Telescope {tel_id}...")
            regressor.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = regressor

    def predict(self, event_data):
        """
        Reconstructs the energies with trained RFs.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Pandas data frame of shower events

        Returns
        -------
        reco_params: pandas.core.frame.DataFrame
            Pandas data frame of the reconstructed energies
        """

        reco_params = pd.DataFrame()

        # Apply trained RFs per telescope
        tel_ids = np.unique(event_data.index.get_level_values("tel_id"))

        for tel_id in tel_ids:

            df_events = event_data.query(f"tel_id == {tel_id}").copy()

            if self.use_unsigned_features:
                x_predict = np.abs(df_events[self.features].to_numpy())
            else:
                x_predict = df_events[self.features].to_numpy()

            reco_energy = 10 ** self.telescope_rfs[tel_id].predict(x_predict)

            responses_per_estimator = []
            for estimator in self.telescope_rfs[tel_id].estimators_:
                responses_per_estimator.append(estimator.predict(x_predict))

            reco_energy_var = np.var(responses_per_estimator, axis=0)

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


class DirectionRegressor:
    """
    RF regressors to reconstruct the arrival directions of primary particles.
    """

    def __init__(
        self, settings={}, features=[], tel_descriptions={}, use_unsigned_features=None
    ):
        """
        Constructor of the class.

        Parameters
        ----------
        settings: dict
            Settings of RF regressors
        features: list
            Parameters for training RFs
        tel_descriptions: dict
            Telescope descriptions
        use_unsigned_features: bool
            If `True`, it trains RFs with unsigned features
        """

        self.settings = settings
        self.features = features
        self.tel_descriptions = tel_descriptions
        self.use_unsigned_features = use_unsigned_features
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

        # Train RFs per telescope
        tel_ids = np.unique(event_data.index.get_level_values("tel_id"))

        for tel_id in tel_ids:

            df_events = event_data.query(f"tel_id == {tel_id}")

            if self.use_unsigned_features:
                x_train = np.abs(df_events[self.features].to_numpy())
            else:
                x_train = df_events[self.features].to_numpy()

            y_train = df_events["true_disp"].to_numpy()
            weights = df_events["event_weight"].to_numpy()

            regressor = sklearn.ensemble.RandomForestRegressor(**self.settings)

            logger.info(f"Telescope {tel_id}...")
            regressor.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = regressor

    def predict(self, event_data):
        """
        Reconstructs the arrival directions with trained RFs, and
        performs the MARS-like head-tail discriminations.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Pandas data frame of shower events

        Returns
        -------
        reco_params: pandas.core.frame.DataFrame
            Pandas data frame of the reconstructed arrival directions
        """

        reco_params = pd.DataFrame()

        # Apply trained RFs per telescope
        tel_ids = np.unique(event_data.index.get_level_values("tel_id"))

        for tel_id in tel_ids:

            df_events = event_data.query(f"tel_id == {tel_id}")

            if self.use_unsigned_features:
                x_predict = np.abs(df_events[self.features].to_numpy())
            else:
                x_predict = df_events[self.features].to_numpy()

            reco_disp = self.telescope_rfs[tel_id].predict(x_predict)

            responses_per_estimator = []
            for estimator in self.telescope_rfs[tel_id].estimators_:
                responses_per_estimator.append(estimator.predict(x_predict))

            reco_disp_var = np.var(responses_per_estimator, axis=0)

            # Reconstruct the Alt/Az directions of the head and tail
            # candidates, i.e., the directions on the major shower axis
            # and separated by the DISP parameter from the image CoG

            tel_pointing = AltAz(
                alt=u.Quantity(df_events["pointing_alt"].to_numpy(), u.rad),
                az=u.Quantity(df_events["pointing_az"].to_numpy(), u.rad),
            )

            tel_frame = TelescopeFrame(telescope_pointing=tel_pointing)

            event_coord = SkyCoord(
                u.Quantity(df_events["x"].to_numpy(), u.m),
                u.Quantity(df_events["y"].to_numpy(), u.m),
                frame=self.tel_descriptions[tel_id].camera.geometry.frame,
            )

            event_coord = event_coord.transform_to(tel_frame)

            for flip in [0, 1]:

                psi_per_flip = df_events["psi"].to_numpy() + 180 * flip

                event_coord_per_flip = event_coord.directional_offset_by(
                    u.Quantity(psi_per_flip, u.deg), u.Quantity(reco_disp, u.deg)
                )

                reco_alt_per_flip = event_coord_per_flip.altaz.alt.to_value(u.deg)
                reco_az_per_flip = event_coord_per_flip.altaz.az.to_value(u.deg)

                df_per_flip = pd.DataFrame(
                    data={
                        "reco_disp": reco_disp,
                        "reco_disp_var": reco_disp_var,
                        "reco_alt": reco_alt_per_flip,
                        "reco_az": reco_az_per_flip,
                        "flip": flip,
                    },
                    index=df_events.index,
                )

                df_per_flip.set_index("flip", append=True, inplace=True)
                reco_params = reco_params.append(df_per_flip)

        reco_params.sort_index(inplace=True)

        # Get the flip combinations minimizing the sum of the angular
        # distances between the head and tail candidates.

        # Here we first define all the possible flip combinations. For
        # example, in case that we have two telescope images, in total
        # 4 combinations can be defined as follows:
        #   [(head, head), (head, tail), (tail, head), (tail, tail)]
        # where the i-th element of each tuple means the i-th telescope
        # image. In case of 3 images we have in total 8 combinations.

        flip_combinations = np.array(
            list(itertools.product([0, 1], repeat=len(tel_ids)))
        )

        # Next, we define all the any 2 telescope combinations. For
        # example, in case of 3 telescopes, in total 3 combinations are
        # defined as follows:
        #                 [(1, 2), (1, 3), (2, 3)]
        # where the elements of the tuples mean the telescope IDs.
        # In case of 2 telescopes there is only one combination.

        tel_combinations = list(itertools.combinations(tel_ids, 2))

        group_size = reco_params.groupby(["obs_id", "event_id"]).size()
        n_events = len(group_size)

        distances = []

        for flip_combo in flip_combinations:

            container = {}

            # Set the directions of a given flip combination
            for tel_id, flip in zip(tel_ids, flip_combo):
                container[tel_id] = reco_params.query(
                    f"(tel_id == {tel_id}) & (flip == {flip})"
                )

            # Then, we calculate the angular distances for each any 2
            # telescope combination and sum up them
            dists_per_flip_combo = np.zeros(n_events)

            for tel_combo in tel_combinations:

                tel_id_1 = tel_combo[0]
                tel_id_2 = tel_combo[1]

                theta = angular_separation(
                    lon1=u.Quantity(container[tel_id_1]["reco_az"].to_numpy(), u.deg),
                    lat1=u.Quantity(container[tel_id_1]["reco_alt"].to_numpy(), u.deg),
                    lon2=u.Quantity(container[tel_id_2]["reco_az"].to_numpy(), u.deg),
                    lat2=u.Quantity(container[tel_id_2]["reco_alt"].to_numpy(), u.deg),
                )

                dists_per_flip_combo += theta.to_value(u.deg)

            distances.append(dists_per_flip_combo.tolist())

        # Finally, we extract the indices of the flip combinations for
        # each event with which the angular distances become minimum
        distances = np.array(distances)
        distances_min = distances.min(axis=0)

        condition = distances == distances_min
        indices = np.where(condition.transpose())[1]

        flips = flip_combinations[indices].ravel()

        obs_ids = event_data.index.get_level_values("obs_id")
        event_ids = event_data.index.get_level_values("event_id")
        tel_ids = event_data.index.get_level_values("tel_id")

        multi_indices = pd.MultiIndex.from_arrays(
            [obs_ids, event_ids, tel_ids, flips],
            names=["obs_id", "event_id", "tel_id", "flip"],
        )

        reco_params = reco_params.loc[multi_indices]
        reco_params.reset_index(level="flip", inplace=True)
        reco_params.drop("flip", axis=1, inplace=True)
        reco_params.sort_index(inplace=True)

        # Add the minimum angular distances to the output data frame,
        # since they are useful to separate gamma and hadron events
        # (hadron events tend to have larger distances than gammas)
        df_disp_diff = pd.DataFrame(
            data={"disp_diff_sum": distances_min}, index=group_size.index
        )

        reco_params = reco_params.join(df_disp_diff)

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
            "tel_descriptions": self.tel_descriptions,
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
        self.tel_descriptions = input_data["tel_descriptions"]
        self.use_unsigned_features = input_data["use_unsigned_features"]
        self.telescope_rfs = input_data["telescope_rfs"]


class EventClassifier:
    """
    RF classifiers to reconstruct the gammaness.
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
            Pandas data frame of shower events
        """

        self.telescope_rfs.clear()

        # Train RFs per telescope
        tel_ids = np.unique(event_data.index.get_level_values("tel_id"))

        for tel_id in tel_ids:

            df_events = event_data.query(f"tel_id == {tel_id}")

            if self.use_unsigned_features:
                x_train = np.abs(df_events[self.features].to_numpy())
            else:
                x_train = df_events[self.features].to_numpy()

            y_train = df_events["true_event_class"].to_numpy()
            weights = df_events["event_weight"].to_numpy()

            classifier = sklearn.ensemble.RandomForestClassifier(**self.settings)

            logger.info(f"Telescope {tel_id}...")
            classifier.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = classifier

    def predict(self, event_data):
        """
        Reconstructs the gammaness.

        Parameters
        ----------
        event_data: pandas.core.frame.DataFrame
            Pandas data frame of shower events

        Returns
        -------
        reco_params: pandas.core.frame.DataFrame
            Pandas data frame of the gammaness
        """

        reco_params = pd.DataFrame()

        # Apply trained RFs per telescope
        tel_ids = np.unique(event_data.index.get_level_values("tel_id"))

        for tel_id in tel_ids:

            df_events = event_data.query(f"tel_id == {tel_id}")

            if self.use_unsigned_features:
                x_predict = np.abs(df_events[self.features].to_numpy())
            else:
                x_predict = df_events[self.features].to_numpy()

            gammaness = self.telescope_rfs[tel_id].predict_proba(x_predict)[:, 0]

            # Compute the variance of the binomial distribution
            gammaness_var = gammaness * (1 - gammaness)

            # Set the artificial finite value in case the variance is 0,
            # to avoid the inverse value becomes infinite
            gammaness_var[gammaness == 1] = 0.99 * (1 - 0.99)
            gammaness_var[gammaness == 0] = 0.01 * (1 - 0.01)

            df_reco_class = pd.DataFrame(
                data={"gammaness": gammaness, "gammaness_var": gammaness_var},
                index=df_events.index,
            )

            reco_params = reco_params.append(df_reco_class)

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
