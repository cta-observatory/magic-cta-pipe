import re
import itertools
import numpy as np
import astropy.units as u
from copy import deepcopy

from abc import ABC, abstractmethod

import sklearn
import sklearn.ensemble
import pandas as pd
import joblib

from ctapipe.image import tailcuts_clean, hillas_parameters
from ctapipe.io import event_source
from ctapipe.containers import ReconstructedContainer, ReconstructedEnergyContainer, ParticleClassificationContainer
from ctapipe.reco import EnergyRegressor
from ctapipe.reco.HillasReconstructor import TooFewTelescopesException

from ctapipe.coordinates import CameraFrame, TelescopeFrame
from astropy.coordinates import AltAz, SkyCoord
from astropy import units as u
from astropy.coordinates.angle_utilities import angular_separation, position_angle


class HillasFeatureSelector(ABC):
    """
    The base class that handles the event Hillas parameter extraction
    for future use with the random forest energy and classification pipelines.
    """

    def __init__(self, hillas_params_to_use, hillas_reco_params_to_use, telescopes):
        """
        Constructor. Stores the settings that will be used during the parameter
        extraction.

        Parameters
        ----------
        hillas_params_to_use: list
            A list of Hillas parameter names that should be extracted.
        hillas_reco_params_to_use: list
            A list of the Hillas "stereo" parameters (after HillasReconstructor),
            that should also be extracted.
        telescopes: list
            List of telescope identifiers. Only events triggering these will be processed.
        """

        self.hillas_params_to_use = hillas_params_to_use
        self.hillas_reco_params_to_use = hillas_reco_params_to_use
        self.telescopes = telescopes

        n_features_per_telescope = len(hillas_params_to_use) + len(hillas_reco_params_to_use)
        self.n_features = n_features_per_telescope * len(telescopes)

    @staticmethod
    def _get_param_value(param):
        """
        An internal method that extracts the parameter value from both
        float and Quantity instances.

        Parameters
        ----------
        param: float or astropy.unit.Quantity
            A parameter whos value should be extracted.

        Returns
        -------
        float:
            An extracted value. For float the param itself is returned,
            for Quantity the Quantity.value is taken.

        """

        if isinstance(param, u.Quantity):
            return param.value
        else:
            return param

    @abstractmethod
    def fill_event(self, event, hillas_reco_result, target):
        """
        A dummy function to process an event.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        hillas_reco_result: ReconstructedShowerContainer
            A container with the computed shower direction properties.
        target: float
            A target variable for future regression/classification model.

        Returns
        -------

        """

        pass


class EventFeatureSelector(HillasFeatureSelector):
    """
    A class that performs the selects event features for further reconstruction with
    ctapipe.reco.RegressorClassifierBase.
    """

    def __init__(self, hillas_params_to_use, hillas_reco_params_to_use, telescopes):
        """
        Constructor. Stores the settings that will be used during the parameter
        extraction.

        Parameters
        ----------
        hillas_params_to_use: list
            A list of Hillas parameter names that should be extracted.
        hillas_reco_params_to_use: list
            A list of the Hillas "stereo" parameters (after HillasReconstructor),
            that should also be extracted.
        telescopes: list
            List of telescope identifiers. Only events triggering these will be processed.
        """

        super(EventFeatureSelector, self).__init__(hillas_params_to_use, hillas_reco_params_to_use, telescopes)

        self.events = []
        self.event_targets = []

    def fill_event(self, event, hillas_reco_result, target=None):
        """
        This method fills the event features to the feature list.
        Optionally it can add a "target" value, which can be used
        to check the accuracy of the reconstruction.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        hillas_reco_result: ReconstructedShowerContainer
            A container with the computed shower direction properties.
        target: float
            A target variable for future regression/classification model.

        Returns
        -------

        """

        event_record = dict()
        for tel_id in self.telescopes:
            feature_entry = []

            for param_name in self.hillas_params_to_use:
                param = event.dl1.tel[tel_id].hillas_params[param_name]

                feature_entry.append(self._get_param_value(param))

            for param_name in self.hillas_reco_params_to_use:
                param = hillas_reco_result[param_name]

                feature_entry.append(self._get_param_value(param))

            if np.all(np.isfinite(feature_entry)):
                event_record[tel_id] = [feature_entry]

        self.events.append(event_record)
        self.event_targets.append(target)


class EventFeatureTargetSelector(HillasFeatureSelector):
    """
    A class that performs the selects event features for further training of the
    ctapipe.reco.RegressorClassifierBase model.
    """

    def __init__(self, hillas_params_to_use, hillas_reco_params_to_use, telescopes):
        """
        Constructor. Stores the settings that will be used during the parameter
        extraction.

        Parameters
        ----------
        hillas_params_to_use: list
            A list of Hillas parameter names that should be extracted.
        hillas_reco_params_to_use: list
            A list of the Hillas "stereo" parameters (after HillasReconstructor),
            that should also be extracted.
        telescopes: list
            List of telescope identifiers. Only events triggering these will be processed.
        """

        super(EventFeatureTargetSelector, self).__init__(hillas_params_to_use, hillas_reco_params_to_use, telescopes)

        self.features = dict()
        self.targets = dict()
        self.events = []

        for tel_id in self.telescopes:
            self.features[tel_id] = []
            self.targets[tel_id] = []

    def fill_event(self, event, hillas_reco_result, target):
        """
        This method fills the event features to the feature list;
        "target" values are added to their own "target" list.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        hillas_reco_result: ReconstructedShowerContainer
            A container with the computed shower direction properties.
        target: float
            A target variable for future regression/classification model.

        Returns
        -------

        """

        event_record = dict()
        for tel_id in self.telescopes:
            feature_entry = []

            for param_name in self.hillas_params_to_use:
                param = event.dl1.tel[tel_id].hillas_params[param_name]

                feature_entry.append(self._get_param_value(param))

            for param_name in self.hillas_reco_params_to_use:
                param = hillas_reco_result[param_name]

                feature_entry.append(self._get_param_value(param))

            if np.all(np.isfinite(feature_entry)):
                self.features[tel_id].append(feature_entry)
                self.targets[tel_id].append(self._get_param_value(target))

                event_record[tel_id] = [feature_entry]
        self.events.append(event_record)


class EventProcessor:
    """
    This class is meant to represents the DL0->DL2 analysis pipeline.
    It handles event loading, Hillas parameter estimation and storage
    (DL0->DL1), stereo (with >=2 telescopes) event direction/impact etc.
    reconstruction and RF energy estimation.
    """

    def __init__(self, calibrator, hillas_reconstructor, min_survived_pixels=10):
        """
        Constructor. Sets the calibration / Hillas processing workers.

        Parameters
        ----------
        calibrator: ctapipe.calib.CameraCalibrator
            A desired camera calibrator instance.
        hillas_reconstructor: ctapipe.reco.HillasReconstructor
            A "stereo" (with >=2 telescopes) Hillas reconstructor instance that
            will be used to determine the event direction/impact etc.
        min_survived_pixels: int, optional
            Minimal number of pixels in the shower image, that should survive
            image cleaning. Hillas parameters are not computed for events falling
            below this threshold.
            Defaults to 10.
        """

        self.calibrator = calibrator
        self.hillas_reconstructor = hillas_reconstructor
        self.min_survived_pixels = min_survived_pixels

        self.events = []
        self.reconstruction_results = []

    def _dl1_process(self, event):
        """
        Internal method that performs DL0->DL1 event processing.
        This involves image cleaning and Hillas parameter calculation.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL0 event data.

        Returns
        -------
        DataContainer:
            Event with computed Hillas parameters.

        """

        tels_with_data = list(event.r1.tels_with_data)

        for tel_id in tels_with_data:
            subarray = event.inst.subarray
            camera = subarray.tel[tel_id].camera

            self.calibrator.calibrate(event)

            event_image = event.dl1.tel[tel_id].image[1]

            mask = tailcuts_clean(camera, event_image,
                                  picture_thresh=6, boundary_thresh=6)
            event_image_cleaned = event_image.copy()
            event_image_cleaned[~mask] = 0

            n_survived_pixels = event_image_cleaned[mask].size

            # Calculate hillas parameters
            # It fails for empty images, so we apply a cut on the number
            # of survived pixels
            if n_survived_pixels > self.min_survived_pixels:
                try:
                    event.dl1.tel[tel_id].hillas_params = hillas_parameters(camera, event_image_cleaned)
                except:
                    print('Failed')
                    pass

        return event

    def _update_event_direction(self, event, reco_container):
        """
        Internal method used to compute the shower direction/impact etc. from
        intersection of the per-telescope image planes (from Hillas parameters)
        and store them to the provided reconstruction container.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        reco_container: ReconstructedContainer
            A container that will hold the computed shower properties.

        Returns
        -------
        ReconstructedContainer:
            Updated shower reconstruction container.

        """

        # Performing a geometrical direction reconstruction
        try:
            reco_container.shower['hillas'] = self.hillas_reconstructor.predict_from_dl1(event)
        except TooFewTelescopesException:
            # No reconstruction possible. Resetting to defaults
            reco_container.shower['hillas'].reset()

        return reco_container

    def _update_event_energy(self, event, reco_container):
        """
        Internal method used to compute the shower energy from a pre-trained RF
        and store it to the provided reconstruction container.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        reco_container: ReconstructedContainer
            A container that will hold the computed shower energy.

        Returns
        -------
        ReconstructedContainer:
            Updated shower reconstruction container.

        """

        return reco_container

    def _update_event_classification(self, event, reco_container):
        """
        Internal method used to compute the classify the using the pre-trained RF
        and store it to the provided reconstruction container.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        reco_container: ReconstructedContainer
            A container that will hold the computed shower class.

        Returns
        -------
        ReconstructedContainer:
            Updated shower reconstruction container.

        """

        return reco_container

    def _load_events(self, file_name, append_to_existing_events):
        """
        Internal method that takes care of the event loading from the specified file
        and DL1 processing. The DL1 events are also stored in the self.events list
        for future usage; their R0/R1/DL0 containers are reset to save memory.

        Parameters
        ----------
        file_name: str
            A file name from which to read the events.
        append_to_existing_events: bool
            Defines whether the previously filled event list should be cleared
            or if the new events should be appended to it.

        Returns
        -------

        """

        with event_source(input_url=file_name) as source:
            event_generator = source._generator()

            if not append_to_existing_events:
                self.events = []

            # Running the parameter computation over the event list
            for event in event_generator:
                event = self._dl1_process(event)

                event.r0.reset()
                event.r1.reset()
                event.dl0.reset()

                self.events.append(deepcopy(event))

    def process_file(self, file_name,
                     append_to_existing_events=True,
                     do_direction_reconstruction=True,
                     do_energy_reconstruction=True,
                     do_classification=True):

        """
        This method represents the file processing pipeline: data loading,
        DL0->DL1 processing and the subsequent direction/energy/classification.

        Parameters
        ----------
        file_name: str
            A file name from which to read the events.
        append_to_existing_events: bool
            Defines whether the previously filled event list should be cleared
            or if the new events should be appended to it.
        do_direction_reconstruction: bool, optional
            Sets whether the direction reconstruction should be performed.
            Defaults to True.
        do_energy_reconstruction: bool, optional
            Sets whether the energy reconstruction should be performed.
            Requires a trained energy random forest.
            Defaults to False. NOT YES IMPLEMENTED
        do_classification: bool, optional
            Sets whether the event classification should be performed.
            Requires a trained classifier random forest.
            Defaults to False. NOT YES IMPLEMENTED

        Returns
        -------

        """

        self._load_events(file_name, append_to_existing_events=append_to_existing_events)

        self.reconstruction_results = []

        # Running the parameter computation over the event list
        for event in self.events:
            # Run the event properties reconstruction
            reco_results = ReconstructedContainer()

            if do_direction_reconstruction:
                reco_results = self._update_event_direction(event, reco_results)

            if do_energy_reconstruction:
                reco_results = self._update_event_energy(event, reco_results)

            if do_classification:
                reco_results = self._update_event_classification(event, reco_results)

            self.reconstruction_results.append(reco_results)


class EnergyEstimator:
    def __init__(self, hillas_params_to_use, hillas_reco_params_to_use, telescopes):
        """
        Constructor. Stores the settings that will be used during the parameter
        extraction.

        Parameters
        ----------
        hillas_params_to_use: list
            A list of Hillas parameter names that should be extracted.
        hillas_reco_params_to_use: list
            A list of the Hillas "stereo" parameters (after HillasReconstructor),
            that should also be extracted.
        telescopes: list
            List of telescope identifiers. Only events triggering these will be processed.
        """

        self.hillas_params_to_use = hillas_params_to_use
        self.hillas_reco_params_to_use = hillas_reco_params_to_use
        self.telescopes = telescopes

        n_features_per_telescope = len(hillas_params_to_use) + len(hillas_reco_params_to_use)
        self.n_features = n_features_per_telescope * len(telescopes)

        self.train_features = dict()
        self.train_targets = dict()
        self.train_events = []

        for tel_id in self.telescopes:
            self.train_features[tel_id] = []
            self.train_targets[tel_id] = []

        self.regressor = EnergyRegressor(cam_id_list=self.telescopes)

    @staticmethod
    def _get_param_value(param):
        """
        An internal method that extracts the parameter value from both
        float and Quantity instances.

        Parameters
        ----------
        param: float or astropy.unit.Quantity
            A parameter whose value should be extracted.

        Returns
        -------
        float:
            An extracted value. For float the param itself is returned,
            for Quantity the Quantity.value is taken.

        """

        if isinstance(param, u.Quantity):
            return param.value
        else:
            return param

    def add_train_event(self, event, reco_result):
        """
        This method fills the event features to the feature list;
        "target" values are added to their own "target" list.

        Parameters
        ----------
        event: DataContainer
            Container instances, holding DL1 event data.
        reco_result: ReconstructedContainer
            A container with the already reconstructed event properties.
            Its 'shower' part must have the 'hillas' key.

        Returns
        -------

        """

        event_record = dict()

        for tel_id in self.telescopes:
            feature_entry = []

            for param_name in self.hillas_params_to_use:
                param = event.dl1.tel[tel_id].hillas_params[param_name]

                feature_entry.append(self._get_param_value(param))

            for param_name in self.hillas_reco_params_to_use:
                param = reco_result.shower['hillas'][param_name]

                feature_entry.append(self._get_param_value(param))

            if np.all(np.isfinite(feature_entry)):
                event_energy = event.mc.energy.to(u.TeV).value
                self.train_features[tel_id].append(feature_entry)
                self.train_targets[tel_id].append(np.log10(event_energy))

                event_record[tel_id] = [feature_entry]

        self.train_events.append(event_record)

    def process_event(self, event, reco_result):
        event_record = dict()
        for tel_id in self.telescopes:
            feature_entry = []

            for param_name in self.hillas_params_to_use:
                param = event.dl1.tel[tel_id].hillas_params[param_name]

                feature_entry.append(self._get_param_value(param))

            for param_name in self.hillas_reco_params_to_use:
                param = reco_result.shower['hillas'][param_name]

                feature_entry.append(self._get_param_value(param))

            if np.all(np.isfinite(feature_entry)):
                event_record[tel_id] = [feature_entry]

        predicted_energy_dict = self.regressor.predict_by_event([event_record])

        reconstructed_energy = 10**predicted_energy_dict['mean'].value * u.TeV
        std = predicted_energy_dict['std'].value
        rel_uncert = 0.5 * (10 ** std - 1 / 10 ** std)

        energy_container = ReconstructedEnergyContainer()
        energy_container.energy = reconstructed_energy
        energy_container.energy_uncert = energy_container.energy * rel_uncert
        energy_container.is_valid = True
        energy_container.tel_ids = list(event_record.keys())

        return energy_container

    def fit_model(self):
        _ = self.regressor.fit(self.train_features, self.train_targets)

    def save_model(self):
        pass

    def load_model(self):
        pass


class EnergyEstimatorPandas:
    """
    This class trains/applies the random forest regressor for event energy,
    using as the input Hillas and stereo parameters, stored in a Pandas data frame.
    It trains a separate regressor for each telescope. Further another "consolidating"
    regressor is applied to combine the per-telescope predictions.
    """

    def __init__(self, feature_names, **rf_settings):
        """
        Constructor. Gets basic settings.

        Parameters
        ----------
        feature_names: list
            Feature names (str type) to be used by the regressor. Must correspond to the
            columns of the data frames that will be processed.
        rf_settings: dict
            The settings to be passed to the random forest regressor.
        """

        self.feature_names = feature_names
        self.rf_settings = rf_settings

        self.telescope_regressors = dict()
        self.consolidating_regressor = None

    def fit(self, shower_data):
        """
        Fits the regressor model.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called self.feature_names.

        Returns
        -------
        None

        """

        self.train_per_telescope_rf(shower_data)

        # shower_data_with_energy = self.apply_per_telescope_rf(shower_data)
        #
        # features = shower_data_with_energy['energy_reco']
        # features = features.fillna(0).groupby(['obs_id', 'event_id']).sum()
        # features = features.values
        #
        # target = shower_data_with_energy['true_energy'].groupby(['obs_id', 'event_id']).mean().values
        #
        # self.consolidating_regressor = sklearn.ensemble.RandomForestRegressor(self.rf_settings)
        # self.consolidating_regressor.fit(features, target)

    def predict(self, shower_data):
        """
        Applies the trained regressor to the data.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called self.feature_names.

        Returns
        -------
        pandas.DataFrame:
            Updated data frame with the computed shower energies.

        """

        shower_data_with_energy = self.apply_per_telescope_rf(shower_data)

        # Selecting and log-scaling the reconstructed energies
        energy_reco = shower_data_with_energy['energy_reco'].apply(np.log10)

        # Getting the weights
        weights = shower_data_with_energy['energy_reco_err']

        # Weighted energies
        weighted = energy_reco * weights

        # Grouping per-event data
        weighted_group = weighted.groupby(level=['obs_id', 'event_id'])
        weight_group = weights.groupby(level=['obs_id', 'event_id'])

        # Weighted mean log-energy
        log_energy = weighted_group.sum() / weight_group.sum()

        shower_data_with_energy['energy_reco_mean'] = 10 ** log_energy

        return shower_data_with_energy

    def train_per_telescope_rf(self, shower_data):
        """
        Trains the energy regressors for each of the available telescopes.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called self.feature_names.

        Returns
        -------
        None

        """

        idx = pd.IndexSlice

        tel_ids = shower_data.index.levels[2]

        self.telescope_regressors = dict()

        for tel_id in tel_ids:
            input_data = shower_data.loc[idx[:, :, tel_id], self.feature_names + ['event_weight', 'true_energy']]
            input_data.dropna(inplace=True)

            x_train = input_data[list(self.feature_names)].values
            y_train = np.log10(input_data['true_energy'].values)
            weight = input_data['event_weight'].values

            regressor = sklearn.ensemble.RandomForestRegressor(**self.rf_settings)
            regressor.fit(x_train, y_train, sample_weight=weight)

            self.telescope_regressors[tel_id] = regressor

    def apply_per_telescope_rf(self, shower_data):
        """
        Applies the regressors, trained per each telescope.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called self.feature_names.

        Returns
        -------
        pandas.DataFrame:
            Updated data frame with the computed shower energies.

        """

        tel_ids = shower_data.index.levels[2]

        energy_reco = pd.DataFrame()

        for tel_id in tel_ids:
            # Selecting data
            this_telescope = shower_data.loc[(slice(None), slice(None), tel_id), self.feature_names]
            this_telescope = this_telescope.dropna()
            features = this_telescope.values

            # Getting the RF response
            response = 10**self.telescope_regressors[tel_id].predict(features)

            per_tree_responses = []
            for tree in self.telescope_regressors[tel_id].estimators_:
                per_tree_responses.append(10**tree.predict(features))
            response_err = np.std(per_tree_responses, axis=0)

            # Storing to a data frame
            result = {'energy_reco': response, 'energy_reco_err': response_err}
            df = pd.DataFrame(data=result, index=this_telescope.index)

            energy_reco = energy_reco.append(df)

        energy_reco.sort_index(inplace=True)

        return energy_reco

    def save(self, file_name):
        """
        Saves trained regressors to the specified joblib file.

        Parameters
        ----------
        file_name: str
            Output file name.

        Returns
        -------
        None

        """

        output = dict()
        output['feature_names'] = self.feature_names
        output['telescope_regressors'] = self.telescope_regressors
        output['consolidating_regressor'] = self.consolidating_regressor

        joblib.dump(output, file_name)

    def load(self, file_name):
        """
        Loads pre-trained regressors to the specified joblib file.

        Parameters
        ----------
        file_name: str
            Output file name.

        Returns
        -------
        None

        """

        data = joblib.load(file_name)

        self.feature_names = data['feature_names']
        self.telescope_regressors = data['telescope_regressors']
        self.consolidating_regressor = data['consolidating_regressor']


class DirectionEstimatorPandas:
    """
    This class trains/applies the random forest regressor for event energy,
    using as the input Hillas and stereo parameters, stored in a Pandas data frame.
    It trains a separate regressor for each telescope. Further another "consolidating"
    regressor is applied to combine the per-telescope predictions.
    """

    def __init__(self, feature_names, tel_descriptions, **rf_settings):
        """
        Constructor. Gets basic settings.

        Parameters
        ----------
        feature_names: dict
            Feature names (str type) to be used by the regressor. Must correspond to the
            columns of the data frames that will be processed. Must be a dict with the keys
            "disp" and "pos_angle", each containing the corresponding feature lists.
        tel_descriptions: dict
            A dictionary of the ctapipe.instrument.TelescopeDescription instances, covering
            the telescopes used to record the data.
        rf_settings: dict
            A dictionary of the parameters to be transferred to regressors
            (sklearn.ensemble.RandomForestRegressor).
        """

        self.feature_names = feature_names
        self.telescope_descriptions = tel_descriptions

        self.rf_settings = rf_settings

        self.telescope_rfs = dict(disp={}, pos_angle_shift={})

    def _get_disp_and_position_angle(self, shower_data):
        """
        Computes the displacement and its position angle between the event
        shower image and its original coordinates.

        Parameters
        ----------
        shower_data: pd.DataFrame
            A data frame with the event properties.

        Returns
        -------
        pandas.DataFrame:
            - "disp_true" column:
                Computed displacement in radians.
            - "pos_angle_true" column:
                Computed displacement position angles in radians.

        """

        tel_ids = shower_data.index.levels[2]

        result = pd.DataFrame()

        for tel_id in tel_ids:
            optics = self.telescope_descriptions[tel_id].optics
            camera = self.telescope_descriptions[tel_id].camera

            values = {'disp_true': {}, 'pos_angle_shift_true': {}}

            this_telescope = shower_data.loc[(slice(None), slice(None), tel_id), shower_data.columns]

            tel_pointing = AltAz(alt=this_telescope['tel_alt'].values * u.rad,
                                 az=this_telescope['tel_az'].values * u.rad)

            camera_frame = CameraFrame(focal_length=optics.equivalent_focal_length,
                                       rotation=camera.cam_rotation)

            telescope_frame = TelescopeFrame(telescope_pointing=tel_pointing)

            camera_coord = SkyCoord(this_telescope['x'].values * u.m,
                                    this_telescope['y'].values * u.m,
                                    frame=camera_frame)
            shower_coord_in_telescope = camera_coord.transform_to(telescope_frame)

            event_coord = SkyCoord(this_telescope['true_az'].values * u.rad,
                                   this_telescope['true_alt'].values * u.rad,
                                   frame=AltAz())
            event_coord_in_telescope = event_coord.transform_to(telescope_frame)

            disp = angular_separation(shower_coord_in_telescope.altaz.az,
                                      shower_coord_in_telescope.altaz.alt,
                                      event_coord_in_telescope.altaz.az,
                                      event_coord_in_telescope.altaz.alt)

            pos_angle = shower_coord_in_telescope.position_angle(event_coord)
            psi = this_telescope['psi'].values * u.deg

            toshift_by_pi = abs(pos_angle.value - 3.14159 - psi.to(u.rad).value) < 1
            toshift_by_pi = np.int16(toshift_by_pi)

            values['disp_true'] = disp.to(u.rad).value
            values['pos_angle_shift_true'] = toshift_by_pi

            df = pd.DataFrame.from_dict(values)
            df.index = this_telescope.index

            result = result.append(df)

        result.sort_index(inplace=True)

        return result

    def fit(self, shower_data):
        """
        Fits the regressor model.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called
            self.feature_names and self.target_name.

        Returns
        -------
        None

        """

        disp_pos_angle = self._get_disp_and_position_angle(shower_data)
        shower_data = shower_data.join(disp_pos_angle)

        print('Training "disp" RFs...')
        self.telescope_rfs['disp'] = self._train_per_telescope_rf(shower_data, 'disp')

        print('Training "PA" RFs...')
        self.telescope_rfs['pos_angle_shift'] = self._train_per_telescope_rf(shower_data, 'pos_angle_shift')

    @staticmethod
    def _set_flip(bit_mask, tel_id):
        flip_code = 2 ** tel_id

        out_mask = np.bitwise_or(bit_mask, flip_code)

        return out_mask

    @staticmethod
    def _get_flip(bit_mask, tel_id):
        flip_code = 2 ** tel_id

        out_mask = np.bitwise_and(bit_mask, flip_code)
        out_mask = np.int8(out_mask == flip_code)

        return out_mask

    @staticmethod
    def _get_flip_combinations(df):
        tel_ids = df.index.levels[2]
        flip_combinations = list(itertools.product([0, 1], repeat=len(tel_ids)))

        return flip_combinations

    @staticmethod
    def _get_telescope_combinations(df):
        tel_ids = df.index.levels[2]

        telescope_combinations = set(itertools.product(tel_ids, repeat=2))
        self_repetitions = set(zip(tel_ids, tel_ids))
        telescope_combinations = telescope_combinations - self_repetitions

        return telescope_combinations

    def _get_directions_with_flips(self, reco_df):
        tel_ids = reco_df.index.levels[2]

        direction_with_flips = pd.DataFrame()

        for tel_id in tel_ids:
            optics = self.telescope_descriptions[tel_id].optics
            camera = self.telescope_descriptions[tel_id].camera

            # Selecting events from this telescope
            this_telescope = reco_df.loc[(slice(None), slice(None), tel_id), reco_df.columns]

            # Definining the coordinate systems
            tel_pointing = AltAz(alt=this_telescope['tel_alt'].values * u.rad,
                                 az=this_telescope['tel_az'].values * u.rad)

            camera_frame = CameraFrame(focal_length=optics.equivalent_focal_length,
                                       rotation=camera.cam_rotation)

            telescope_frame = TelescopeFrame(telescope_pointing=tel_pointing)

            camera_coord = SkyCoord(this_telescope['x'].values * u.m,
                                    this_telescope['y'].values * u.m,
                                    frame=camera_frame)

            # Shower image coordinates on the sky
            shower_coord_in_telescope = camera_coord.transform_to(telescope_frame)

            disp_reco = this_telescope['disp_reco'].values * u.rad
            position_angle_reco = this_telescope['psi'].values * u.deg

            # Shower direction coordinates on the sky
            for flip in [0, 1]:
                event_coord_reco = shower_coord_in_telescope.directional_offset_by(
                    position_angle_reco + flip * u.rad * np.pi, disp_reco)

                # Saving to a data frame
                alt = event_coord_reco.altaz.alt.to(u.rad).value
                az = event_coord_reco.altaz.az.to(u.rad).value

                df = pd.DataFrame(data={'alt_reco': alt,
                                        'az_reco': az,
                                        'disp_reco': this_telescope['disp_reco'].values,
                                        'psi': this_telescope['psi'].values,
                                        },
                                  index=this_telescope.index)

                # Adding the "flip" index
                df = pd.concat([df], keys=[flip] * len(df), names=['flip'])
                df = df.reset_index()
                df = df.set_index(['obs_id', 'event_id', 'tel_id', 'flip'])

                direction_with_flips = direction_with_flips.append(df)

        return direction_with_flips

    def _get_total_pairwise_dist_with_flips(self, df_with_flips):
        telescope_combinations = self._get_telescope_combinations(df_with_flips)
        flip_combinations = self._get_flip_combinations(df_with_flips)

        tel_ids = df_with_flips.index.levels[2]
        tel_ids = tel_ids.sort_values()

        dist2 = pd.DataFrame()

        for flip_comb in flip_combinations:

            flip_code = 0
            for flip, tel_id in zip(flip_comb, tel_ids):
                if flip:
                    flip_code = self._set_flip(flip_code, tel_id)

            for tel_comb in telescope_combinations:
                tel_id1, tel_id2 = tel_comb

                tel_df1 = df_with_flips.xs(tel_id1, level='tel_id')
                tel_df2 = df_with_flips.xs(tel_id2, level='tel_id')
                flip1 = self._get_flip(flip_code, tel_id1)
                flip2 = self._get_flip(flip_code, tel_id2)

                az1 = tel_df1.xs(flip1, level='flip')['az_reco']
                az2 = tel_df2.xs(flip2, level='flip')['az_reco']
                alt1 = tel_df1.xs(flip1, level='flip')['alt_reco']
                alt2 = tel_df2.xs(flip2, level='flip')['alt_reco']

                dist = angular_separation(az1, alt1, az2, alt2)
                dist.fillna(0, inplace=True)

                dist2_ = dist.apply(np.square)

                dist2_ = pd.DataFrame(data={f'dist2': dist2_})
                dist2_['tel_comb'] = [tel_comb] * len(dist2_)
                dist2_['flip_code'] = [flip_code] * len(dist2_)
                dist2_.reset_index(inplace=True)
                dist2_.set_index(['obs_id', 'event_id', 'flip_code', 'tel_comb'], inplace=True)

                dist2 = dist2.append(dist2_)

        _group = dist2.groupby(level=['obs_id', 'event_id', 'flip_code'])
        total_dist2 = _group.sum()

        return total_dist2

    def _get_flip_choice_from_pairwise_dist2(self, dist2_df, tel_ids):
        _group = dist2_df.groupby(level=['obs_id', 'event_id'])
        min_dist2 = _group.min()
        min_dist2.head()

        min_dist2_ = min_dist2.reindex(dist2_df.index)
        result = dist2_df.join(min_dist2_, rsuffix='_')
        result = result.query('(dist2 == dist2_)')

        result.reset_index(inplace=True)

        flip_choice = pd.DataFrame()
        for tel_id in tel_ids:
            df_ = result[['obs_id', 'event_id', 'flip_code']]
            df_['tel_id'] = tel_id
            df_['flip'] = self._get_flip(df_['flip_code'], tel_id)
            df_.head()

            flip_choice = flip_choice.append(df_)

        flip_choice = flip_choice.drop('flip_code', axis=1, errors='raise')
        flip_choice.set_index(['obs_id', 'event_id', 'tel_id', 'flip'], inplace=True)

        return flip_choice

    def _get_average_direction(self, df):
        # Stereo estimate - arithmetical mean
        # First getting cartensian XYZ
        _x = np.cos(df['az_reco']) * np.cos(df['alt_reco'])
        _y = np.sin(df['az_reco']) * np.cos(df['alt_reco'])
        _z = np.sin(df['alt_reco'])

        # Getting the weights
        weights = df['weight']

        # Weighted XYZ
        _x = _x * weights
        _y = _y * weights
        _z = _z * weights

        # Grouping per-event data
        x_group = _x.groupby(level=['obs_id', 'event_id'])
        y_group = _y.groupby(level=['obs_id', 'event_id'])
        z_group = _z.groupby(level=['obs_id', 'event_id'])

        # Averaging: weighted mean
        x_mean = x_group.sum() / weights.sum()
        y_mean = y_group.sum() / weights.sum()
        z_mean = z_group.sum() / weights.sum()

        # Computing the averaged spherical coordinates
        coord_mean = SkyCoord(representation_type='cartesian',
                              x=x_mean.values,
                              y=y_mean.values,
                              z=z_mean.values)

        # Converting to a data frame
        coord_mean_df = pd.DataFrame(data={'az_reco_mean': coord_mean.spherical.lon.to(u.rad),
                                           'alt_reco_mean': coord_mean.spherical.lat.to(u.rad)},
                                     index=x_mean.index)

        return coord_mean_df

    def predict(self, shower_data):
        """
        Applies the trained regressor to the data.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called
            self.feature_names and self.target_name.

        Returns
        -------
        pandas.DataFrame:
            Updated data frame with the computed shower energies.

        """

        # Computing the estimates from individual telescopes
        # This is the "mono" estimate
        direction_reco = self._apply_per_telescope_rf(shower_data)

        shower_data_new = shower_data.join(direction_reco)
        shower_data_new['multiplicity'] = shower_data_new['intensity'].groupby(level=['obs_id', 'event_id']).count()

        tel_ids = shower_data_new.index.levels[2]

        # Selecting "stereo" (multi-telescope) events only
        multi_events = shower_data_new.query('multiplicity > 1')

        # Computing direction of all possible head-tail flips
        direction_with_flips = self._get_directions_with_flips(multi_events)
        # Computing the total distance between the per-telescope positions in all flip combinations
        pairwise_dist = self._get_total_pairwise_dist_with_flips(direction_with_flips)
        # Selecting the flip combination with the smallest distance
        flip_choice = self._get_flip_choice_from_pairwise_dist2(pairwise_dist, tel_ids)

        # Filtering the reconstructed directions for the "min distance" flip combination
        common_idx = flip_choice.index.intersection(direction_with_flips.index)
        direction_with_selected_flips = direction_with_flips.loc[common_idx]

        # Assigning weights to average out per-telescope estimates
        direction_with_selected_flips['weight'] = multi_events['disp_reco_err']
        # Getting the final direction prediction
        multi_events_reco = self._get_average_direction(direction_with_selected_flips)

        # Merging the "multi-telescope" and "mono" estimates
        direction_reco = direction_reco.join(multi_events_reco.reindex(direction_reco.index))

        return direction_reco

    def _train_per_telescope_rf(self, shower_data, kind):
        """
        Trains the energy regressors for each of the available telescopes.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called
            self.feature_names and self.target_name.
        kind: str
            RF kind. Can be "disp" (regressor is used) or 'pos_angle_shift' (classifier is used).

        Returns
        -------
        dict:
            Regressors/classifiers for each of the telescopes. Keys - telescope IDs.

        """

        idx = pd.IndexSlice

        tel_ids = shower_data.index.levels[2]

        target_name = kind + '_true'

        telescope_rfs = dict()

        for tel_id in tel_ids:
            print(f'Training telescope {tel_id}...')

            input_data = shower_data.loc[idx[:, :, tel_id], self.feature_names[kind] + ['event_weight', target_name]]
            input_data.dropna(inplace=True)

            x_train = input_data[self.feature_names[kind]].values
            y_train = input_data[target_name].values
            weight = input_data['event_weight'].values

            if kind == 'pos_angle_shift':
                # This is a binary classification problem - to shift or not to shift
                rf = sklearn.ensemble.RandomForestClassifier(**self.rf_settings)
            else:
                # Regression is needed for "disp"
                rf = sklearn.ensemble.RandomForestRegressor(**self.rf_settings)

            rf.fit(x_train, y_train, sample_weight=weight)

            telescope_rfs[tel_id] = rf

        return telescope_rfs

    def _apply_per_telescope_rf(self, shower_data):
        """
        Applies the regressors, trained per each telescope.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns from self.feature_names.

        Returns
        -------
        pandas.DataFrame:
            Data frame with the computed shower directions. It also contains the "displacement"
            ("disp_reco") of this direction w.r.t. the shower image center and its position angle
            ("pos_angle_reco").

        """

        tel_ids = shower_data.index.levels[2]

        # -----------------------------------------
        # *** Computing disp and position angle ***
        predictions = pd.DataFrame()

        prediction_list = []

        for kind in ['disp', 'pos_angle_shift']:
            pred_ = pd.DataFrame()

            for tel_id in tel_ids:
                # Selecting data
                this_telescope = shower_data.loc[(slice(None), slice(None), tel_id), self.feature_names[kind]]
                this_telescope = this_telescope.dropna()
                features = this_telescope.values

                # Getting the RF response
                if kind == 'pos_angle_shift':
                    response = self.telescope_rfs[kind][tel_id].predict_proba(features)
                    response = response[:, 1]

                    # Storing to a data frame
                    name = f'{kind:s}_reco'
                    df = pd.DataFrame(data={name: response}, index=this_telescope.index)
                else:
                    response = self.telescope_rfs[kind][tel_id].predict(features)

                    per_tree_responses = []
                    for tree in self.telescope_rfs[kind][tel_id].estimators_:
                        per_tree_responses.append(tree.predict(features))
                    response_err = np.std(per_tree_responses, axis=0)

                    # Storing to a data frame
                    val_name = f'{kind:s}_reco'
                    err_name = f'{kind:s}_reco_err'
                    data_ = {val_name: response, err_name: response_err}
                    df = pd.DataFrame(data=data_, index=this_telescope.index)

                if pred_.empty:
                    pred_ = predictions.append(df)
                else:
                    pred_ = pred_.append(df)

            pred_.sort_index(inplace=True)
            prediction_list.append(pred_)

        # Merging the resulting data frames
        for pred_ in prediction_list:
            if predictions.empty:
                predictions = predictions.append(pred_)
            else:
                predictions = predictions.join(pred_)

        # Merging with input for an easier reference
        shower_data = shower_data.join(predictions)
        # -----------------------------------------

        # -----------------------------
        # Computing the sky coordinates
        direction_reco = pd.DataFrame()

        for tel_id in tel_ids:
            optics = self.telescope_descriptions[tel_id].optics
            camera = self.telescope_descriptions[tel_id].camera

            # Selecting events from this telescope
            this_telescope = shower_data.loc[(slice(None), slice(None), tel_id), shower_data.columns]

            # Definining the coordinate systems
            tel_pointing = AltAz(alt=this_telescope['tel_alt'].values * u.rad,
                                 az=this_telescope['tel_az'].values * u.rad)

            camera_frame = CameraFrame(focal_length=optics.equivalent_focal_length,
                                       rotation=camera.cam_rotation)

            telescope_frame = TelescopeFrame(telescope_pointing=tel_pointing)

            camera_coord = SkyCoord(this_telescope['x'].values * u.m,
                                    this_telescope['y'].values * u.m,
                                    frame=camera_frame)

            # Shower image coordinates on the sky
            shower_coord_in_telescope = camera_coord.transform_to(telescope_frame)

            disp_reco = this_telescope['disp_reco'].values * u.rad
            position_angle_reco = this_telescope['psi'].values * u.deg
            # In some cases the position angle should be flipped by pi
            shift = u.rad * np.pi * np.round(this_telescope['pos_angle_shift_reco'].values)
            position_angle_reco += shift

            # Shower direction coordinates on the sky
            event_coord_reco = shower_coord_in_telescope.directional_offset_by(position_angle_reco, disp_reco)

            # Saving to a data frame
            alt = event_coord_reco.altaz.alt.to(u.rad).value
            az = event_coord_reco.altaz.az.to(u.rad).value
            df = pd.DataFrame(data={'alt_reco': alt, 'az_reco': az}, index=this_telescope.index)

            direction_reco = direction_reco.append(df)

        direction_reco = direction_reco.join(predictions)
        direction_reco.sort_index(inplace=True)
        # -----------------------------

        return direction_reco

    def save(self, file_name):
        """
        Saves trained regressors to the specified joblib file.

        Parameters
        ----------
        file_name: str
            Output file name.

        Returns
        -------
        None

        """

        output = dict()
        output['rf_settings'] = self.rf_settings
        output['feature_names'] = self.feature_names
        output['telescope_regressors'] = self.telescope_rfs
        output['telescope_descriptions'] = self.telescope_descriptions

        joblib.dump(output, file_name)

    def load(self, file_name):
        """
        Loads pre-trained regressors to the specified joblib file.

        Parameters
        ----------
        file_name: str
            Output file name.

        Returns
        -------
        None

        """

        data = joblib.load(file_name)

        self.rf_settings = data['rf_settings']
        self.feature_names = data['feature_names']
        self.telescope_rfs = data['telescope_regressors']
        self.telescope_descriptions = data['telescope_descriptions']


class EventClassifierPandas:
    """
    This class trains/applies the random forest classifier for event types (e.g. gamma / proton),
    using as the input Hillas and stereo parameters, stored in a Pandas data frame.
    It trains a separate classifier for each telescope. Further their outputs are combined
    to give the final answer.
    """

    def __init__(self, feature_names, **rf_settings):
        """
        Constructor. Gets basic settings.

        Parameters
        ----------
        feature_names: list
            Feature names (str type) to be used by the classifier. Must correspond to the
            columns of the data frames that will be processed.
        rf_settings: dict
            The settings to be passed to the random forest classifier.
        """

        self.feature_names = feature_names
        self.rf_settings = rf_settings

        self.telescope_classifiers = dict()

    def fit(self, shower_data):
        """
        Fits the classification model.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called self.feature_names.

        Returns
        -------
        None

        """

        self.train_per_telescope_rf(shower_data)

        # shower_data_with_energy = self.apply_per_telescope_rf(shower_data)
        #
        # features = shower_data_with_energy['energy_reco']
        # features = features.fillna(0).groupby(['obs_id', 'event_id']).sum()
        # features = features.values
        #
        # target = shower_data_with_energy['true_energy'].groupby(['obs_id', 'event_id']).mean().values
        #
        # self.consolidating_regressor = sklearn.ensemble.RandomForestRegressor(self.rf_settings)
        # self.consolidating_regressor.fit(features, target)

    def predict(self, shower_data):
        """
        Applies the trained classifiers to the data.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called self.feature_names.

        Returns
        -------
        pandas.DataFrame:
            Updated data frame with the computed shower classes.

        """

        shower_class = self.apply_per_telescope_rf(shower_data)

        # Grouping per-event data
        class_group = shower_class.groupby(level=['obs_id', 'event_id'])

        # Averaging
        class_mean = class_group.mean()

        for class_name in class_mean.columns:
            shower_class[class_name + '_mean'] = class_mean[class_name]

        return shower_class

    def train_per_telescope_rf(self, shower_data):
        """
        Trains the event classifiers for each of the available telescopes.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called self.feature_names.

        Returns
        -------
        None

        """

        idx = pd.IndexSlice

        tel_ids = shower_data.index.levels[2]

        self.telescope_classifiers = dict()

        for tel_id in tel_ids:
            input_data = shower_data.loc[idx[:, :, tel_id], self.feature_names + ['event_weight', 'true_event_class']]
            input_data.dropna(inplace=True)

            x_train = input_data[list(self.feature_names)].values
            y_train = input_data['true_event_class'].values
            weight = input_data['event_weight'].values

            classifier = sklearn.ensemble.RandomForestClassifier(**self.rf_settings)
            classifier.fit(x_train, y_train, sample_weight=weight)

            self.telescope_classifiers[tel_id] = classifier

    def apply_per_telescope_rf(self, shower_data):
        """
        Applies the classifiers, trained per each telescope.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters. Must contain columns called self.feature_names.

        Returns
        -------
        pandas.DataFrame:
            Updated data frame with the computed shower classes.

        """

        tel_ids = shower_data.index.levels[2]

        event_class_reco = pd.DataFrame()

        for tel_id in tel_ids:
            # Selecting data
            this_telescope = shower_data.loc[(slice(None), slice(None), tel_id), self.feature_names]
            this_telescope = this_telescope.dropna()
            features = this_telescope.values

            # Getting the RF response
            response = self.telescope_classifiers[tel_id].predict_proba(features)

            # Storing to a data frame
            response_data = dict()
            for class_i in range(response.shape[1]):
                name = f'event_class_{class_i}'
                response_data[name] = response[:, class_i]
            df = pd.DataFrame(response_data, index=this_telescope.index)

            event_class_reco = event_class_reco.append(df)

        event_class_reco.sort_index(inplace=True)

        return event_class_reco

    def save(self, file_name):
        """
        Saves trained regressors to the specified joblib file.

        Parameters
        ----------
        file_name: str
            Output file name.

        Returns
        -------
        None

        """

        output = dict()
        output['feature_names'] = self.feature_names
        output['telescope_classifiers'] = self.telescope_classifiers

        joblib.dump(output, file_name)

    def load(self, file_name):
        """
        Loads pre-trained regressors to the specified joblib file.

        Parameters
        ----------
        file_name: str
            Output file name.

        Returns
        -------
        None

        """

        data = joblib.load(file_name)

        self.feature_names = data['feature_names']
        self.telescope_classifiers = data['telescope_classifiers']


class DirectionStereoEstimatorPandas:
    """
    This class trains/applies the random forest regressor for event direction,
    using the event parameters, stored in a Pandas data frame.
    It trains a separate regressor for each telescope. Further outputs of these
    regressors are combined to deliver the average predictions.
    """

    def __init__(self, feature_names, tel_descriptions, **rf_settings):
        """
        Constructor. Gets basic settings.

        Parameters
        ----------
        feature_names: dict
            Feature names (str type) to be used by the regressor. Must correspond to the
            columns of the data frames that will be processed. Must be a dict with the keys
            "disp" and "pos_angle", each containing the corresponding feature lists.
        tel_descriptions: dict
            A dictionary of the ctapipe.instrument.TelescopeDescription instances, covering
            the telescopes used to record the data.
        rf_settings: dict
            A dictionary of the parameters to be transferred to regressors
            (sklearn.ensemble.RandomForestRegressor).
        """

        self.feature_names = feature_names
        self.telescope_descriptions = tel_descriptions

        self.rf_settings = rf_settings

        self.telescope_regressors = dict(disp={}, pos_angle={})

    @staticmethod
    def _get_tel_ids(shower_data):
        """
        Retrieves the telescope IDs from the input data frame.

        Parameters
        ----------
        shower_data: pd.DataFrame
            A data frame with the event properties.

        Returns
        -------
        set:
            Telescope IDs.

        """

        tel_ids = set()

        for column in shower_data:
            parse = re.findall(r".*_(\d)", column)
            if parse:
                tel_ids.add(int(parse[0]))

        return tel_ids

    def _get_disp_and_position_angle(self, shower_data):
        """
        Computes the displacement and its position angle between the event
        shower image and its original coordinates.

        Parameters
        ----------
        shower_data: pd.DataFrame
            A data frame with the event properties.

        Returns
        -------
        dict:
            - disp: dict
                Computed displacement in radians. Keys - telescope IDs.
            - pos_angle: dict
                Computed displacement position angles in radians. Keys - telescope IDs.

        """

        result = {'disp': {}, 'pos_angle': {}}

        for tel_id in self._get_tel_ids(shower_data):
            optics = self.telescope_descriptions[tel_id].optics
            camera = self.telescope_descriptions[tel_id].camera

            tel_pointing = AltAz(alt=shower_data[f'tel_alt_{tel_id:d}'].values * u.rad,
                                 az=shower_data[f'tel_az_{tel_id:d}'].values * u.rad)

            camera_frame = CameraFrame(focal_length=optics.equivalent_focal_length,
                                       rotation=camera.cam_rotation)

            telescope_frame = TelescopeFrame(telescope_pointing=tel_pointing)

            camera_coord = SkyCoord(shower_data[f'x_{tel_id:d}'].values * u.m,
                                    shower_data[f'y_{tel_id:d}'].values * u.m,
                                    frame=camera_frame)
            shower_coord_in_telescope = camera_coord.transform_to(telescope_frame)

            event_coord = SkyCoord(shower_data[f'true_az_{tel_id:d}'].values * u.rad,
                                   shower_data[f'true_alt_{tel_id:d}'].values * u.rad,
                                   frame=AltAz())
            event_coord_in_telescope = event_coord.transform_to(telescope_frame)

            disp = angular_separation(shower_coord_in_telescope.altaz.az,
                                      shower_coord_in_telescope.altaz.alt,
                                      event_coord_in_telescope.altaz.az,
                                      event_coord_in_telescope.altaz.alt)

            pos_angle = shower_coord_in_telescope.position_angle(event_coord)

            result['disp'][tel_id] = disp.to(u.rad).value
            result['pos_angle'][tel_id] = pos_angle.value

        return result

    def _get_per_telescope_features(self, shower_data, feature_names):
        """
        Extracts the shower features specific to each telescope of
        the available ones.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters.

        Returns
        -------
        output: dict
            Shower features for each telescope (keys - telescope IDs).

        """

        tel_ids = self._get_tel_ids(shower_data)

        features = dict()

        for tel_id in tel_ids:
            tel_feature_names = [name + f'_{tel_id:d}' for name in feature_names]

            this_telescope = shower_data[tel_feature_names]
            this_telescope = this_telescope.dropna()

            features[tel_id] = this_telescope.values

        return features

    def _train_per_telescope_rf(self, shower_data, disp_pa, target_name):
        """
        Trains the regressors for each of the available telescopes.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters.
        disp_pa: dict
            A dictionary with the keys "disp" and "pos_angle", containing
            the target disp and pos_angle values.
        target_name: str
            The name of the target variable. Must one of the keys of disp_pa.

        Returns
        -------
        dict:
            A dictionary of sklearn.ensemble.RandomForestRegressor instances.
            Keys - telescope IDs.

        """

        features = self._get_per_telescope_features(shower_data, self.feature_names[target_name])
        targets = disp_pa[target_name]

        tel_ids = features.keys()

        telescope_regressors = dict()

        for tel_id in tel_ids:
            print(f"Working on telescope {tel_id:d}")
            x_train = features[tel_id]
            y_train = targets[tel_id]

            regressor = sklearn.ensemble.RandomForestRegressor(**self.rf_settings)
            regressor.fit(x_train, y_train)

            telescope_regressors[tel_id] = regressor

        return telescope_regressors

    def _apply_per_telescope_rf(self, shower_data):
        """
        Applies the regressors, trained per each telescope.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters.

        Returns
        -------
        dict:
            Dictionary with predictions for "disp" and "pos_angle".
            Internal keys - telescope IDs.

        """

        predictions = dict(disp={}, pos_angle={})

        for kind in predictions:
            features = self._get_per_telescope_features(shower_data, self.feature_names[kind])
            tel_ids = features.keys()

            for tel_id in tel_ids:
                predictions[kind][tel_id] = self.telescope_regressors[kind][tel_id].predict(features[tel_id])

        return predictions

    def fit(self, shower_data):
        """
        Fits the regressor model.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters.

        Returns
        -------
        None

        """

        disp_pa = self._get_disp_and_position_angle(shower_data)

        print('Training "disp" RFs...')
        self.telescope_regressors['disp'] = self._train_per_telescope_rf(shower_data, disp_pa, 'disp')
        print('Training "PA" RFs...')
        self.telescope_regressors['pos_angle'] = self._train_per_telescope_rf(shower_data, disp_pa, 'pos_angle')

    def predict(self, shower_data, output_suffix='reco'):
        """
        Applies the trained regressor to the data.

        Parameters
        ----------
        shower_data: pandas.DataFrame
            Data frame with the shower parameters.
        output_suffix: str, optional
            Suffix to use with the data frame columns, that will host the regressors
            predictions. Columns will have names "{param_name}_{tel_id}_{output_suffix}".
            Defaults to 'reco'.

        Returns
        -------
        pandas.DataFrame:
            Data frame with the computed shower coordinates.

        """

        disp_pa_reco = self._apply_per_telescope_rf(shower_data)

        coords_reco = dict()
        event_coord_reco = dict()

        tel_ids = list(self._get_tel_ids(shower_data))

        for tel_id in tel_ids:
            optics = self.telescope_descriptions[tel_id].optics
            camera = self.telescope_descriptions[tel_id].camera

            tel_pointing = AltAz(alt=shower_data[f'tel_alt_{tel_id:d}'].values * u.rad,
                                 az=shower_data[f'tel_az_{tel_id:d}'].values * u.rad)

            camera_frame = CameraFrame(focal_length=optics.equivalent_focal_length,
                                       rotation=camera.cam_rotation)

            telescope_frame = TelescopeFrame(telescope_pointing=tel_pointing)

            camera_coord = SkyCoord(shower_data[f'x_{tel_id:d}'].values * u.m,
                                    shower_data[f'y_{tel_id:d}'].values * u.m,
                                    frame=camera_frame)
            shower_coord_in_telescope = camera_coord.transform_to(telescope_frame)

            disp_reco = disp_pa_reco['disp'][tel_id] * u.rad
            position_angle_reco = disp_pa_reco['pos_angle'][tel_id] * u.rad

            event_coord_reco[tel_id] = shower_coord_in_telescope.directional_offset_by(position_angle_reco, disp_reco)

            coords_reco[f'alt_{tel_id:d}_{output_suffix:s}'] = event_coord_reco[tel_id].altaz.alt.to(u.rad).value
            coords_reco[f'az_{tel_id:d}_{output_suffix:s}'] = event_coord_reco[tel_id].altaz.az.to(u.rad).value

        # ------------------------------
        # *** Computing the midpoint ***
        for tel_id in tel_ids:
            event_coord_reco[tel_id] = SkyCoord(coords_reco[f'az_{tel_id:d}_{output_suffix:s}'] * u.rad,
                                                coords_reco[f'alt_{tel_id:d}_{output_suffix:s}'] * u.rad,
                                                frame=AltAz())

        data = [event_coord_reco[tel_id].data for tel_id in tel_ids]

        midpoint_data = data[0]
        for d in data[1:]:
            midpoint_data += d
        midpoint_data /= len(data)

        event_coord_reco[-1] = SkyCoord(midpoint_data, representation_type='unitspherical', frame=AltAz())

        coords_reco[f'alt_{output_suffix:s}'] = event_coord_reco[-1].altaz.alt.to(u.rad).value
        coords_reco[f'az_{output_suffix:s}'] = event_coord_reco[-1].altaz.az.to(u.rad).value
        # ------------------------------

        coord_df = pd.DataFrame.from_dict(coords_reco)
        coord_df.index = shower_data.index

        return coord_df

    def save(self, file_name):
        """
        Saves trained regressors to the specified joblib file.

        Parameters
        ----------
        file_name: str
            Output file name.

        Returns
        -------
        None

        """

        output = dict()
        output['rf_settings'] = self.rf_settings
        output['feature_names'] = self.feature_names
        output['telescope_descriptions'] = self.telescope_descriptions
        output['telescope_regressors'] = self.telescope_regressors

        joblib.dump(output, file_name)

    def load(self, file_name):
        """
        Loads pre-trained regressors to the specified joblib file.

        Parameters
        ----------
        file_name: str
            Output file name.

        Returns
        -------
        None

        """

        data = joblib.load(file_name)

        self.rf_settings = data['rf_settings']
        self.feature_names = data['feature_names']
        self.telescope_regressors = data['telescope_regressors']
        self.telescope_descriptions = data['telescope_descriptions']
