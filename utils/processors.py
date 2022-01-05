import joblib
import itertools
import numpy as np
import pandas as pd
import sklearn.ensemble
from astropy import units as u
from astropy.coordinates import AltAz, SkyCoord
from astropy.coordinates.angle_utilities import angular_separation
from ctapipe.coordinates import CameraFrame, TelescopeFrame


__all__ = [
    'EnergyEstimatorPandas',
    'DirectionEstimatorPandas',
    'EventClassifierPandas'
]

class EnergyEstimatorPandas:

    def __init__(self, feature_names=[], rf_settings={}):

        self.feature_names = feature_names
        self.rf_settings = rf_settings
        self.telescope_rfs = dict()

    def fit(self, input_data):

        param_names = self.feature_names + ['event_weight', 'mc_energy']

        # --- train the RFs per telescope ---
        telescope_ids = np.unique(input_data.index.get_level_values('tel_id'))

        for tel_id in telescope_ids:

            df_tel = input_data.loc[(slice(None), slice(None), tel_id), param_names]

            x_train = df_tel[self.feature_names].values
            y_train = np.log10(df_tel['mc_energy'].values)
            weights = df_tel['event_weight'].values

            regressor = sklearn.ensemble.RandomForestRegressor(**self.rf_settings)

            print(f'Telescope {tel_id}...')
            regressor.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = regressor

    def predict(self, input_data):

        reco_params = pd.DataFrame()

        # --- apply the RFs per telescope ---
        telescope_ids = np.unique(input_data.index.get_level_values('tel_id'))

        for tel_id in telescope_ids:
            
            df_tel = input_data.loc[(slice(None), slice(None), tel_id), self.feature_names]

            reco_energy = 10**self.telescope_rfs[tel_id].predict(df_tel.values)

            responces_per_estimator = []

            for estimator in self.telescope_rfs[tel_id].estimators_:
                responces_per_estimator.append(10**estimator.predict(df_tel.values))

            reco_energy_err = np.std(responces_per_estimator, axis=0)

            df_reco_energy = pd.DataFrame(
                data={'reco_energy': reco_energy, 'reco_energy_err': reco_energy_err}, 
                index=df_tel.index
            )

            reco_params = reco_params.append(df_reco_energy)

        reco_params.sort_index(inplace=True)

        # --- compute the mean weighted by the error ---
        weights = 1 / reco_params['reco_energy_err']
        weighted_energy = np.log10(reco_params['reco_energy']) * weights

        weights_sum = weights.groupby(['obs_id', 'event_id']).sum()
        weighted_energy_sum = weighted_energy.groupby(['obs_id', 'event_id']).sum()
        
        reco_energy_mean = weighted_energy_sum / weights_sum

        reco_params['reco_energy_mean'] = 10 ** reco_energy_mean

        return reco_params

    def save(self, file_name):

        output_data = dict()

        output_data['feature_names'] = self.feature_names
        output_data['rf_settings'] = self.rf_settings 
        output_data['telescope_rfs'] = self.telescope_rfs

        joblib.dump(output_data, file_name)

    def load(self, file_name):

        input_data = joblib.load(file_name)

        self.feature_names = input_data['feature_names']
        self.rf_settings = input_data['rf_settings']
        self.telescope_rfs = input_data['telescope_rfs']


class DirectionEstimatorPandas:

    def __init__(self, feature_names=[], tel_descriptions={}, rf_settings={}):

        self.feature_names = feature_names
        self.tel_descriptions = tel_descriptions
        self.rf_settings = rf_settings
        self.telescope_rfs = dict()

    def fit(self, input_data):

        param_names = self.feature_names + ['event_weight', 'mc_disp']

        # --- train the RFs per telescope ---
        telescope_ids = np.unique(input_data.index.get_level_values('tel_id'))

        for tel_id in telescope_ids:

            df_tel = input_data.loc[(slice(None), slice(None), tel_id), param_names]

            x_train = df_tel[self.feature_names].values
            y_train = df_tel['mc_disp'].values
            weights = df_tel['event_weight'].values

            regressor = sklearn.ensemble.RandomForestRegressor(**self.rf_settings)

            print(f'Telescope {tel_id}...')
            regressor.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = regressor

    def predict(self, input_data):

        param_names = self.feature_names + ['alt_tel', 'az_tel', 'x', 'y', 'psi']

        reco_params = pd.DataFrame()

        # --- apply the RFs per telescope ---
        telescope_ids = np.unique(input_data.index.get_level_values('tel_id'))

        for tel_id in telescope_ids:

            df_tel = input_data.loc[(slice(None), slice(None), tel_id), param_names]

            reco_disp = self.telescope_rfs[tel_id].predict(df_tel[self.feature_names].values)

            responces_per_estimator = []

            for estimator in self.telescope_rfs[tel_id].estimators_:
                responces_per_estimator.append(estimator.predict(df_tel[self.feature_names].values))

            reco_disp_err = np.std(responces_per_estimator, axis=0)

            # --- reconstruct the Alt/Az directions per flip ---
            tel_pointing = AltAz(
                alt=u.Quantity(df_tel['alt_tel'].values, u.deg), az=u.Quantity(df_tel['az_tel'].values, u.deg)
            )

            telescope_frame = TelescopeFrame(telescope_pointing=tel_pointing)

            camera_frame = CameraFrame(
                focal_length=self.tel_descriptions[tel_id].optics.equivalent_focal_length, 
                rotation=self.tel_descriptions[tel_id].camera.geometry.cam_rotation
            )

            event_coord = SkyCoord(
                u.Quantity(df_tel['x'].values, u.m), u.Quantity(df_tel['y'].values, u.m), frame=camera_frame
            )

            event_coord = event_coord.transform_to(telescope_frame)

            for flip in [0, 1]:

                psi_per_flip = u.Quantity(df_tel['psi'].values, u.deg) + u.Quantity(180*flip, u.deg)
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

        # --- get the flip combinations of minimum distance ---
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
                    lat2=u.Quantity(container[tel_id_2]['reco_alt'].values, u.deg)
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
            [obs_ids, event_ids, tel_ids, flips], names=['obs_id', 'event_id', 'tel_id', 'flip']
        )

        reco_params = reco_params.loc[multi_indices]
        reco_params.reset_index(level='flip', inplace=True)
        reco_params.drop('flip', axis=1, inplace=True)
        reco_params.sort_index(inplace=True)

        df_dist = pd.DataFrame(
            data={'dist_min': distances_min}, index=reco_params.groupby(['obs_id', 'event_id']).size().index
        )

        reco_params = reco_params.join(df_dist)

        # --- compute the mean weighted by the error ---
        x_coords = np.cos(np.deg2rad(reco_params['reco_alt'])) * np.cos(np.deg2rad(reco_params['reco_az']))
        y_coords = np.cos(np.deg2rad(reco_params['reco_alt'])) * np.sin(np.deg2rad(reco_params['reco_az']))
        z_coords = np.sin(np.deg2rad(reco_params['reco_alt']))

        weights = 1 / reco_params['reco_disp_err']

        weighted_x_coords = x_coords * weights
        weighted_y_coords = y_coords * weights
        weighted_z_coords = z_coords * weights

        weights_sum = weights.groupby(['obs_id', 'event_id']).sum()

        weighted_x_coords_sum = weighted_x_coords.groupby(['obs_id', 'event_id']).sum()
        weighted_y_coords_sum = weighted_y_coords.groupby(['obs_id', 'event_id']).sum()
        weighted_z_coords_sum = weighted_z_coords.groupby(['obs_id', 'event_id']).sum()

        x_coords_mean = weighted_x_coords_sum / weights_sum
        y_coords_mean = weighted_y_coords_sum / weights_sum
        z_coords_mean = weighted_z_coords_sum / weights_sum

        coord_mean = SkyCoord(
            x=x_coords_mean.values, y=y_coords_mean.values, z=z_coords_mean.values, representation_type='cartesian'
        ) 
            
        df_mean = pd.DataFrame(
            data={'reco_alt_mean': coord_mean.spherical.lat.to(u.deg).value,
                  'reco_az_mean': coord_mean.spherical.lon.to(u.deg).value},
            index=reco_params.groupby(['obs_id', 'event_id']).sum().index
        )

        reco_params = reco_params.join(df_mean)

        return reco_params

    def save(self, file_name):

        output_data = dict()

        output_data['feature_names'] = self.feature_names
        output_data['tel_descriptions'] = self.tel_descriptions
        output_data['rf_settings'] = self.rf_settings
        output_data['telescope_rfs'] = self.telescope_rfs

        joblib.dump(output_data, file_name)

    def load(self, file_name):

        input_data = joblib.load(file_name)

        self.feature_names = input_data['feature_names']
        self.tel_descriptions = input_data['tel_descriptions']
        self.rf_settings = input_data['rf_settings']
        self.telescope_rfs = input_data['telescope_rfs']
        

class EventClassifierPandas:

    def __init__(self, feature_names=[], rf_settings={}):

        self.feature_names = feature_names
        self.rf_settings = rf_settings
        self.telescope_rfs = dict()

    def fit(self, input_data):

        param_names = self.feature_names + ['event_weight', 'event_class']

        # --- train the RFs per telescope ---
        telescope_ids = np.unique(input_data.index.get_level_values('tel_id'))

        for tel_id in telescope_ids:

            df_tel = input_data.loc[(slice(None), slice(None), tel_id), param_names]

            x_train = df_tel[self.feature_names].values
            y_train = df_tel['event_class'].values
            weights = df_tel['event_weight'].values

            classifier = sklearn.ensemble.RandomForestClassifier(**self.rf_settings)

            print(f'Telescope {tel_id}...')
            classifier.fit(x_train, y_train, sample_weight=weights)

            self.telescope_rfs[tel_id] = classifier

    def predict(self, input_data):

        reco_params = pd.DataFrame()

        # --- apply the RFs per telescope ---
        telescope_ids = np.unique(input_data.index.get_level_values('tel_id'))

        for tel_id in telescope_ids:

            df_tel = input_data.loc[(slice(None), slice(None), tel_id), self.feature_names]

            responses = self.telescope_rfs[tel_id].predict_proba(df_tel.values)

            df_reco_class = pd.DataFrame(
                data={'gammaness': responses[:, 0], 'hadronness': responses[:, 1]},
                index=df_tel.index
            )

            reco_params = reco_params.append(df_reco_class)

        reco_params.sort_index(inplace=True)

        # --- compute the mean ---
        reco_params_mean = reco_params.groupby(['obs_id', 'event_id']).mean()

        reco_params['gammaness_mean'] = reco_params_mean['gammaness']
        reco_params['hadronness_mean'] = reco_params_mean['hadronness']

        return reco_params

    def save(self, file_name):

        output_data = dict()

        output_data['feature_names'] = self.feature_names
        output_data['rf_settings'] = self.rf_settings 
        output_data['telescope_rfs'] = self.telescope_rfs

        joblib.dump(output_data, file_name)

    def load(self, file_name):

        input_data = joblib.load(file_name)

        self.feature_names = input_data['feature_names']
        self.rf_settings = input_data['rf_settings']
        self.telescope_rfs = input_data['telescope_rfs']

