# coding: utf-8

import datetime
import yaml
import time
import argparse
import pandas as pd
import numpy as np
import sklearn
import sklearn.ensemble
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz
from astropy.coordinates.angle_utilities import angular_separation, position_angle
from matplotlib import colors
import matplotlib.pyplot as plt

import ctapipe
from ctapipe.instrument import CameraGeometry
from ctapipe.instrument import TelescopeDescription
from ctapipe.instrument import OpticsDescription
from ctapipe.instrument import SubarrayDescription

from magicctapipe.utils.plot import *
from magicctapipe.utils.tels import *
from magicctapipe.utils.utils import *
from magicctapipe.utils.filedir import *
from magicctapipe.train.event_processing import DirectionEstimatorPandas

PARSER = argparse.ArgumentParser(
    description="This tools fits the event classification random forest on "
    "the specified events files. For stereo data.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
PARSER.add_argument('-cfg', '--config_file', type=str, required=True,
                    help='Configuration file to steer the code execution')


def compute_separation_angle(shower_data):
    separation = dict()

    tel_ids = get_tel_ids(shower_data)

    for tel_id in tel_ids:
        event_coord_true = SkyCoord(
            shower_data_test.loc[(slice(None), slice(None), tel_id),
                                 'true_az'].values * u.rad,
            shower_data_test.loc[(slice(None), slice(None), tel_id),
                                 'true_alt'].values * u.rad,
            frame=AltAz()
        )

        event_coord_reco = SkyCoord(
            shower_data_test.loc[(slice(None), slice(None), tel_id),
                                 'az_reco'].values * u.rad,
            shower_data_test.loc[(slice(None), slice(None), tel_id),
                                 'alt_reco'].values * u.rad,
            frame=AltAz()
        )

        separation[tel_id] = event_coord_true.separation(event_coord_reco)

    event_coord_true = SkyCoord(
        shower_data_test['true_az'].values * u.rad,
        shower_data_test['true_alt'].values * u.rad,
        frame=AltAz()
    )

    event_coord_reco = SkyCoord(
        shower_data_test['az_reco_mean'].values * u.rad,
        shower_data_test['alt_reco_mean'].values * u.rad,
        frame=AltAz()
    )

    separation[0] = event_coord_true.separation(event_coord_reco)

    # Converting to a data frame
    separation_df = pd.DataFrame(
        data={'sep_0': separation[0]}, index=shower_data_test.index)
    for tel_id in separation_df.index.levels[2]:
        df = pd.DataFrame(
            data={f'sep_{tel_id:d}': separation[tel_id]},
            index=shower_data_test.loc[(slice(None), slice(None), tel_id),
                                       'true_az'].index)
        separation_df = separation_df.join(df)

    separation_df = separation_df.join(shower_data)

    for tel_id in [0]+tel_ids:
        print(f"  Tel {tel_id} scatter: ",
              f"{separation[tel_id].to(u.deg).std():.2f}")

    return separation_df


def get_weights(mc_data, alt_edges, intensity_edges):
    mc_hist, _, _ = np.histogram2d(mc_data['tel_alt'],
                                   mc_data['intensity'],
                                   bins=[alt_edges, intensity_edges])

    availability_hist = np.clip(mc_hist, 0, 1)

    # --- MC weights ---
    mc_alt_bins = np.digitize(mc_data['tel_alt'], alt_edges) - 1
    mc_intensity_bins = np.digitize(mc_data['intensity'], intensity_edges) - 1

    # Treating the out-of-range events
    mc_alt_bins[mc_alt_bins == len(alt_edges) - 1] = len(alt_edges) - 2
    mc_intensity_bins[mc_intensity_bins == len(
        intensity_edges) - 1] = len(intensity_edges) - 2

    mc_weights = 1 / mc_hist[mc_alt_bins, mc_intensity_bins]
    mc_weights *= availability_hist[mc_alt_bins, mc_intensity_bins]

    # --- Storing to a data frame ---
    mc_weight_df = pd.DataFrame(data={'event_weight': mc_weights},
                                index=mc_data.index)

    return mc_weight_df


# =================
# === Main code ===
# =================
def train_direction_rf_stereo(config_file):
    # ------------------------------
    cfg = load_cfg_file_check(config_file=config_file, label='direction_rf')
    # ------------------------------

    # MAGIC telescope positions in m wrt. to the center of CTA simulations
    # magic_tel_positions = {
    #     1: [-27.24, -146.66, 50.00] * u.m,
    #     2: [-96.44, -96.77, 51.00] * u.m
    # }

    # --- MAGIC description ----------------------------------------------------
    # magic_optics = OpticsDescription.from_name('MAGIC')
    # magic_cam = CameraGeometry.from_name('MAGICCam')
    # magic_tel_description = TelescopeDescription(name='MAGIC',
    #                                              tel_type='MAGIC',
    #                                              optics=magic_optics,
    #                                              camera=magic_cam)
    # magic_tel_descriptions = {
    #     cfg['MAGIC']['tel_ids'][0]: magic_tel_description,
    #     cfg['MAGIC']['tel_ids'][1]: magic_tel_description
    # }
    # !!! MAGIC sub-array !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # magic_subarray = SubarrayDescription(
    #     name='MAGIC',
    #     tel_positions=convert_positions_dict(cfg['MAGIC']['positions']),
    #     tel_descriptions=magic_tel_descriptions
    # )

    # --- LST description ------------------------------------------------------
    # lst_optics = OpticsDescription.from_name('LST')
    # lst_cam = CameraGeometry.from_name('LSTCam')
    # lst_tel_description = TelescopeDescription(name='LST',
    #                                            tel_type='LST',
    #                                            optics=lst_optics,
    #                                            camera=lst_cam)
    # lst_tel_descriptions = {
    #     cfg['LST']['tel_ids'][0]: lst_tel_description,
    #     cfg['LST']['tel_ids'][1]: lst_tel_description,
    #     cfg['LST']['tel_ids'][2]: lst_tel_description,
    #     cfg['LST']['tel_ids'][3]: lst_tel_description
    # }
    # !!! LST sub-array !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # lst_subarray = SubarrayDescription(
    #     name='LST',
    #     tel_positions=convert_positions_dict(cfg['LST']['positions']),
    #     tel_descriptions=lst_tel_descriptions
    # )

    # --- MAGIC - LST description ----------------------------------------------
    array_tel_descriptions = {
        **get_tel_descriptions(
            name='LST', cam='LSTCam', tel_ids=cfg['LST']['tel_ids']),
        **get_tel_descriptions(
            name='MAGIC', cam='MAGICCam', tel_ids=cfg['MAGIC']['tel_ids'])
    }

    # --- Train sample ---
    # info_message('Loading M1 train data...', prefix='DirRF')
    # hillas_data_m1 = pd.read_hdf(
    #     cfg['data_files']['mc']['train_sample']['magic1']['hillas_output'],
    #     key='dl1/hillas_params')
    # hillas_data_m1.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    # hillas_data_m1.sort_index(inplace=True)

    # info_message('Loading M2 train data...', prefix='DirRF')
    # hillas_data_m2 = pd.read_hdf(
    #     cfg['data_files']['mc']['train_sample']['magic2']['hillas_output'],
    #     key='dl1/hillas_params')
    # hillas_data_m2.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    # hillas_data_m2.sort_index(inplace=True)

    # info_message('Merging the data sets...', prefix='DirRF')
    # shower_data_train = hillas_data_m1.append(hillas_data_m2)
    # shower_data_train.sort_index(inplace=True)

    f_ = cfg['data_files']['mc']['train_sample']['hillas_h5']
    shower_data_train = load_dl1_data(f_)

    # Computing event weights
    info_message('Computing the train sample event weights...', prefix='DirRF')
    sin_edges = np.linspace(0, 1, num=51)
    alt_edges = np.lib.scimath.arcsin(sin_edges)
    intensity_edges = np.logspace(1, 5, num=51)

    mc_weights = get_weights(shower_data_train, alt_edges, intensity_edges)

    shower_data_train = shower_data_train.join(mc_weights)

    # --- Test sample ---
    # info_message('Loading M1 test data...', prefix='DirRF')
    # hillas_data_m1 = pd.read_hdf(
    #     cfg['data_files']['mc']['test_sample']['magic1']['hillas_output'],
    #     key='dl1/hillas_params')
    # hillas_data_m1.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    # hillas_data_m1.sort_index(inplace=True)

    # info_message('Loading M2 test data...', prefix='DirRF')
    # hillas_data_m2 = pd.read_hdf(
    #     cfg['data_files']['mc']['test_sample']['magic2']['hillas_output'],
    #     key='dl1/hillas_params')
    # hillas_data_m2.set_index(['obs_id', 'event_id', 'tel_id'], inplace=True)
    # hillas_data_m2.sort_index(inplace=True)

    # info_message('Merging the data sets...', prefix='DirRF')
    # shower_data_test = hillas_data_m1.append(hillas_data_m2)
    # shower_data_test.sort_index(inplace=True)

    f_ = cfg['data_files']['mc']['test_sample']['hillas_h5']
    shower_data_test = load_dl1_data(f_)

    # --- Data preparation ---
    l_ = ['obs_id', 'event_id']
    shower_data_train['multiplicity'] = \
        shower_data_train['intensity'].groupby(level=l_).count()
    shower_data_test['multiplicity'] = \
        shower_data_test['intensity'].groupby(level=l_).count()

    # Applying the cuts
    shower_data_train = shower_data_train.query(cfg['direction_rf']['cuts'])
    shower_data_test = shower_data_test.query(cfg['direction_rf']['cuts'])

    # --- Training the direction RF ---
    info_message('Training the RF\n', prefix='DirRF')

    direction_estimator = DirectionEstimatorPandas(
        cfg['direction_rf']['features'],
        array_tel_descriptions,
        **cfg['direction_rf']['settings']
    )
    direction_estimator.fit(shower_data_train)
    direction_estimator.save(os.path.join(cfg['direction_rf']['save_dir'],
                                          cfg['direction_rf']['joblib_name']))
    # direction_estimator.load(cfg['direction_rf']['save_name'])

    tel_ids = get_tel_ids(shower_data_test)

    # Printing the parameter "importances"
    for kind in direction_estimator.telescope_rfs:
        rfs_ = direction_estimator.telescope_rfs[kind]
        for tel_id in rfs_:
            feature_importances = rfs_[tel_id].feature_importances_
            print(f'  Kind: {kind}, tel_id: {tel_id}')
            z = zip(cfg['direction_rf']['features'][kind], feature_importances)
            for feature, importance in z:
                print(f"  {feature:.<15s}: {importance:.4f}")
            print('')

    # --- Applying RF to the "test" sample ---
    info_message('Applying RF to the "test" sample', prefix='DirRF')
    coords_reco = direction_estimator.predict(shower_data_test)
    shower_data_test = shower_data_test.join(coords_reco)

    # --- Evaluating the performance ---
    info_message('Evaluating the performance', prefix='DirRF')
    separation_df = compute_separation_angle(shower_data_test)

    # Energy-dependent resolution
    info_message('Estimating the energy-dependent resolution', prefix='DirRF')
    energy_edges = np.logspace(-1, 1.3, num=20)
    energy = (energy_edges[1:] * energy_edges[:-1])**0.5

    energy_psf = dict()
    # for i in range(3):
    for i in [0]+tel_ids:
        energy_psf[i] = np.zeros_like(energy)

    for ei in range(len(energy_edges) - 1):
        cuts = f'(true_energy>= {energy_edges[ei]:.2e})'
        cuts += f' & (true_energy < {energy_edges[ei+1]: .2e})'
        # cuts += ' & (intensity > 100)'
        # cuts += ' & (length > 0.05)'
        cuts += ' & (multiplicity > 1)'
        query = separation_df.query(cuts)

        # for pi in range(3): # OLD
        for pi in [0]+tel_ids:
            if pi > 0:
                tel_id = pi
            else:
                tel_id = 1
            selection = query.loc[(slice(None), slice(None), tel_id),
                                  f'sep_{pi}'].dropna()
            energy_psf[pi][ei] = np.percentile(selection, 68)

    # Offset-dependent resolution
    info_message('Estimating the offset-dependent resolution', prefix='DirRF')
    offset = angular_separation(
        separation_df['tel_az'], separation_df['tel_alt'],
        separation_df['true_az'], separation_df['true_alt']
    )

    separation_df['offset'] = np.degrees(offset)

    offset_edges = np.linspace(0, 1.3, num=10)
    offset = (offset_edges[1:] * offset_edges[:-1])**0.5

    offset_psf = dict()
    # for i in range(3): # OLD
    for i in [0]+tel_ids:
        offset_psf[i] = np.zeros_like(offset)

    for oi in range(len(offset_edges) - 1):
        cuts = f'(offset >= {offset_edges[oi]:.2f})'
        cuts += f' & (offset < {offset_edges[oi+1]:.2f})'
        # cuts += ' & (intensity > 100)'
        # cuts += ' & (length > 0.05)'
        cuts += ' & (multiplicity > 1)'
        query = separation_df.query(cuts)

        # for pi in range(3):
        for pi in [0]+tel_ids:
            if pi > 0:
                tel_id = pi
            else:
                tel_id = 1
            selection = query.loc[(slice(None), slice(None), tel_id),
                                  [f'sep_{pi}']].dropna()
            offset_psf[pi][oi] = np.percentile(selection[f'sep_{pi}'], 68)

    # ================
    # === Plotting ===
    # ================

    plt.figure(figsize=(12, 12))
    # plt.style.use('presentation')

    plt.xlabel(r'$\theta^2$, deg$^2$')

    # for tel_id in [0, 1, 2]:
    for tel_id in [0]+tel_ids:
        plt.subplot2grid((3, 2), (tel_id, 0))
        plt.title(f'Tel {tel_id}')
        plt.xlabel(r'$\theta^2$, deg$^2$')
        # plt.semilogy()
        plt.hist(separation_df[f'sep_{tel_id}']**2, bins=100,
                 range=(0, 0.5), density=True, alpha=0.1, color='C0')
        plt.hist(separation_df[f'sep_{tel_id}']**2, bins=100,
                 range=(0, 0.5), density=True, histtype='step', color='C0')
        plt.grid(linestyle=':')

        plt.subplot2grid((3, 2), (tel_id, 1))
        plt.xlabel(r'$\theta$, deg')
        plt.xlim(0, 2.0)
        plt.hist(separation_df[f'sep_{tel_id}'], bins=400, range=(
            0, 5), cumulative=True, density=True, alpha=0.1, color='C0')
        plt.hist(separation_df[f'sep_{tel_id}'], bins=400, range=(
            0, 5), cumulative=True, density=True, histtype='step', color='C0')
        plt.grid(linestyle=':')

    plt.tight_layout()
    save_plt(n=cfg['direction_rf']['fig_name_theta2'],
             rdir=cfg['direction_rf']['save_dir'],
             vect='')
    plt.close()

    plt.clf()
    plt.semilogx()
    plt.xlabel('Energy [TeV]')
    plt.ylabel(r'$\sigma_{68}$ [deg]')
    plt.ylim(0, 1.0)
    plt.plot(energy, energy_psf[0], linewidth=4, label='Total')
    plt.plot(energy, energy_psf[1], label='M1')
    plt.plot(energy, energy_psf[2], label='M2')
    plt.grid(linestyle=':')
    plt.legend()
    save_plt(n=cfg['direction_rf']['fig_name_PSF_energy'],
             rdir=cfg['direction_rf']['save_dir'],
             vect='')
    plt.close()

    plt.clf()
    plt.xlabel('Offset [deg]')
    plt.ylabel(r'$\sigma_{68}$ [deg]')
    plt.ylim(0, 0.5)
    plt.plot(offset, offset_psf[0], linewidth=4, label='Total')
    plt.plot(offset, offset_psf[1], label='M1')
    plt.plot(offset, offset_psf[2], label='M2')
    plt.grid(linestyle=':')
    plt.legend()
    save_plt(n=cfg['direction_rf']['fig_name_PSF_offset'],
             rdir=cfg['direction_rf']['save_dir'],
             vect='')
    plt.close()


if __name__ == '__main__':
    args = PARSER.parse_args()
    kwargs = args.__dict__
    start_time = time.time()
    train_direction_rf_stereo(
        config_file=kwargs['config_file'],
    )
    print("Execution time: %.2f s" % (time.time() - start_time))
