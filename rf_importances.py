# coding: utf-8

import argparse
import yaml
import datetime

import pandas as pd
import numpy as np

import sklearn
import sklearn.ensemble

import ctapipe
from ctapipe.instrument import CameraDescription
from ctapipe.instrument import TelescopeDescription
from ctapipe.instrument import OpticsDescription
from ctapipe.instrument import SubarrayDescription
import sys
sys.path.append("/home/gpirola/ctasoft/magic-cta-pipe/")
from event_processing import EnergyEstimatorPandas,DirectionEstimatorPandas,EventClassifierPandas

from astropy import units as u

from matplotlib import pyplot, colors

arg_parser = argparse.ArgumentParser(description="""
This tools fits the energy random forest regressor on the specified events files.
""")

arg_parser.add_argument("--config", default="config.yaml",
                        help='Configuration file to steer the code execution.')
arg_parser.add_argument("--stereo",
                        help='Use stereo DL1 files.',
                        action='store_true')
arg_parser.add_argument("--output",default="Feature_importances.png",
                        help='output image file')

parsed_args = arg_parser.parse_args()
# ------------------------------
# ------------------------------
# Reading the configuration file

file_not_found_message = """
Error: can not load the configuration file {:s}.
Please check that the file exists and is of YAML or JSON format.
Exiting.
"""

try:
    config = yaml.safe_load(open(parsed_args.config, "r"))
except IOError:
    print(file_not_found_message.format(parsed_args.config))
    exit()

if parsed_args.stereo:
    is_stereo = True
else:
    is_stereo = False
output_file=parsed_args.output
energy_estimator = EnergyEstimatorPandas(config['energy_rf']['features'],config['energy_rf']['training_conditions'],
                                         **config['energy_rf']['settings'])
energy_estimator.load(config['energy_rf']['save_name'])

#DIRECTION
magic_optics = OpticsDescription.from_name('MAGIC')
magic_cam = CameraDescription.from_name('MAGICCam')
magic_tel_description = TelescopeDescription(name='MAGIC',
                                             tel_type='MAGIC',
                                             optics=magic_optics,
                                             camera=magic_cam)
magic_tel_descriptions = {1: magic_tel_description,
                          2: magic_tel_description}
direction_estimator = DirectionEstimatorPandas(config['direction_rf']['features'],
                                               magic_tel_descriptions,config['energy_rf']['training_conditions'],
                                               **config['direction_rf']['settings'])

direction_estimator.load(config['direction_rf']['save_name'])

#CLASSIFIER
class_estimator = EventClassifierPandas(config['classifier_rf']['features'],config['energy_rf']['training_conditions'],
                                         **config['classifier_rf']['settings'])
class_estimator.load(config['classifier_rf']['save_name'])


if energy_estimator.single_RF:
    fig, axs = pyplot.subplots(1, 1, figsize=(15, 12))
    kind = 'energy'
    print(kind)
    df = pd.DataFrame()
    df['feature_importances'] = energy_estimator.telescope_regressors[1].feature_importances_
    df['features_names'] = config['energy_rf']['features']
    df = df.sort_values(by='feature_importances')
    df['features_num'] = np.arange(len(df))
    width = 0.3
    # print(f'  tel_id: {tel_id}')
    # for feature, importance in zip(energy_estimator.feature_names, feature_importances):
    # print(f"  {feature:.<15s}: {importance:.4f}")
    # print('')
    # print(features_names)
    axs.barh(df['features_num'], df['feature_importances'], width, align='center', color='blue', label='M1')

    axs.set_yticks(df['features_num'])
    axs.set_yticklabels(df['features_names'], fontsize=20)
    pyplot.xticks(fontsize=20)
    axs.set_xlabel('Importance', fontsize=20)
    axs.set_title(f'{kind} RF feature importances (medium Zenith)', fontsize=20)
    axs.grid()
else:

    kind_list = ['energy','event_class']
    fig, axs = pyplot.subplots(1, len(kind_list), figsize=(15, 7))
    for i,kind in enumerate(kind_list):
        df=pd.DataFrame()
        for tel_id in energy_estimator.telescope_regressors:
            if kind=='energy':
                df['feature_importances'] = energy_estimator.telescope_regressors[tel_id].feature_importances_
                df['features_names'] =config['energy_rf']['features']

            elif kind=='event_class':
                df['feature_importances'] = class_estimator.telescope_classifiers[tel_id].feature_importances_
                df['features_names'] = config['classifier_rf']['features']
            else:
                df['feature_importances'] = direction_estimator.telescope_rfs[kind][tel_id].feature_importances_
                df['features_names']=config['direction_rf']['features'][kind]


            df=df.sort_values(by='feature_importances')
            df['features_num'] = np.arange(len(df))
            width=0.3
            if tel_id==1:
                axs[i].barh(df['features_num'],df['feature_importances'],width,align='center',color='blue',label='M1')

            elif tel_id==2:
                axs[i].barh(df['features_num']+width,df['feature_importances'],width,align='center',color='red',label='M2')


        axs[i].set_yticks(df['features_num'])
        axs[i].set_yticklabels(df['features_names'])
        axs[i].set_xlabel('Importance')
        axs[i].set_title(f'{kind} RF feature importances (medium Zenith)')
        axs[i].legend()
        axs[i].grid()
        #axs[i].text(0.05,3.1,'MCP parametrization\nstarting from MARS Calibrated files',fontsize=16)
fig.show()
fig.savefig(output_file,bbox_inches='tight')

