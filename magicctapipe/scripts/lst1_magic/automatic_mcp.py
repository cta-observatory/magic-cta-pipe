#!/usr/bin/env python
# coding: utf-8

"""
This script runs the whole magic-cta-pipeline atomatically.
"""

import glob
import time
import logging
import multiprocessing
import os
import yaml

from pathlib import Path
from functools import partial
from subprocess import Popen, PIPE

# TODO 
# do we need the times??yes
# option for conig.yaml with different name
# divide mcs into test and train
# mc dl0 to dl1 ?
# class fuer stereo recos
# try except fuer die einzelnen schritte?
# maybe make merging the runs optional?

def calib_to_dl1(output_path, file):
	date = file[0:10]
	os.system(f"python magic_calib_to_dl1.py --input-file {file} --output-dir {output_path}/{date}/dl1 --config-file ./config.yaml")

def stereo_reco(type, output_path, file, use_magic_only=True):
	if use_magic_only == True:
		magic_only = "--magic-only"
	else:
		magic_only = ""

	if type == "data":
		date = file[-40:-30]
		os.system(f"python lst1_magic_stereo_reco.py --input-file {file} --output-dir {output_path}/{date}/dl1_stereo --config-file ./config.yaml \
		{magic_only}")
	elif type == "mc_test":
		os.system(f"python lst1_magic_stereo_reco.py --input-file {file} --output-dir {output_path}/MCs/gamma_off0.4deg/dl1_stereo/test \
		 --config-file ./config.yaml {magic_only}")	
	elif type == "mc_train":
		os.system(f"python lst1_magic_stereo_reco.py --input-file {file} --output-dir {output_path}/MCs/gamma_off0.4deg/dl1_stereo/train \
		 --config-file ./config.yaml {magic_only}")	
	elif type == "proton":
		os.system(f"python lst1_magic_stereo_reco.py --input-file {file} --output-dir {output_path}/MCs/proton/dl1_stereo \
		 --config-file ./config.yaml {magic_only}")	

def train_rfs(output_path, rf):
	os.system(f"python lst1_magic_train_rfs.py --input-file-gamma {output_path}/MCs/gamma_off0.4deg/dl1_stereo/train/merged/*.h5 \
		--input-file-bkg {output_path}/MCs/proton/dl1_stereo/merged/*.h5 \
		--output-dir {output_path}/MCs/RFs --config-file ./config.yaml --train-{rf}")

def dl1_stereo_to_dl2(output_path, date, file):
	rf_path = f"{output_path}/MCs/RFs"
	os.system(f"python lst1_magic_dl1_stereo_to_dl2.py --input-file-dl1 {file} --input-dir-rfs {rf_path} --output-dir {output_path}/{date}/dl2")

def dl2_to_dl3(output_path, date, file):
	irf_path = f"{output_path}/MCs/IRF/*.gz"
	os.system(f"python lst1_magic_dl2_to_dl3.py --input-file-dl2 {file} --input-file-irf {irf_path} --output-dir {output_path}/{date}/dl3 \
		--config-file ./config.yaml")

def main():
	logger = logging.getLogger(__name__)
	logger.addHandler(logging.StreamHandler())
	logger.setLevel(logging.INFO)
	
	config_file = Path("./automatic_mcp_config.yaml").absolute()
	with open(config_file, 'rb') as f:
		config = yaml.safe_load(f)

	dates = config["Data"]["dates"]
	runnumbers = config["Data"]["run-numbers"]
	input_path = config["Data"]["input-path"]	
	output_path = config["Data"]["output-path"]
	input_path_mc = config["MCs"]["input-path"]
	print(dates)
	start_time = time.time()
	dates = config["Data"]["dates"]

	#=================================
	# create folders (sorted by date) 
	#=================================
	"""
	data_dirs = [[date+"/dl1", date+"/dl1_stereo", date+"/dl2", date+"/dl3"] for date in dates]
	all_data_dirs = [str(i) for sublist in data_dirs for i in sublist]

	mc_directories = ["MCs/gamma_off0.4deg/dl1/test", "MCs/gamma_off0.4deg/dl1/train", "MCs/gamma_off0.4deg/dl1_stereo/test", \
		"MCs/gamma_off0.4deg/dl1_stereo/train", "MCs/gamma_off0.4deg/dl2","MCs/proton/dl1", "MCs/proton/dl1_stereo", \
		"MCs/proton/dl2", "MCs/IRF", "MCs/RFs"]

	mc_dirs = [os.makedirs(os.path.join(output_path, mc_directory), exist_ok=True) for mc_directory in mc_directories]
	data_dirs = [os.makedirs(os.path.join(output_path, directory), exist_ok=True) for directory in all_data_dirs]

	logger.info("\nDirectories created.")

	#============
	# dl0 -> dl1
	#============
	#TODO change mask before running the final test

	input_file_masks = [input_path+date+"/M*/*_M*_"+runnum+".*_Y_CrabNebula-W0.*.root" for runnum in runnumbers for date in dates]
	input_files = [glob.glob(input_file_mask) for input_file_mask in input_file_masks]	
	input_files = [str(i) for sublist in input_files for i in sublist]
	input_files.sort()
	logger.info(f'\nFiles: {input_files}')

	pool_dl1 = multiprocessing.Pool()
	pool_dl1.map(partial(calib_to_dl1, output_path), input_files)
	pool_dl1.close()
	pool_dl1.join()
	"""
	logger.info('\nCalibrated Data to dl1: Done.')
	calib_to_dl1_time = time.time() - start_time
	logger.info(f'\nProcess time: {calib_to_dl1_time:.0f} [sec]\n')

	#======================
	# merge I (subrunwise)
	#======================
	# TODO gives error, but still works??

	for date in dates:
		p_merge1 = Popen(f"python merge_hdf_files.py --input-dir {output_path}/{date}/dl1 --subrun-wise", shell=True)
		p_merge1.wait()
		
	logger.info('\nSubrun files from M1 and M2 merged.')
	
	#=======================
	# stereo reconstruction
	#=======================
	#ATTENTION: stereo reco does not work at the moment. 
	#Solved, but atm just locally!
	"""
	stereo_masks = [f"{output_path}/{date}/dl1/merged/*.h5" for date in dates]
	stereo_data_files = [glob.glob(stereo_mask) for stereo_mask in stereo_masks]
	stereo_data_files = [str(i) for sublist in stereo_data_files for i in sublist]

	stereo_mc_files_test = glob.glob(f"{input_path_mc}/gamma_off0.4deg/dl1/test/*_run*.h5")
	stereo_mc_files_train = glob.glob(f"{input_path_mc}/gamma_off0.4deg/dl1/train/*_run*.h5")
	stereo_proton_files = glob.glob(f"{input_path_mc}/proton/dl1/*_*.h5")

	#TODO koennte man hier eine class oder funktion einfuerhen???
	pool_mc_stereo_train = multiprocessing.Pool()
	pool_mc_stereo_train.map(partial(stereo_reco, "mc_train", output_path), stereo_mc_files_train)
	pool_mc_stereo_train.close()

	pool_mc_stereo_test = multiprocessing.Pool()
	pool_mc_stereo_test.map(partial(stereo_reco, "mc_test", output_path), stereo_mc_files_test)
	pool_mc_stereo_test.close()

	pool_proton	= multiprocessing.Pool()
	pool_proton.map(partial(stereo_reco, "proton", output_path), stereo_proton_files)
	pool_proton.close()

	pool_data_stereo = multiprocessing.Pool()
	pool_data_stereo.map(partial(stereo_reco, "data", output_path), stereo_data_files)
	pool_data_stereo.close()

	pool_mc_stereo_train.join()
	pool_mc_stereo_test.join()
	pool_proton.join()
	pool_data_stereo.join()

	logger.info('\nStereo Reconstruction: Done.')
	stereo_reco_time = time.time() - calib_to_dl1_time
	logger.info(f'\nProcess time: {stereo_reco_time:.0f} [sec]\n')

	#====================
	# merge II (runwise)
	#====================

	for date in dates:
		p_merge2 = Popen(f"python merge_hdf_files.py --input-dir {output_path}/{date}/dl1_stereo --run-wise", shell=True)
		p_merge2.wait()
		logger.info('\nSubrun files for each run merged.')

	p_merge3 = Popen(f"python merge_hdf_files.py --input-dir {output_path}/MCs/gamma_off0.4deg/dl1_stereo/test", shell=True)
	p_merge4 = Popen(f"python merge_hdf_files.py --input-dir {output_path}/MCs/gamma_off0.4deg/dl1_stereo/train", shell=True)
	
	p_merge3.wait()
	p_merge4.wait()
	logger.info('\nMC files merged.')
	
	#===========
	# train RFs
	#===========
	#ATTENTION: this part raises an error, if old data is used. 
	# wurde noch nicht ausprobiert!

	rfs = config["Settings"]["RFs"]

	pool_rf = multiprocessing.Pool()
	pool_rf.map(partial(train_rfs, output_path), rfs)
	pool_rf.close()
	pool_rf.join()

	logger.info('\nTrain RFs: Done.')
	rf_time = time.time() - stereo_reco_time
	logger.info(f'\nProcess time: {rf_time:.0f} [sec]\n')

	#===================
	# dl1_stereo -> dl2
	#===================

	dl2_mc_files = glob.glob(f"{output_path}/MCs/gamma_off0.4deg/dl1_stereo/test/merged/*.h5")
	dl2_proton_files = glob.glob(f"{output_path}/MCs/proton/dl1_stereo")

	dl2_masks = [f"{output_path}/{date}/dl1_stereo/merged/*.h5" for date in dates]
	dl2_data_files = [glob.glob(dl2_mask) for dl2_mask in dl2_masks]
	dl2_data_files = [str(i) for sublist in dl2_data_files for i in sublist]

	pool_dl2_data = multiprocessing.Pool()
	pool_dl2_data.map(partial(dl1_stereo_to_dl2, output_path, date), dl2_data_files)
	pool_dl2_data.close()

	pool_dl2_mc = multiprocessing.Pool()
	pool_dl2_mc.map(partial(dl1_stereo_to_dl2, output_path, date), dl2_mc_files)
	pool_dl2_mc.close()

	pool_dl2_proton = multiprocessing.Pool()
	pool_dl2_proton.map(partial(dl1_stereo_to_dl2, output_path, date), dl2_proton_files)
	pool_dl2_proton.close()
	
	pool_dl2_data.join()
	pool_dl2_mc.join()
	pool_dl2_proton.join()

	logger.info('\ndl1_stereo to dl2: Done.')
	dl2_time = time.time() - rf_time
	logger.info(f'\nProcess time: {dl2_time:.0f} [sec]\n')

	#============
	# create IRF
	#============

	p_irf = Popen(f"python lst1_magic_create_irf.py --input-file-gamma {output_path}MCs/gamma_off0.4deg/dl2/dl2_*.h5 \
		--output-dir {output_path}/MCs/gamma_off0.4deg/IRF --config-file ./config.yaml \
		--input-file-proton {output_path}/MCs/proton/dl2/*.h5", shell=True)
	p_irf.wait()

	logger.info('\nIRF created sucessfully.')
	irf_time = time.time() - dl2_time
	logger.info(f'\nProcess time: {irf_time:.0f} [sec]\n')

	#============
	# dl2 -> dl3
	#============

	dl3_masks = [f"{output_path}/{date}/dl2/*.h5" for date in dates]
	dl3_files = [glob.glob(dl3_mask) for dl3_mask in dl3_masks]
	dl3_files = [str(i) for sublist in dl3_files for i in sublist]

	pool_dl3 = multiprocessing.Pool()
	pool_dl3.map(partial(dl2_to_dl3, output_path), dl3_files)
	pool_dl3.close()
	pool_dl3.wait()

	logger.info('\ndl2_ to dl3: Done.')
	dl3_time = time.time() - irf_time
	logger.info(f'\nProcess time: {dl3_time:.0f} [sec]\n')

	#=============
	# dl3_index
	#=============

	for date in dates:
		p_index = Popen(f"python create_dl3_index_files.py --input-dir {output_path}/{date}/dl3", shell=True)
		p_index.wait()
		logger.info('\ndl3_index created.')
	"""
	end_time = time.time() - start_time
	logger.info(f'\nProcess time: {end_time:.0f} [sec]\n')

if __name__ == '__main__':
	main()
