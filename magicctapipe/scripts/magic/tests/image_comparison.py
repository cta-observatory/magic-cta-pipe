#!/usr/bin/env python
# coding: utf-8

import argparse
import yaml
import pandas as pd
import h5py
import uproot
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import astropy.units as u
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from ctapipe_io_magic import MAGICEventSource
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry, CameraDescription
from ctapipe.image import hillas_parameters
from ctapipe.image.timing import timing_parameters
from ctapipe.containers import HillasParametersContainer
from magicctapipe.utils import MAGIC_Badpixels, MAGIC_Cleaning  
from astropy.coordinates import Angle
from ctapipe.image.morphology import number_of_islands
from magicctapipe.scripts import read_images, ImageContainerCleaned, ImageContainerCalibrated

#define camera geometry
def new_camera_geometry(camera_geom):
	return CameraGeometry(
		camera_name="MAGICCam",
		pix_id=camera_geom.pix_id,
		pix_x=-1.*camera_geom.pix_y,
		pix_y=-1.*camera_geom.pix_x,
		pix_area=camera_geom.guess_pixel_area(camera_geom.pix_x, camera_geom.pix_y, camera_geom.pix_type),
		pix_type=camera_geom.pix_type,
		pix_rotation=camera_geom.pix_rotation,
		cam_rotation=camera_geom.cam_rotation
	)

#set configurations for the cleaning
cleaning_config = dict(
	picture_thresh = 6,
	boundary_thresh = 3.5,
	max_time_off = 4.5 * 1.64,
	max_time_diff = 1.5 * 1.64,
	usetime = True,
	usesum = True,
	findhotpixels = True,
)

bad_pixels_config = dict(
	pedestalLevel = 400,
	pedestalLevelVariance = 4.5,
	pedestalType = "FromExtractorRndm"
)

# define the image comparison function
def image_comparison(config_file = "config.yaml", mode = "use_ids_config", tel_id=1): 
	"""
	This tool compares the camera images of events processed by MARS and the magic-cta-pipeline. 
	The output is a png file with the camera images and a hdf5 file that contains the pixel information.
	---
	config_file: Configuration file
	mode: 
		use_all: use all events in the given input file
		use_ids_config: use events, whose ids are given in the config file
	tel_id: Telescope of which the data will be compared. use "1" for M1 and "2" for M2
	"""


	# load config file--------------------------------------------------------------------------------------------
	config = yaml.safe_load(open(config_file, "r"))
	run_num = config["information"]["run_number"]
	out_path = config["output_files"]["file_path"]
	comparison = []

	# get id to compare from config file---------------------------------------------------------------------------
	if mode == "useall":
		with uproot.open(config["input_files"]["magic_cta_pipe"][f"M{tel_id}"]) as mcp_file:
			ids_to_compare = mcp_file["Events"]["MRawEvtHeader./MRawEvtHeader.fStereoEvtNumber"].array(library="np")
			ids_to_compare = ids_to_compare["event_id"].tolist()
			# we could leave it as an array probably
	elif mode == "use_ids_config":
		ids_to_compare = config["event_list"]
 
	print(len(ids_to_compare), "events will be compared", ids_to_compare)   

	# we will now load the data files, and afterwards select the corresponding data for our events
	# get mars data ------------------------------------------------------------------------------------------------------
	mars_input = config["input_files"]["mars"]
	image = []	  	#pixel charges
	events=[]	  	#event ids
	telescope=[]	#telescope id
	obs_id=[]		#run number
	
	for image_container in read_images(mars_input):
		events.append(image_container.event_id)
		telescope.append(image_container.tel_id)
		obs_id.append(image_container.obs_id)
		image.append(image_container.image_cleaned)
		#this contains both M1 and M2 data!! the first half of the lists corresponds to M1 and the sceond half to M2

	mars_data = pd.DataFrame(list(zip(events, telescope, obs_id, image)), columns=["event_id", "tel_id", "obs_id", "image"])
	#mars_data.set_index("event_id", inplace=True)
	

	# get original data-----------------------------------------------------------------------------------------------
	data_path_original = config["input_files"]["magic_cta_pipe"][f"M{tel_id}"]
	with uproot.open(data_path_original) as input_data:
		event_ids_original = input_data["Events"]["MRawEvtHeader.fStereoEvtNumber"].array(library="np")
		images_original = input_data["Events"]["MCerPhotEvt.fPixels.fPhot"].array(library="np")


	# get other data----------------------------------------------------------------------------------------------------
	# we loop through the events, and only compare those that are in ids_to_compare
	
	f=h5py.File(f"Image_comparison_{run_num}.h5", "a")


	source = MAGICEventSource(input_url=config["input_files"]["magic_cta_pipe"][f"M{tel_id}"])
	# im folgenden ist event.index.event_id die Event ID
	for event in source:
		if event.index.event_id not in ids_to_compare:
			continue
		else:
			print("Event ID:", event.index.event_id, "- Telecope ID:", tel_id)


			# mars data -------------------------------
			MAGICCAM = CameraDescription.from_name("MAGICCam")
			GEOM = MAGICCAM.geometry
			geometry_mars = new_camera_geometry(GEOM)

			mars_event = mars_data.loc[(mars_data["event_id"]==event.index.event_id)&(mars_data["tel_id"]==tel_id)]
			
			event_image_mars = np.array(mars_event["image"][:1039])
			event_image_mars = event_image_mars[:1039]
			clean_mask_mars = event_image_mars != 0
			print(np.where(event_image_mars[0] != 0))
			
			
			#compare number of islands
			# do we need this? 
			# num_islands_mars, island_labels_mars = number_of_islands(geometry_mars, (np.array(event_image[:1039]) > 0))



			#original data (calibrated data before cleaning) -------------------------------------------------------------------------------

			event_index_array = np.where(event_ids_original == event.index.event_id)
			event_index = event_index_array[0][0]
			calibrated_data_images = images_original[event_index][:1039]

			# get mcp data-------------------------------------------------------------------------------------------------------
			tel = source.subarray.tel 		
			
			r0tel = event.r0.tel[tel_id]
			geometry_old = tel[tel_id].camera.geometry
			geometry_mcp = new_camera_geometry(geometry_old)
			magic_clean = MAGIC_Cleaning.magic_clean(geometry_mcp,cleaning_config)
			badpixel_calculator = MAGIC_Badpixels.MAGICBadPixelsCalc(config=bad_pixels_config, is_simulation=source.is_mc)
			event_image = event.dl1.tel[tel_id].image
			event_pulse_time = event.dl1.tel[tel_id].peak_time


			badrmspixel_indices = [[None],[None]]

			badrmspixel_mask = badpixel_calculator.get_badrmspixel_mask(event)

			deadpixel_mask = badpixel_calculator.get_deadpixel_mask(event)
			unsuitable_mask = np.logical_or(badrmspixel_mask[tel_id-1], deadpixel_mask[tel_id-1])
			bad_pixel_indices = [i for i, x in enumerate(badrmspixel_mask[tel_id-1]) if x]
			dead_pixel_indices = [i for i, x in enumerate(deadpixel_mask[tel_id-1]) if x]
			bad_not_dead_pixels_test = [i for i in bad_pixel_indices if i not in dead_pixel_indices]


			clean_mask, event_image, event_pulse_time = magic_clean.clean_image(event_image, event_pulse_time, unsuitable_mask=unsuitable_mask)

			event_image_cleaned = event_image.copy()
			event_image_cleaned[~clean_mask] = 0
			print(event_image_cleaned)


			event_pulse_time_cleaned = event_pulse_time.copy()
			event_pulse_time_cleaned[~clean_mask] = 0
			num_islands_mcp, island_labels_mcp = number_of_islands(geometry_mcp, (np.array(event_image_cleaned[:1039]) > 0))

			if np.any(event_image_cleaned):
			 	hillas_params = hillas_parameters(geometry_mcp, event_image_cleaned)
			 	image_mask = event_image_cleaned > 0
			 	timing_params = timing_parameters(
			 	geometry_mcp,
			 	event_image_cleaned,
			 	event_pulse_time_cleaned,
			 	hillas_params,
			 	image_mask)
			 	print(timing_params.slope)
			else:
				print("Cleaning failed.")
				continue


			# get max value for colorbar--------------------------------------------------------------------------------
			mcp_max = np.amax(event_image_cleaned[clean_mask])
			mars_max = np.amax(event_image[clean_mask])
			if mcp_max >= mars_max:
				vmax = mcp_max
			else:
				vmax = mars_max

			vmin = 0
			print(vmax)

			# find differences------------------------------------------------------------------------------------------
			charge_differences = abs(event_image_mars[0][:1039] - event_image_cleaned)
			clean_mask_pixels = charge_differences != 0


			# for some reason it raises an error for all events if we set the threshold to 0 percent. according to the images there are no
			# differences for 3 events? what is going on? TODO!
		
			errors = False
			charge_differences_list = charge_differences.tolist()
			for i in charge_differences_list:
				if i >= 0.00000001*vmax: 		#threshold for differences allowed. The difference cannot be higher than x% of the highest pixel charge
					errors = True

			if errors:
				comparison.append(event.index.event_id)
				print(f"errors found for {event.index.event_id}!")
	
			# differences between calibrated data before cleaning and mcp/mars data
			pix_diff_mars = abs(event_image_mars[0][:1039] - calibrated_data_images)
			pix_diff_mcp = abs(event_image_cleaned - calibrated_data_images)

			#create output file that contains the pixel values and differences------------------------------------------------
			# BEI DIESEM TEIL IST ES NOCH NICHT KLAR OB ER FUNKTIONIERT
			#data_value =[calibrated_data_images[i] for i in range(1039)]
			"""
			# for i in range(1039):
			# 	data_value.append(calibrated_data_images[i])


			df_pixel = pd.DataFrame()
			data = np.transpose(np.array([event_image_mars[0][:1039], event_image_cleaned, calibrated_data_images]))
			#data_real = np.transpose(data)
			
			df_pixel = pd.DataFrame(data, columns=["MARS_charge", "mcp_charge", "calibrated_data"])
			#df_pixel["calibrated data"] = np.transpose(data_value)#(calibrated_data_images[:1039])
			df_pixel["difference_MARS-mcp"] = np.transpose(charge_differences)
			df_pixel["relative_error_MARS-mcp"] = np.transpose(charge_differences/event_image_cleaned)
			# alternatively one could also calculate the relative error with the MARS charge value
			# df_pixel["relative_error_MARS-mcp"] = np.transpose(charge_differences/event_image_mars[0][:1039])
			df_pix_diff = df_pixel.loc[df_pixel["difference_MARS-mcp"] > 0]
			#pix_diff_ids = df_pix_diff.index.tolist()
			print(df_pix_diff)
			
			#saving the output
			if config["save_only_when_differences"] == True:
				#the file only gets saved if there are differences between the images
				if any(charge_differences) == True: 
					print("Differences found. Saving files!")
					df_pixel.to_hdf(f"{out_path}/{run_num}_{event.index.event_id}_M{tel_id}_pixel.h5", "/pixel_differences", "w")
					#df_pix_diff.to_hdf(f"{out_path}{run_num}_{id_event}_M{telescope_id}_pixel_diff.h5", "/pixel_differences", "w")
					#print(pix_diff_ids)
			
				else:
					print("No differences found. No files will be saved!")
					continue
			
			elif config["save_only_when_differences"] == False:
				#the file gets saved in any case
				df_pixel.to_hdf(f"{out_path}/{run_num}_{event.index.event_id}_M{tel_id}_pixel_info.h5", "/pixel_information", "w")
				df_pix_diff.to_hdf(f"{out_path}/{run_num}_{event.index.event_id}_M{tel_id}_pixel_diff.h5", "/pixel_differences", "w")
				#print(pix_diff_ids)
			else:
				print("No criteria for saving data specified. Exiting.")
				#exit()
				"""
			# alternatively --------------------------------------------------------------------------------
			
			# .h5 file is created outside of the loop
			df_pixel = pd.DataFrame()
			data = np.transpose(np.array([event_image_mars[0][:1039], event_image_cleaned, calibrated_data_images]))
			
			df_pixel = pd.DataFrame(data, columns=["MARS_charge", "mcp_charge", "calibrated_data"])
			df_pixel["difference_MARS-mcp"] = np.transpose(charge_differences)
			df_pixel["relative_error_MARS-mcp"] = np.transpose(charge_differences/event_image_cleaned)
			# alternatively one could also calculate the relative error with the MARS charge value
			# df_pixel["relative_error_MARS-mcp"] = np.transpose(charge_differences/event_image_mars[0][:1039])
			df_pix_diff = df_pixel.loc[df_pixel["difference_MARS-mcp"] > 0]
			#pix_diff_ids = df_pix_diff.index.tolist()
			print(df_pix_diff)
			
			#saving the output
			if config["save_only_when_differences"] == True:
				#the file only gets saved if there are differences between the images
				if any(charge_differences) == True: 
					#print("Differences found. Saving files!")
					df_pixel.to_hdf(f"Image_comparison_{run_num}.h5", f"/{event.index.event_id}", "a")
					#df_pix_diff.to_hdf(f"{out_path}{run_num}_{id_event}_M{telescope_id}_pixel_diff.h5", "/pixel_differences", "w")
					#print(pix_diff_ids)

			# I dont know if we need that one....	
			elif config["save_only_when_differences"] == False:
				#the file gets saved in any case
				df_pixel.to_hdf(f"Image_comparison_{run_num}.h5", f"/{event.index.event_id}", "a")
				#print(pix_diff_ids)

			
			# plotting ------------------------------------------------------------------------------------------------------
			fig = plt.figure(figsize=(20, 10))
			grid_shape = (2, 3)
		
			#original data
			plt.subplot2grid(grid_shape, (0, 0))
			geom = CameraGeometry.from_name("MAGICCamMars")
			disp = CameraDisplay(geom, calibrated_data_images)
			# pixels whose original value is negative
			# negative_mask = calibrated_data_images < 0
			# disp.highlight_pixels(negative_mask, color="white", alpha=1, linewidth=1)
			disp.add_colorbar(label="pixel charge")
			disp.set_limits_minmax(vmin, vmax)
			plt.title("original data")
				
			#mars_data
			plt.subplot2grid(grid_shape, (0, 1))
			norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
			disp_mars = CameraDisplay(geometry_mars, image=event_image_mars[0][:1039])
			#disp_mars.highlight_pixels(clean_mask, color="white", alpha=0.5, linewidth=1)
			disp_mars.set_limits_minmax(vmin, vmax)
			disp_mars.add_colorbar(label="pixel charge")
			disp_mars.highlight_pixels(clean_mask_pixels[:1039], color="red", alpha=1, linewidth=1)		
			plt.title("MARS data")

			#mcp_data
			plt.subplot2grid(grid_shape, (0, 2))		
			disp_mcp = CameraDisplay(geometry_mcp, image=event_image_cleaned)
			#disp_mcp.highlight_pixels(clean_mask, color="white", alpha=0.5, linewidth=1) 
			disp_mcp.add_colorbar(label="pixel charge")
			disp_mcp.set_limits_minmax(vmin, vmax)
			disp_mcp.highlight_pixels(clean_mask_pixels[:1039], color="red", alpha=1, linewidth=1)
			plt.title("magic_cta_pipe data")
		
			#differences between MARS and mcp
			plt.subplot2grid(grid_shape, (1, 0))		
			disp = CameraDisplay(geom, charge_differences[:1039])
			disp.add_colorbar(label="pixel charge")
			disp.highlight_pixels(clean_mask_pixels[:1039], color="red", alpha=1, linewidth=1)
			plt.title("differences MARS-mcp")  
			
			# the white outline showes the pixels used for the image after cleaning, the ones that are filled yellow show where differences are
			#differences between MARS and the calibrated data
			plt.subplot2grid(grid_shape, (1, 1))
			pix_diff_mars_copy = np.array(pix_diff_mars).copy()
			pix_diff_mars_copy[np.array(event_image_mars[0][:1039]) == 0] = 0
			disp = CameraDisplay(geom, pix_diff_mars_copy>0)
			disp.highlight_pixels(np.array(event_image_mars[0][:1039]) != 0, color="white", alpha=1, linewidth=3)
			#disp.set_limits_minmax(vmin, vmax)
			#disp.add_colorbar(label="pixel charge")
			plt.title("differences MARS-original data")
		
			#mcp-orig-data-diff
			plt.subplot2grid(grid_shape, (1, 2))
			pix_diff_mcp_copy = np.array(pix_diff_mcp).copy()
			pix_diff_mcp_copy[np.array(event_image_cleaned) == 0] = 0
			disp = CameraDisplay(geom, pix_diff_mcp_copy>0)
			disp.highlight_pixels(np.array(event_image_cleaned) != 0, color="white", alpha=1, linewidth=3)			   
			#disp.set_limits_minmax(vmin, vmax)
			#disp.add_colorbar(label="pixel charge")
			plt.title("differences mcp-original data")
			
			
			fig.suptitle(f"Comparison_MARS_magic-cta-pipe: Event ID {event.index.event_id}, {run_num}, M{tel_id}", fontsize=16)
			fig.savefig(f"{out_path}/image-comparison-{run_num}_{event.index.event_id}_M{tel_id}_ver2.pdf")
			# fig.savefig(f"{out_path}/image-comparison-{run_num}_{event.index.event_id}_M{tel_id}.png")

		# calculating relative errors
		# writing output h5 file
	return comparison


print(image_comparison(config_file = "image_comparison_config.yaml", mode = "use_ids_config", tel_id=2))