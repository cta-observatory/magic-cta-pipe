#!/usr/bin/env python
# coding: utf-8

import argparse
import yaml
import pandas as pd
import h5py
import uproot
from ctapipe_io_magic import MAGICEventSource
from ctapipe.visualization import CameraDisplay
from ctapipe.instrument import CameraGeometry
from ctapipe.image import hillas_parameters
from ctapipe.image.timing import timing_parameters
from ctapipe.containers import HillasParametersContainer
from ctapipe.instrument import CameraDescription
from utils import MAGIC_Badpixels  
from utils import MAGIC_Cleaning
import astropy.units as u
from astropy.coordinates import Angle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from ctapipe.image.morphology import number_of_islands
from mars_images_to_hdf5 import read_images, ImageContainerCleaned

#----------------------------------------------------------------

#create a list of the events you want to be compared
ids_to_compare = []

#add argparser arguments
arg_parser = argparse.ArgumentParser(description="""
This tool compares the images of events processed by MARS and the magic-cta-pipeline. 
The output is a png file with the camera images and a hdf5 file that contains the pixel information.
""")

arg_parser.add_argument("--config", default="config.yaml",
                        help='Configuration file to steer the code execution.')
arg_parser.add_argument("--useall",
                        help='Compare images for all events in the input file',
                        action='store_true')
arg_parser.add_argument("--use_ids_config",
                        help='Compare images for certain events',
                        action='store_true')

parsed_args = arg_parser.parse_args()

file_not_found_message = """
Error: can not load the configuration file {:s}.
Please check that the file exists and is of YAML or JSON format.
Exiting.
"""


#load get_images_config file
try:
    config = yaml.safe_load(open(parsed_args.config, "r"))
except IOError:
    print(file_not_found_message.format(parsed_args.config))
    exit()
#check for the important keys
if 'input_files' not in config:
    print('Error: the configuration file is missing the "input_files" section. Exiting.')
    exit()
    
if 'output_files' not in config:
    print('Error: the configuration file is missing the "output_files" section. Exiting.')
    exit() 


#---------------------------------------------------------
#    Compare images for all events in the given file
#---------------------------------------------------------
if parsed_args.useall:
    mcp_input = config['input_files']['magic-cta-pipe']
    tel_id = config['information']['tel_id']
    #does only work for stereo files
    mcp_file = uproot.open(config['input_files']['magic-cta-pipe'])
    #print(mcp_file.keys())
    if 'Events' not in mcp_file.keys():
        print("Missing key: Events. Unaccebtable file format. Exiting.")
        exit()
    else:
        if 'MRawEvtHeader./MRawEvtHeader.fStereoEvtNumber' not in mcp_file['Events'].keys():
            print(f"Missing key: MRawEvtHeader_{tel_id}.fStereoEvtNumber. Unaccebtable file format. Exiting.")
            exit()
        else:
            df_mcp_data = pd.DataFrame()
            df_mcp_data['event_id'] = mcp_file['Events']['MRawEvtHeader./MRawEvtHeader.fStereoEvtNumber'].array().to_numpy()
            ids_to_compare = df_mcp_data['event_id'].tolist()
            #the keys depend on the file, there may be a case where they have to be changed
#--------------------------------------------------------- 
#       Compare images for only certain events
#---------------------------------------------------------
    
elif parsed_args.use_ids_config:
     ids_to_compare = config['event_list']

else:
    print("No comparing argument specified. Exiting.")
    exit()
    
    
#print the list
if len(ids_to_compare) >= 21:
    print('EVENTS:', ids_to_compare[:10],'...',ids_to_compare[-10:])
else:
    print('EVENTS:', ids_to_compare)
    
print(len(ids_to_compare))   

#------------------------------------------------------------
#define camera geometry
def new_camera_geometry(camera_geom):
    """Scale given camera geometry of a given (constant) factor

    Parameters
    ----------
    camera : CameraGeometry
        Camera geometry
    factor : float
        Scale factor

    Returns
    -------
    CameraGeometry
        Scaled camera geometry
    """

    return CameraGeometry(
        camera_name='MAGICCam',
        pix_id=camera_geom.pix_id,
        pix_x=-1.*camera_geom.pix_y,
        pix_y=-1.*camera_geom.pix_x,
        pix_area=camera_geom.guess_pixel_area(camera_geom.pix_x, camera_geom.pix_y, camera_geom.pix_type),
        pix_type=camera_geom.pix_type,
        pix_rotation=camera_geom.pix_rotation,
        cam_rotation=camera_geom.cam_rotation
    )

#get information from the config file
telescope_id = config['information']['tel_id']
run_num = config['information']['run_number']
out_path = config['output_files']['file_path']


#compare events one at a time
ids = []
for id_to_compare in ids_to_compare:
    ids.append(id_to_compare)
    print("EVENT ID:", ids)
    for id_event in ids:
        
        #------------------
        #       MARS
        #-----------------

        mars_input = config['input_files']['mars']

        image = []      #pixel charges
        events=[]       #event ids
        telescope=[]    #telescope id
        runnum=[]       #run number
        
        for image_container in read_images(mars_input):
            events.append(image_container.event_id)
            telescope.append(image_container.tel_id)
            runnnum.append(image_container.obs_id)
            image.append(image_container.image)

        image = image.to_numpy()
        stereo_id = events.to_numpy()
		
        MAGICCAM = CameraDescription.from_name("MAGICCam")
        GEOM = MAGICCAM.geometry
        geometry_mars = new_camera_geometry(GEOM)


        #uncomment this list to compare different events
        #ids = [1962]
        
        for id in range(len(image)):
            if stereo_id[id] not in ids:
                continue
            event_image = np.array(image[id][:1039])
            clean_mask = event_image != 0
            print('Event ID:', stereo_id[id])
            print(event_image[clean_mask])
            #print(event_image)
            event_image_mars = event_image.copy()
            
            #compare numbe rof islands
            num_islands_mars, island_labels_mars = number_of_islands(geometry_mars, (np.array(event_image[:1039]) > 0))
            #print(num_islands_mars)


        #-----------------------------
        #        magic-cta-pipe
        #-----------------------------
        
        source = MAGICEventSource(input_url=config['input_files']['mcp_source'])

        cleaning_config = dict(
            picture_thresh = 6,
            boundary_thresh = 3.5,
            max_time_off = 4.5 * 1.64,
            max_time_diff = 1.5 * 1.64,
            usetime = True,
            usesum = True,
            findhotpixels = False,
        )

        bad_pixels_config = dict(
            pedestalLevel = 400,
            pedestalLevelVariance = 4.5,
            pedestalType = 'FromExtractorRndm'
        )

        
        #uncomment this list to compare different events
        #ids = [1962]
        
        for event in source._mono_event_generator(telescope=f'{telescope_id}'):
            if event.index.event_id not in ids:
                continue
            print(f"Event ID: {event.index.event_id}")
            tel = source.subarray.tel
            for tel_id in [telescope_id]:  
                print("Telescope ID:", tel_id)
                r0tel = event.r0.tel[tel_id]
                geometry_old = tel[tel_id].camera.geometry
                geometry_mcp = new_camera_geometry(geometry_old)
                magic_clean = MAGIC_Cleaning.magic_clean(geometry_mcp,cleaning_config)
                badpixel_calculator = MAGIC_Badpixels.MAGICBadPixelsCalc(config=bad_pixels_config)
                event_image = event.dl1.tel[tel_id].image
                event_pulse_time = event.dl1.tel[tel_id].peak_time

                badrmspixel_indices = [[None],[None]]
                
                badrmspixel_mask = badpixel_calculator.get_badrmspixel_mask(event)
                
                deadpixel_mask = badpixel_calculator.get_deadpixel_mask(event)
                unsuitable_mask = np.logical_or(badrmspixel_mask[tel_id-1], deadpixel_mask[tel_id-1])

                
                
                bad_pixel_indices = [i for i, x in enumerate(badrmspixel_mask[telescope_id-1]) if x]
                #print("badpixel_indices:", bad_pixel_indices)
                dead_pixel_indices = [i for i, x in enumerate(deadpixel_mask[telescope_id-1]) if x]
                #print("deadpixel_indices:", dead_pixel_indices)
                bad_not_dead_pixels = []
                for i in bad_pixel_indices:
                    if i not in dead_pixel_indices:
                        bad_not_dead_pixels.append(i)
                print("bad but not dead pixels:", bad_not_dead_pixels)
                #print(len(bad_pixel_indices))

                clean_mask, event_image, event_pulse_time = magic_clean.clean_image(event_image, event_pulse_time, unsuitable_mask=unsuitable_mask)

                event_image_cleaned = event_image.copy()
                event_image_cleaned[~clean_mask] = 0
                #print(event_image_cleaned[clean_mask])
                
                event_pulse_time_cleaned = event_pulse_time.copy()
                event_pulse_time_cleaned[~clean_mask] = 0
                
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
                
                #compare number of islands
                num_islands_mcp, island_labels_mcp = number_of_islands(geometry_mcp, (np.array(event_image_cleaned[:1039]) > 0))
                #print(num_islands_mcp)

  
        #------------------
        # original calibrated data
        #---------------------
        
        data_path = config['input_files']['original_data']
        with uproot.open(data_path) as input_data:
            event_ids = input_data['Events']['MRawEvtHeader.fStereoEvtNumber'].array()
            images = input_data['Events']['MCerPhotEvt.fPixels.fPhot'].array()
        
            event_index_array = np.where(event_ids == id_event)
            event_index = event_index_array[0][0]
            #print(event_index)
            calibrated_data_images = images[event_index][:1039]
            #print(np.array(images[event_index][:1039]).shape)
            #print(np.array(images).shape)
            #print(np.array(calibrated_data_images).shape)

            #print(len(data_images))
            
        #-------------------------------------------
        #get maximum pixel charge for the colorbar
        #-------------------------------------------
        
        mcp_max = np.amax(event_image_cleaned[clean_mask])
        mars_max = np.amax(event_image[clean_mask])
        if mcp_max >= mars_max:
            vmax = mcp_max
        else:
            vmax = mars_max
        print("maximum charge:", vmax)

        vmin = 0
       
        #---------------------------------------------
        # find pixel differences and write h5 output file
        #-------------------------------------------------
        
        pix_mars = np.array([event_image_mars])
        pix_mcp = np.array([event_image_cleaned])
        pix_diff = abs(pix_mars - pix_mcp)
        #if you want negative values instead of the absolutes
        #pix_diff = pix_mars - pix_mcp
        #pix_diff = abs(pix_mars - pix_mcp[0])
        pix_image = pix_diff[0]
        clean_mask_pixels = pix_image != 0
        
        #differences between outputs from MARS and mcp and input calibrated data
        pix_diff_mars = abs(pix_mars[0] - calibrated_data_images)
        pix_diff_mcp = abs(pix_mcp[0] - calibrated_data_images)
        print(pix_diff_mars)
        #print(calibrated_data_images)
        



        data_value =[]

        for i in range(1039):
            data_value.append(calibrated_data_images[i])


        df_pixel = pd.DataFrame()
        data = np.array([event_image_mars, event_image_cleaned])
        data_real = np.transpose(data)
        
        df_pixel = pd.DataFrame(data_real, columns=['MARS charge', 'mcp charge'])
        df_pixel['calibrated data'] = np.transpose(data_value)#(calibrated_data_images[:1039])
        df_pixel['difference MARS mcp'] = np.transpose(pix_diff)
        df_pix_diff = df_pixel.loc[df_pixel['difference MARS mcp'] > 0]
        pix_diff_ids = df_pix_diff.index.tolist()
        print(df_pix_diff)
        
        #saving the output
        if config['save_only_when_differences'] == True:
            #the file only gets saved if there are differences between the images
            if any(pix_image) == True: 
                print("Differences found. Saving files!")
                df_pixel.to_hdf(f'{out_path}{run_num}_{id_event}_M{telescope_id}_pixel_info.h5', "/pixel_information", 'w')
                df_pix_diff.to_hdf(f'{out_path}{run_num}_{id_event}_M{telescope_id}_pixel_diff.h5', "/pixel_differences", 'w')
                print(pix_diff_ids)
                
            else:
                print('No differences found. No files will be saved!')
                continue
                
        elif config['save_only_when_differences'] == False:
            #the file gets saved in any case
            df_pixel.to_hdf(f'{out_path}{run_num}_{id_event}_M{telescope_id}_pixel_info.h5', "/pixel_information", 'w')
            df_pix_diff.to_hdf(f'{out_path}{run_num}_{id_event}_M{telescope_id}_pixel_diff.h5', "/pixel_differences", 'w')
            print(pix_diff_ids)
        else:
            print("No criteria for saving data specified. Exiting.")
            exit()
        
	#--------------------------------
	#  creating output image
	#----------------------------
        fig = plt.figure(figsize=(20, 10))
        grid_shape = (2, 3)
	
	#original data
        plt.subplot2grid(grid_shape, (0, 0))
	
        geom = CameraGeometry.from_name('MAGICCamMars')
        disp = CameraDisplay(geom, calibrated_data_images)
        #pixels whose original value is negative
        negative_mask = calibrated_data_images < 0
        disp.highlight_pixels(negative_mask, color='white', alpha=1, linewidth=1)
        disp.add_colorbar(label='pixel charge')
        disp.set_limits_minmax(vmin, vmax)
	
        plt.title("original data")
	
	
        #mars_data
        plt.subplot2grid(grid_shape, (0, 1))
	
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        disp_mars = CameraDisplay(geometry_mars, event_image_mars[:1039])
        #disp_mars.highlight_pixels(clean_mask, color='white', alpha=0.5, linewidth=1)
        disp_mars.set_limits_minmax(vmin, vmax)
        disp_mars.add_colorbar(label='pixel charge')
        disp_mars.highlight_pixels(clean_mask_pixels, color='red', alpha=1, linewidth=1)
	
        plt.title("MARS data")


        #mcp_data
        plt.subplot2grid(grid_shape, (0, 2))
	
        disp_mcp = CameraDisplay(geometry_mcp, image=event_image_cleaned)
        #disp_mcp.highlight_pixels(clean_mask, color='white', alpha=0.5, linewidth=1) 
        disp_mcp.add_colorbar(label='pixel charge')
        disp_mcp.set_limits_minmax(vmin, vmax)
        disp_mcp.highlight_pixels(clean_mask_pixels, color='red', alpha=1, linewidth=1)
	
        plt.title("magic_cta_pipe data")
	
	#differences between MARS and mcp
        plt.subplot2grid(grid_shape, (1, 0))
	
        disp = CameraDisplay(geom, pix_image[:1039])
        disp.add_colorbar(label='pixel charge')
        disp.highlight_pixels(clean_mask_pixels, color='red', alpha=1, linewidth=1)
	
	#alternative image: used pixels after the cleaning and differences mask
        '''for id in range(len(image)):
           if stereo_id[id] not in ids:
                continue
            event_image = np.array(image[id][:1039])
            clean_mask = event_image != 0
            #print(event_image[clean_mask])
            norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
            disp = CameraDisplay(geometry_mars, image=event_image)
            disp.highlight_pixels(clean_mask_pixels, color='red', alpha=0.5, linewidth=2)
            disp.set_limits_minmax(vmin, 1)'''

        plt.title("differences MARS-mcp")  
	    
	#differences between MARS and the calibrated data
        plt.subplot2grid(grid_shape, (1, 1))

        pix_diff_mars_copy = np.array(pix_diff_mars).copy()
        pix_diff_mars_copy[np.array(event_image_mars[:1039]) == 0] = 0


        disp = CameraDisplay(geom, pix_diff_mars_copy>0)
	# disp.highlight_pixels(clean_mask_pixels, color='red', alpha=1, linewidth=1)
        disp.highlight_pixels(np.array(event_image_mars[:1039]) != 0, color='white', alpha=1, linewidth=3)
	#disp.set_limits_minmax(vmin, vmax)
	#disp.add_colorbar(label='pixel charge')
	
        plt.title("differences MARS-original data")
	
	#mcp-orig-data-diff
        plt.subplot2grid(grid_shape, (1, 2))
	        
        pix_diff_mcp_copy = np.array(pix_diff_mcp).copy()
        pix_diff_mcp_copy[np.array(event_image_cleaned) == 0] = 0

        disp = CameraDisplay(geom, pix_diff_mcp_copy>0)
        disp.highlight_pixels(np.array(event_image_cleaned) != 0, color='white', alpha=1, linewidth=3)                       
        #disp.set_limits_minmax(vmin, vmax)
        #disp.add_colorbar(label='pixel charge')
        plt.title("differences mcp-original data")

        
        fig.suptitle(f'Comparison_MARS_magic-cta-pipe: Event ID {id_event}, {run_num}, M{tel_id}', fontsize=16)
        fig.savefig(f"{out_path}/data-comparison-{run_num}_{id_event}_M{tel_id}.png")
        #fig.savefig(f"{out_path}/data-comparison-{run_num}_{id_event}_M{tel_id}.pdf")
        
    ids.clear()

