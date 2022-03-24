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
from ctapipe.containers import HillasParametersContainer
from ctapipe.io.eventseeker import EventSeeker
from magicctapipe.utils import MAGIC_Cleaning  
from astropy.coordinates import Angle
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
    The function returns a list with the Event IDs for events, where there's a difference between the images.
    ---
    config_file: Configuration file
    mode: 
        use_all: use all events in the given input file
        use_ids_config: use events, whose ids are given in the config file
    tel_id: Telescope of which the data will be compared. use "1" for M1 and "2" for M2
    """


    # load config file--------------------------------------------------------------------------------------------
    config = yaml.safe_load(open(config_file, "r"))
    out_path = config["output_files"]["file_path"]
    comparison = []

    # get id to compare from config file---------------------------------------------------------------------------
    if mode == "useall":
        with uproot.open(config["input_files"]["magic_cta_pipe"][f"M{tel_id}"]) as mcp_file:
            ids_to_compare = mcp_file["Events"]["MRawEvtHeader.fStereoEvtNumber"].array(library="np")
            ids_to_compare = ids_to_compare["event_id"].tolist()
    elif mode == "use_ids_config":
        ids_to_compare = config["event_list"]
 
    print(len(ids_to_compare), "events will be compared", ids_to_compare)   

    # we will now load the data files, and afterwards select the corresponding data for our events
    # get mars data ------------------------------------------------------------------------------------------------------
    mars_input = config["input_files"]["mars"]
    image = []      #pixel charges
    events=[]       #event ids
    telescope=[]    #telescope id
    obs_id=[]       #run number
    
    for image_container in read_images(mars_input):
        events.append(image_container.event_id)
        telescope.append(image_container.tel_id)
        obs_id.append(image_container.obs_id)
        image.append(image_container.image_cleaned)

    mars_data = pd.DataFrame(list(zip(events, telescope, obs_id, image)), columns=["event_id", "tel_id", "obs_id", "image"])

    # get original data-----------------------------------------------------------------------------------------------
    data_path_original = config["input_files"]["magic_cta_pipe"][f"M{tel_id}"]
    with uproot.open(data_path_original) as input_data:
        event_ids_original = input_data["Events"]["MRawEvtHeader.fStereoEvtNumber"].array(library="np")
        images_original = input_data["Events"]["MCerPhotEvt.fPixels.fPhot"].array(library="np")

    # get other data----------------------------------------------------------------------------------------------------
    # we loop through the events, and only compare those that are in ids_to_compare
    
    source = MAGICEventSource(input_url=config["input_files"]["magic_cta_pipe"][f"M{tel_id}"])

    run_num = source.obs_ids[0]

    seeker = EventSeeker(event_source=source)

    f=h5py.File(f"Image_comparison_{run_num}.h5", "a")

    MAGICCAM = CameraDescription.from_name("MAGICCam")
    GEOM = MAGICCAM.geometry
    geometry_mars = new_camera_geometry(GEOM)       

    geometry_old = source.subarray.tel[tel_id].camera.geometry
    geometry_mcp = new_camera_geometry(geometry_old)


    for event_id in ids_to_compare:
        try:
            event = seeker.get_event_id(event_id)
        except IndexError:
            print(f"Event with ID {event_id} not found in calibrated file. Skipping...")
            continue
        
        print("Event ID:", event.index.event_id, "- Telecope ID:", tel_id)

        

        # mars data -----------------------------------------------------------------------------
        mars_event = mars_data.loc[(mars_data["event_id"]==event.index.event_id)&(mars_data["tel_id"]==tel_id)]
        if mars_event.empty:
            print(f"Event with ID {event_id} not found in MARS file. Skipping...")
            continue
        idx = mars_event["image"].index[0]
        event_image_mars = np.array(mars_event["image"][idx][:1039])
        clean_mask_mars = event_image_mars != 0

        #compare number of islands
        # num_islands_mars, island_labels_mars = number_of_islands(geometry_mars, (np.array(event_image[:1039]) > 0))

        # get mcp data------------------------------------------------------------------------------
        magic_clean = MAGIC_Cleaning.magic_clean(geometry_mcp,cleaning_config)
        original_data_images = event.dl1.tel[tel_id].image
        original_data_images_copy = original_data_images.copy()
        event_pulse_time = event.dl1.tel[tel_id].peak_time

        badrmspixel_mask = event.mon.tel[tel_id].pixel_status.pedestal_failing_pixels[2]
        deadpixel_mask = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[0]
        unsuitable_mask = np.logical_or(badrmspixel_mask, deadpixel_mask)
        bad_pixel_indices = [i for i, x in enumerate(badrmspixel_mask) if x]
        dead_pixel_indices = [i for i, x in enumerate(deadpixel_mask) if x]
        bad_not_dead_pixels_test = [i for i in bad_pixel_indices if i not in dead_pixel_indices]

        clean_mask, calibrated_data_images, event_pulse_time = magic_clean.clean_image(original_data_images_copy, event_pulse_time, unsuitable_mask=unsuitable_mask)
        event_image_mcp = calibrated_data_images.copy()
        event_image_mcp[~clean_mask] = 0

        if not np.any(event_image_mcp):
            print(f"Event ID {event.index.event_id} for telescope {tel_id} does not have any surviving pixel. Skipping...")
            continue

        # get max value for colorbar--------------------------------------------------------------------------------
        mcp_max = np.amax(event_image_mcp[clean_mask])
        mars_max = np.amax(event_image_mars[clean_mask_mars])
        if mcp_max >= mars_max:
            vmax = mcp_max
        else:
            vmax = mars_max

        vmin = 0
        # print(vmax)

        # find differences------------------------------------------------------------------------------------------
        charge_differences = abs(event_image_mars - event_image_mcp)
        clean_mask_pixels = charge_differences != 0

        errors = False
        if np.any(charge_differences >= 0.00000001*vmax):       #threshold for differences allowed. The difference cannot be higher than x% of the highest pixel charge
            errors = True

        if errors:
            comparison.append(event.index.event_id)
            print(f"errors found for {event.index.event_id}!")

        # differences between calibrated data before cleaning and mcp/mars data
        pix_diff_mars = abs(event_image_mars - calibrated_data_images)
        pix_diff_mcp = abs(event_image_mcp - calibrated_data_images)

        #create output file that contains the pixel values and differences------------------------------------------------
        
        """

        df_pixel = pd.DataFrame()
        data = np.transpose(np.array([event_image_mars, event_image_mcp, calibrated_data_images]))
        #data_real = np.transpose(data)
        
        df_pixel = pd.DataFrame(data, columns=["MARS_charge", "mcp_charge", "calibrated_data"])
        #df_pixel["calibrated data"] = np.transpose(data_value)#(calibrated_data_images[:1039])
        df_pixel["difference_MARS-mcp"] = np.transpose(charge_differences)
        df_pixel["relative_error_MARS-mcp"] = np.transpose(charge_differences/event_image_mcp)
        # alternatively one could also calculate the relative error with the MARS charge value
        # df_pixel["relative_error_MARS-mcp"] = np.transpose(charge_differences/event_image_mars)
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

        df_pixel = pd.DataFrame()
        data = np.transpose(np.array([event_image_mars, event_image_mcp, calibrated_data_images]))
        
        df_pixel = pd.DataFrame(data, columns=["MARS_charge", "mcp_charge", "calibrated_data"])
        df_pixel["difference_MARS-mcp"] = np.transpose(charge_differences)
        df_pixel["relative_error_MARS-mcp"] = np.transpose(charge_differences/event_image_mcp)
        # alternatively one could also calculate the relative error with the MARS charge value
        # df_pixel["relative_error_MARS-mcp"] = np.transpose(charge_differences/event_image_mars)
        df_pix_diff = df_pixel.loc[df_pixel["difference_MARS-mcp"] > 0]
        # print(df_pix_diff)
        
        #saving the output
        #the file only gets saved if there are differences between the images
        if config["save_only_when_differences"] == True:
            if any(charge_differences) == True: 
                df_pixel.to_hdf(f"{run_num}_image_comparison.h5", f"/{event.index.event_id}_M{tel_id}", "a")
                #df_pix_diff.to_hdf(f"{out_path}{run_num}_{id_event}_M{telescope_id}_pixel_diff.h5", "/pixel_differences", "w")
                
        #the file gets saved in any case
        elif config["save_only_when_differences"] == False:
            df_pixel.to_hdf(f"{run_num}_image_comparison.h5", f"/{event.index.event_id}_M{tel_id}", "a")

        
        # plotting ------------------------------------------------------------------------------------------------------
        fig = plt.figure(figsize=(20, 10))
        grid_shape = (2, 3)
    
        #original data
        plt.subplot2grid(grid_shape, (0, 0))
        geom = CameraGeometry.from_name("MAGICCamMars")
        disp = CameraDisplay(geom, original_data_images)
        # pixels whose original value is negative
        # negative_mask = calibrated_data_images < 0
        disp.add_colorbar(label="pixel charge")
        disp.set_limits_minmax(vmin, vmax)
        plt.title("original data")
            
        #mars_data
        plt.subplot2grid(grid_shape, (0, 1))
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        disp_mars = CameraDisplay(geometry_mars, image=event_image_mars)
        disp_mars.set_limits_minmax(vmin, vmax)
        disp_mars.add_colorbar(label="pixel charge")
        disp_mars.highlight_pixels(clean_mask_pixels[:1039], color="red", alpha=1, linewidth=1)     
        plt.title("MARS data")

        #mcp_data
        plt.subplot2grid(grid_shape, (0, 2))        
        disp_mcp = CameraDisplay(geometry_mcp, image=event_image_mcp)
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
        
        # the white outline shows the pixels used for the image after cleaning, the ones that are filled yellow show where differences are
        # differences between MARS and the calibrated data
        plt.subplot2grid(grid_shape, (1, 1))
        pix_diff_mars_copy = np.array(pix_diff_mars).copy()
        pix_diff_mars_copy[np.array(event_image_mars) == 0] = 0
        disp = CameraDisplay(geom, pix_diff_mars_copy>0)
        disp.highlight_pixels(np.array(event_image_mars) != 0, color="white", alpha=1, linewidth=3)
        plt.title("differences MARS-original data")
    
        #mcp-orig-data-diff
        plt.subplot2grid(grid_shape, (1, 2))
        pix_diff_mcp_copy = np.array(pix_diff_mcp).copy()
        pix_diff_mcp_copy[np.array(event_image_mcp) == 0] = 0
        disp = CameraDisplay(geom, pix_diff_mcp_copy>0)
        disp.highlight_pixels(np.array(event_image_mcp) != 0, color="white", alpha=1, linewidth=3)             
        plt.title("differences mcp-original data")
        
        
        fig.suptitle(f"Comparison_MARS_magic-cta-pipe: Event ID {event.index.event_id}, {run_num}, M{tel_id}", fontsize=16)
        fig.savefig(f"{out_path}/image-comparison-{run_num}_{event.index.event_id}_M{tel_id}.png")
        # print(f"{out_path}/image-comparison-{run_num}_{event.index.event_id}_M{tel_id}.png")
        # fig.savefig(f"{out_path}/image-comparison-{run_num}_{event.index.event_id}_M{tel_id}.pdf")
            
        
    return comparison
