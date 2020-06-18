import glob, sys
import uproot   
import h5py
import re
import uproot
import traitlets
import itertools
import copy

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd 
import scipy
from scipy.sparse.csgraph import connected_components
from scipy.sparse import lil_matrix, csr_matrix

# import ctapipe_io_magic
# import importlib 
# importlib.reload(ctapipe_io_magic)
import ctapipe
from ctapipe_io_magic import MAGICEventSourceMC

from astropy import units

from ctapipe.instrument import CameraGeometry
from ctapipe.calib import CameraCalibrator
from ctapipe.visualization import CameraDisplay
from ctapipe.image import hillas_parameters
from ctapipe.image.cleaning import tailcuts_clean, apply_time_delta_cleaning

from ctapipe.io.eventseeker import EventSeeker

from matplotlib import pyplot, colors

from MAGIC_Cleaning import magic_clean as clean

def main():

    plot = True

    tel_id = 1
    # mars_file_mask = '/remote/ceph/group/magic/MAGIC-LST/MCs/MAGIC/ST.03.07/za05to35/Train_sample/1.Calibrated/GA_M1*root'
    mars_file_mask = '/home/iwsatlas1/damgreen/CTA/MAGIC_Cleaning/datafile/*.root'
    mars_file_list = glob.glob(mars_file_mask)  #  Here makes array which contains files matching the input condition (GA_M1*root).

    file_name = (list(filter(lambda name : '8' in name, mars_file_list)))[0]
    # file_name = (list(mars_file_list))

    magic_event_source = MAGICEventSourceMC(input_url=file_name)
    event_generator = magic_event_source._mono_event_generator()

    mars_camera = CameraGeometry.from_name("MAGICCamMars")

    config = dict(
      picture_thresh = 6, 
      boundary_thresh = 3.5,
      max_time_off = 4.5 * 1.64,
      max_time_diff = 1.5 * 1.64, 
      usetime = True,
      usesum = True,
    )

    magic_clean = clean(mars_camera,config)

    for i, event in enumerate(event_generator):
        tels_with_data = list(event.r1.tels_with_data)
       
        subarray = event.inst.subarray
        camera = subarray.tel[tel_id].camera

        event_image = event.dl1.tel[tel_id].image
        event_pulse_time = event.dl1.tel[tel_id].pulse_time

        clean_mask = magic_clean.clean_image(event_image, event_pulse_time)

        event_image_cleaned = event_image.copy()
        event_image_cleaned[~clean_mask] = 0

        event_pulse_time_cleaned = event_pulse_time.copy()
        event_pulse_time_cleaned[~clean_mask] = 0

        event_image_cleaned = event_image.copy()
        event_image_cleaned[~clean_mask] = 0

        event_pulse_time_cleaned = event_pulse_time.copy()
        event_pulse_time_cleaned[~clean_mask] = 0

        if scipy.any(event_image_cleaned):
            hillas_params = hillas_parameters(mars_camera, event_image_cleaned)

            print("Event Number : %i, Size : %.2f, Length : %.2f, Width : %.2f" % (i, hillas_params.intensity, hillas_params.length.to(units.mm).value, hillas_params.width.to(units.mm).value))

            if plot:

                pyplot.figure(figsize=(10, 8))
                pyplot.clf()

                # pyplot.style.use('presentation')

                pyplot.subplot(221)
                disp = CameraDisplay(mars_camera, event_image, cmap='jet')
                pyplot.title('Original image')
                disp.highlight_pixels(event_image_cleaned > 0,color='red')
                disp.add_colorbar()

                pyplot.subplot(222)
                disp = CameraDisplay(mars_camera, event_pulse_time, cmap='jet')
                # disp.set_limits_minmax(20,30)
                disp.highlight_pixels(event_image_cleaned > 0,color='red')
                pyplot.title('Original arrival time map')
                disp.add_colorbar()

                pyplot.subplot(223)
                disp = CameraDisplay(mars_camera, event_image_cleaned, cmap='bwr')
                # disp.overlay_moments(hillas_params, color='red', lw=3)
                # disp.set_limits_minmax(-20,20)

                disp.highlight_pixels(event_image_cleaned > 0,color='red')

                pyplot.title('Cleaned image')
                disp.add_colorbar()

                time_mean = np.mean(event_pulse_time_cleaned[event_image_cleaned > 0])
                pyplot.subplot(224)
                disp = CameraDisplay(mars_camera, event_pulse_time_cleaned, cmap='jet')
                # disp.set_limits_minmax(-5,5)
                disp.highlight_pixels(event_image_cleaned > 0,color='red')
                pyplot.title('Cleaned arrival time amp')
                disp.add_colorbar()

                pyplot.tight_layout()

                pyplot.savefig("plots/event_%04i.pdf" % i)

        # if i >= 17:
        #     break

if __name__ == "__main__":
    main()
