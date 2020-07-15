import glob, sys
import uproot   
import h5py
import re
import uproot
import traitlets
import itertools
import copy
import math
from texttable import Texttable

import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import pandas as pd 
import scipy
from scipy.sparse.csgraph import connected_components
from scipy.sparse import lil_matrix, csr_matrix
import numpy.ma as ma

import ctapipe
from ctapipe_io_magic import MAGICEventSource

from astropy import units as u

from ctapipe.instrument import CameraGeometry
from ctapipe.calib import CameraCalibrator
from ctapipe.visualization import CameraDisplay
from ctapipe.image import hillas_parameters
from ctapipe.image.cleaning import tailcuts_clean, apply_time_delta_cleaning

from ctapipe.io.eventseeker import EventSeeker

from matplotlib import pyplot, colors
import pylab as plt

import argparse
import csv

from utils import MAGIC_Badpixels
# from utils import bad_pixel_treatment
from utils import MAGIC_Cleaning

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def file_reader(filename, max_row = 1e30):

    datafile = open(filename, 'r')
    datareader = csv.reader(datafile)
    data = []
    for ix, row in enumerate(datareader):
        if ix >= max_row:
            break
        if len(row) == 0:
            data.append(row)
        else:
            data.append(list(map(int,row[0][:-1].split(' '))))

    datafile.close()
    return data


def main():

    usage = "usage: %(prog)s [options] "
    description = "Run gtselect and gtmktime on one or more FT1 files.  "
    "Note that gtmktime will be skipped if no FT2 file is provided."
    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-m','--mc', dest='mc', action='store_true')

    parser.add_argument('--base',dest='base',action='store_true')
    parser.add_argument('--nopix', dest='nopix', action='store_true')

    parser.add_argument('--test',dest='test',action='store_true')
    # parser.add_argument('--testnopix', dest='testnopix', action='store_true')

    args = parser.parse_args()

    plot = False
    tel_id = 1
    max_events = 1e40

    mars_camera = CameraGeometry.from_name("MAGICCamMars")
    equivalent_focal_length = 16.97 * u.m

    neighbors = mars_camera.neighbor_matrix_sparse
    outermost = []
    for pix in range(mars_camera.n_pixels):
        if neighbors[pix].getnnz() < 5:
            outermost.append(pix)

    cleaning_config = dict(
      picture_thresh = 6,
      boundary_thresh = 3.5,
      max_time_off = 4.5 * 1.64,
      max_time_diff = 1.5 * 1.64,
      usetime = True,
      usesum = True,
      findhotpixels=True,
      test = False,
    )

    if args.base:

        base_dir = "/home/iwsatlas1/damgreen/CTA/MAGIC_Cleaning/star/Star_Base/"

        star_file = "%s/20191127_M1_05086952.001_I_CrabNebula-W0.40+035.root" % base_dir
        event_images_stardump = np.genfromtxt("%s/event_images.txt"  % base_dir,dtype=float,max_rows=max_events)
        event_times_stardump = np.genfromtxt("%s/event_times.txt"  % base_dir,dtype=float,max_rows=max_events)
        event_unsuitables_stardump = np.genfromtxt("%s/event_unsuitable.txt"  % base_dir,dtype=bool,max_rows=max_events)
        event_unmappeds_stardump = np.genfromtxt("%s/event_unmapped.txt"  % base_dir,dtype=bool,max_rows=max_events)

        event_cleanstep1 = file_reader("%s/event_cleanstep1.txt"  % base_dir,max_row=max_events)
        event_cleanstep2 = file_reader("%s/event_cleanstep2.txt"  % base_dir,max_row=max_events)
        event_cleanstep3 = file_reader("%s/event_cleanstep3.txt"  % base_dir,max_row=max_events)

        cleaning_config['findhotpixels'] = True

    if args.nopix:
        base_dir = "/home/iwsatlas1/damgreen/CTA/MAGIC_Cleaning/star/Star_No_HotPix/"

        star_file = "%s/20191127_M1_05086952.001_I_CrabNebula-W0.40+035.root" % base_dir
        event_images_stardump = np.genfromtxt("%s/event_images.txt"  % base_dir,dtype=float,max_rows=max_events)
        event_times_stardump = np.genfromtxt("%s/event_times.txt"  % base_dir,dtype=float,max_rows=max_events)
        event_unsuitables_stardump = np.genfromtxt("%s/event_unsuitable.txt"  % base_dir,dtype=bool,max_rows=max_events)
        event_unmappeds_stardump = np.genfromtxt("%s/event_unmapped.txt"  % base_dir,dtype=bool,max_rows=max_events)

        event_cleanstep1 = file_reader("%s/event_cleanstep1.txt"  % base_dir,max_row=max_events)
        event_cleanstep2 = file_reader("%s/event_cleanstep2.txt"  % base_dir,max_row=max_events)
        event_cleanstep3 = file_reader("%s/event_cleanstep3.txt"  % base_dir,max_row=max_events)

        cleaning_config['findhotpixels'] = False

    bad_pixels_config = dict(
        pedestalLevel = 400,
        pedestalLevelVariance = 4.5,
        pedestalType = 'FromExtractorRndm'
    )


    # Grab the processed star information
    hillas_array_list = [
                        'MHillas.fSize',
                        'MHillas.fWidth',
                        'MHillas.fLength',
                        'MRawEvtHeader.fStereoEvtNumber', 
                        'MNewImagePar.fNumCorePixels',
                        'MNewImagePar.fNumUsedPixels',
                        ]

    input_file = uproot.open(star_file)
    events = input_file['Events'].arrays(hillas_array_list)

    evtnum_arr = events[b'MRawEvtHeader.fStereoEvtNumber']
    size_arr = events[b'MHillas.fSize']
    width_arr = events[b'MHillas.fWidth']
    length_arr = events[b'MHillas.fLength']
    core_arr = events[b'MNewImagePar.fNumCorePixels']
    used_arr = events[b'MNewImagePar.fNumUsedPixels']


    star_size_arr = []
    ctapipe_size_arr = []

    star_length_arr = []
    ctapipe_length_arr = []

    star_width_arr = []
    ctapipe_width_arr = []

    magic_clean = MAGIC_Cleaning.magic_clean(mars_camera,cleaning_config)
    badpixel_calculator = MAGIC_Badpixels.MAGICBadPixelsCalc(config=bad_pixels_config)

    size_cut = 0
    leakage_cut = 0.15
    AberrationCorrection = 1.0713

    if args.mc:
        """ MC FILES"""
        mars_file_mask = '/remote/ceph/group/magic/MAGIC-LST/MCs/MAGIC/ST.03.07/za05to35/Train_sample/1.Calibrated/GA_M1*root'
        mars_file_list = glob.glob(mars_file_mask)  #  Here makes array which contains files matching the input condition (GA_M1*root).
        file_list = (list(filter(lambda name : '123' in name, mars_file_list)))
        print("Number of files : %s" % len(file_list))
    else:
        """DATA FILES"""
        mars_file_mask = '/home/iwsatlas1/damgreen/CTA/MAGIC_Cleaning/datafile/Calib/*.root'
        mars_file_list = glob.glob(mars_file_mask)  #  Here makes array which contains files matching the input condition (GA_M1*root).
        file_list = (list(filter(lambda name : '0' in name, mars_file_list)))
        print("Number of files : %s" % len(file_list))

    tolerance = 0.5

    offset = -1
    for ix, file_name in enumerate(file_list):

        magic_event_source = MAGICEventSource(input_url=file_name)
        event_generator = magic_event_source._pedestal_event_generator(telescope='M1')

        for i, event in enumerate(event_generator):


            # This is used to make sure the star and ctapipe events are lined up
            if i == 0:
                continue

            if i == 9399:
                offset -= 1
                continue

            if i > max_events:
                break


            # Basic ctapipe image processing and cleaning
            tels_with_data = list(event.r1.tels_with_data)
            subarray = event.inst.subarray

            badrmspixel_mask = badpixel_calculator.get_badrmspixel_mask(event)
            deadpixel_mask = badpixel_calculator.get_deadpixel_mask(event)
            unsuitable_mask = np.logical_or(badrmspixel_mask[0], deadpixel_mask[0])

            event_image = event.dl1.tel[tel_id].image
            event_pulse_time = event.dl1.tel[tel_id].pulse_time

            clean_mask, event_image, event_pulse_time = magic_clean.clean_image(event_image, event_pulse_time,unsuitable_mask=unsuitable_mask)

            event_image_cleaned = event_image.copy()
            event_image_cleaned[~clean_mask] = 0

            try:
                hillas_params = hillas_parameters(mars_camera, event_image_cleaned)
            except:
                continue


            # All of the comparisons between star and ctapipe
            event_image_stardump = event_images_stardump[i+offset,:1039]
            event_time_stardump = event_times_stardump[i+offset,:1039]

            image_diff = 100*np.abs(event_image_stardump - event_image)/event_image_stardump
            image_diff_selection = image_diff > tolerance

            time_diff = 100*np.abs(event_time_stardump - event_pulse_time)/event_time_stardump
            time_diff_selection = time_diff > tolerance


            ctapipe_unsuitable_mask = copy.copy(unsuitable_mask)
            star_unsuitable_mask = copy.copy(event_unsuitables_stardump[i+offset,:1039])
            ctapipe_unsuitable_pixels = np.where(ctapipe_unsuitable_mask)[0]
            star_unsuitable_pixels = np.where(star_unsuitable_mask)[0]

            ctapipe_unmapped_mask = copy.copy(magic_clean.unmapped_mask)
            star_unmapped_mask = copy.copy(event_unmappeds_stardump[i+offset,:1039])
            ctapipe_unmapped_pixels = np.where(ctapipe_unmapped_mask)[0]
            star_unmapped_pixels = np.where(star_unmapped_mask)[0]

            diff_unsuitable = np.setdiff1d(np.union1d(ctapipe_unsuitable_pixels, star_unsuitable_pixels), np.intersect1d(ctapipe_unsuitable_pixels, star_unsuitable_pixels))
            diff_unmapped = np.setdiff1d(np.union1d(ctapipe_unmapped_pixels, star_unmapped_pixels), np.intersect1d(ctapipe_unmapped_pixels, star_unmapped_pixels))

            ctapipe_length = hillas_params.length.to('mm')/AberrationCorrection
            ctapipe_width = hillas_params.width.to('mm')/AberrationCorrection
            ctapipe_size = hillas_params.intensity

            star_idx = np.where(evtnum_arr == event.dl0.event_id)[0]
            if len(star_idx) != 1:
                continue


            star_length = length_arr[star_idx[0]]*u.mm
            star_width = width_arr[star_idx[0]]*u.mm
            star_size = size_arr[star_idx[0]]

            if star_length.value < 1e-1 or star_width.value < 1e-1:
                continue

            length_ratio = np.fabs((ctapipe_length - star_length)/star_length)*100
            width_ratio = np.fabs((ctapipe_width - star_width)/star_width)*100
            size_ratio = np.fabs((ctapipe_size - star_size)/star_size)*100

            ctapipe_step1 = np.where(magic_clean.mask_step1)[0]
            star_step1 = np.asarray(event_cleanstep1[i+offset])

            ctapipe_step2 = np.where(magic_clean.mask_step2)[0]
            star_step2 = np.asarray(event_cleanstep2[i+offset])

            ctapipe_step3 = np.where(magic_clean.mask_step3)[0]
            star_step3 = np.asarray(event_cleanstep3[i+offset])

            diff_step1 = np.setdiff1d(np.union1d(ctapipe_step1, star_step1), np.intersect1d(ctapipe_step1, star_step1))
            diff_step2 = np.setdiff1d(np.union1d(ctapipe_step2, star_step2), np.intersect1d(ctapipe_step2, star_step2))
            diff_step3 = np.setdiff1d(np.union1d(ctapipe_step3, star_step3), np.intersect1d(ctapipe_step3, star_step3))

            if size_ratio > tolerance or length_ratio > tolerance or width_ratio > tolerance:
            # if len(diff_step3) > 0 or len(diff_unsuitable) > 0 or len(diff_unmapped) > 0:# or np.sum(image_diff_selection) > 0 or np.sum(time_diff_selection) > 0:
                print("*"*50)
                print(bcolors.OKGREEN + "Count : %i, Stereo Event Number : %s" % (i,event.dl0.event_id) + bcolors.ENDC)


                if len(diff_unsuitable) > 0:
                    print("+"*50)
                    print(bcolors.FAIL + "Unsuitable Diff : " + bcolors.ENDC)
                    print("diff:\t", diff_unsuitable.tolist())

                if len(diff_unmapped) > 0:
                    print("+"*50)
                    print(bcolors.FAIL + "Unmapped Diff : " + bcolors.ENDC)
                    print("diff:\t", diff_unmapped.tolist())

                if np.sum(image_diff_selection) > 0:
                    print("+"*50)
                    print(bcolors.FAIL + "Image Diff : " + bcolors.ENDC)
                    print("image ctapipe : ",event_image[image_diff_selection].tolist())
                    print("image dump : ",event_image_stardump[image_diff_selection].tolist())
                    print("image diff (%) : ",image_diff[image_diff_selection].tolist())

                if np.sum(time_diff_selection) > 0:
                    print("+"*50)
                    print(bcolors.FAIL + "Time Diff : " + bcolors.ENDC)
                    print("time ctapipe : ",event_pulse_time[time_diff_selection].tolist())
                    print("time dump : ",event_time_stardump[time_diff_selection].tolist())
                    print("time diff : ",time_diff[time_diff_selection].tolist())

                string = ""
                if len(diff_step1) > 0:
                    string += "STEP1 DIFF Size %s, " % len(diff_step1)
                if len(diff_step2) > 0:
                    string += "STEP2 DIFF Size %s, " % len(diff_step2)
                if len(diff_step3) > 0:
                    string += "STEP3 DIFF Size %s" % len(diff_step3)
        
                if len(string) > 0:
                    print(bcolors.FAIL + string + bcolors.ENDC)

                string = ""
                if size_ratio > tolerance:
                    string += "SIZE RATIO : %.2f, " % size_ratio
                if length_ratio > tolerance:
                    string +=" LENGTH RATIO : %.2f, " % length_ratio
                if width_ratio > tolerance:
                    string += " WIDTH RATIO : %.2f" % width_ratio

                if len(string) > 0:
                    print(bcolors.WARNING + string + bcolors.ENDC)
                    t = Texttable()
                    t.add_rows([['var', 'ctapipe','star','ratio'], ['Size', hillas_params.intensity,size_arr[star_idx[0]],size_ratio], ['Width', ctapipe_width.value, star_width.value, width_ratio], ['Length', ctapipe_length.value, star_length.value, length_ratio ], ['CorePix',magic_clean.core_pix, core_arr[star_idx[0]], np.nan], ['UsedPix', magic_clean.used_pix, used_arr[star_idx[0]],np.nan]   ])
                    print(bcolors.BOLD + bcolors.OKBLUE + t.draw() + bcolors.ENDC)

                # print("cta mask :\t" , np.where(clean_mask)[0].tolist())
                # print("star mask :\t" , star_step3.tolist())

                if len(diff_step1) > 0:
                    print(bcolors.FAIL + "Clean Step 1 Diff : " + bcolors.ENDC,end='')
                    # print("star:\t", star_step1.tolist())
                    # print("cta:\t", ctapipe_step1.tolist())
                    if len(star_step1) > len(ctapipe_step1):
                        print("Pixel not found in ctapipe : \t", diff_step1.tolist())
                    else:
                        print("Extra pixel found in ctapipe : \t", diff_step1.tolist())

                if len(diff_step2) > 0:
                    print(bcolors.FAIL + "Clean Step 2 Diff : " + bcolors.ENDC,end='')
                    # print("star:\t", star_step2.tolist())
                    # print("cta:\t", ctapipe_step2.tolist())
                    if len(star_step2) > len(ctapipe_step2):
                        print("Pixel not found in ctapipe : \t", diff_step2.tolist())
                    else:
                        print("Extra pixel found in ctapipe : \t", diff_step2.tolist())

                if len(diff_step3) > 0:
                    print(bcolors.FAIL + "Clean Step 3 Diff : " + bcolors.ENDC,end='')
                    # print("star:\t", star_step3.tolist())
                    # print("cta:\t", ctapipe_step3.tolist())
                    if len(star_step3) > len(ctapipe_step3):
                        print("Pixel not found in ctapipe : \t", diff_step3.tolist())
                    else:
                        print("Extra pixel found in ctapipe : \t", diff_step3.tolist())

            if star_size > size_cut:

                star_size_arr.append(star_size)
                ctapipe_size_arr.append(ctapipe_size)

                star_length_arr.append(star_length.value)
                ctapipe_length_arr.append(ctapipe_length.value)

                star_width_arr.append(star_width.value)
                ctapipe_width_arr.append(ctapipe_width.value)

    star_size_arr = np.asarray(star_size_arr)
    star_length_arr = np.asarray(star_length_arr)
    star_width_arr = np.asarray(star_width_arr)

    ctapipe_size_arr = np.asarray(ctapipe_size_arr)
    ctapipe_length_arr = np.asarray(ctapipe_length_arr)
    ctapipe_width_arr = np.asarray(ctapipe_width_arr)

    plt.clf()
    fig, axs = plt.subplots(2,3,figsize=(15,10))
    bins = 101

    axs[0,0].scatter(np.log10(star_size_arr),np.log10(ctapipe_size_arr))
    axs[0,0].set_title("Size")
    axs[0,0].set_xlim([1.0,5.5])
    axs[0,0].set_ylim([1.0,5.5])
    axs[0,0].set_xlabel("Mars - log10(size)")
    axs[0,0].set_ylabel("ctapipe - log10(size)")

    axs[0,1].scatter(np.log10(star_length_arr),np.log10(ctapipe_length_arr))
    axs[0,1].set_title("Length")
    axs[0,1].set_xlim([0.0,3.0])
    axs[0,1].set_ylim([0.0,3.0])
    axs[0,1].set_xlabel("Mars - log10(length)")
    axs[0,1].set_ylabel("ctapipe - log10(length)")

    axs[0,2].scatter(np.log10(star_width_arr),np.log10(ctapipe_width_arr))
    axs[0,2].set_title("Width")
    axs[0,2].set_xlim([0.0,3.0])
    axs[0,2].set_ylim([0.0,3.0])
    axs[0,2].set_xlabel("Mars - log10(width)")
    axs[0,2].set_ylabel("ctapipe - log10(width)")

    size_ratio = (ctapipe_size_arr - star_size_arr)/star_size_arr
    axs[1,0].hist(size_ratio,bins=bins)
    plt.text(0.15, 0.9, 'mean : %.4f' % np.nanmean(size_ratio), horizontalalignment='left', verticalalignment='center', transform=axs[1,0].transAxes)
    plt.text(0.15, 0.85, 'rms : %.4f' % np.nanstd(size_ratio), horizontalalignment='left', verticalalignment='center', transform=axs[1,0].transAxes)
    axs[1,0].set_xlabel("Size Ratio [(ctapipe - star)/star]")
    axs[1,0].set_yscale('log', nonposy='clip')

    length_ratio = (ctapipe_length_arr - star_length_arr)/star_length_arr
    axs[1,1].hist(length_ratio,bins=bins)
    plt.text(0.15, 0.9, 'mean : %.4f' % np.nanmean(length_ratio), horizontalalignment='left', verticalalignment='center', transform=axs[1,1].transAxes)
    plt.text(0.15, 0.85, 'rms : %.4f' % np.nanstd(length_ratio), horizontalalignment='left', verticalalignment='center', transform=axs[1,1].transAxes)
    axs[1,1].set_xlabel("Length Ratio [(ctapipe - star)/star]")
    axs[1,1].set_yscale('log', nonposy='clip')

    width_ratio = (ctapipe_width_arr - star_width_arr)/star_width_arr
    axs[1,2].hist(width_ratio,bins=bins)
    plt.text(0.15, 0.9, 'mean : %.4f' % np.nanmean(width_ratio), horizontalalignment='left', verticalalignment='center', transform=axs[1,2].transAxes)
    plt.text(0.15, 0.85, 'rms : %.4f' % np.nanstd(width_ratio), horizontalalignment='left', verticalalignment='center', transform=axs[1,2].transAxes)
    axs[1,2].set_xlabel("Width Ratio [(ctapipe - star)/star]")
    axs[1,2].set_yscale('log', nonposy='clip')

    if args.base:
        plt.savefig("comp_M1_base.pdf")
    elif args.nopix:
        plt.savefig("comp_M1_base_nopix.pdf")

if __name__ == "__main__":
    main()
