import numpy as np
from ctapipe.instrument import CameraGeometry

class MAGICBadPixelsCalc():

    def __init__(self, camera=None, config=None, tool=None, **kwargs):

        # MAGIC telescope description
        if camera == None:
            camera = CameraGeometry.from_name('MAGICCam')

        self.n_camera_pixels = camera.n_pixels

        self.hotpixels = np.zeros(self.n_camera_pixels, dtype=np.bool)
        self.coldpixels = np.zeros(self.n_camera_pixels, dtype=np.bool)
        self.badrmspixels = np.zeros(self.n_camera_pixels, dtype=np.bool)

        self.config = config
        self.is_update = np.zeros(2, dtype=np.bool) + 1

        if 'pedestalLevel' in config:
            self.pedestalLevel = config['pedestalLevel']
        else:
            self.pedestalLevel = 400.

        if 'pedestalLevelVariance' in config:
            self.pedestalLevelVariance = config['pedestalLevelVariance']
        else:
            self.pedestalLevelVariance = 4.5
            
        if 'pedestalType' in config:
            pedestalTypeName = config['pedestalType']
            if pedestalTypeName == 'Fundamental':
                self.pedestalType = 0
            elif pedestalTypeName == 'FromExtractor':
                self.pedestalType = 1
            elif pedestalTypeName == 'FromExtractorRndm':
                self.pedestalType = 2
            else:
                raise ValueError("pedestalType must be chosen from 'Fundamental', 'FromExtractor', or 'FromExtractorRndm'")
        else:
            self.pedestalType = 2
        
        self.sample_times_hot = [None, None]
        self.n_samples_hot = np.zeros(2, dtype=np.int16) - 1

        self.sample_ranges_dead = [None, None]
        self.n_samples_dead = np.zeros(2, dtype=np.int16) - 1
        #self.pedestal_info = pedestal_info
        
    def _check_pedestal_rms(self, charge_std):

        if (len(charge_std)) != self.n_camera_pixels:
            print(len(charge_std))
            print(self.n_camera_pixels)
            raise ValueError("charge_std must be an array of length equal to number of MAGIC camera pixels")

        meanrms = 0.
        npix = 0
        for i in range(self.n_camera_pixels):

            if (charge_std[i] <= 0 or charge_std[i] >= 200 * self._getpixratiosqrt(i)):
                continue

            #const Byte_t aidx = (*fGeomCam)[i].GetAidx();
            meanrms += charge_std[i]
            npix += 1

        # if no pixel has a minimum signal, return
        if meanrms == 0:
            return False;

        meanrms /= npix

        meanrms2 = 0.
        varrms2 = 0.
        npix = 0
        for i in range(self.n_camera_pixels):

            # Calculate the corrected means:
    
            if (charge_std[i] <= 0.5 * meanrms or charge_std[i] >= 1.5 * meanrms):
                continue
    
            meanrms2 += charge_std[i]
            varrms2 += charge_std[i]**2
            npix += 1

        # if no pixel has a minimum signal, return
        lolim1 = 0
        lolim2 = 0  # Precalcualtion of limits
        uplim1 = 0
        uplim2 = 0  # for speeed reasons

        if npix == 0 or meanrms2 == 0:
            return False

        meanrms2 /= npix

        if self.pedestalLevel > 0:
            lolim1 = meanrms2 / self.pedestalLevel
            uplim1 = meanrms2 * self.pedestalLevel

        if self.pedestalLevelVariance > 0:
            varrms2 /= npix
            varrms2 = np.sqrt(varrms2 - meanrms2 * meanrms2 )

            lolim2 = meanrms2 - self.pedestalLevelVariance * varrms2
            uplim2  = meanrms2 + self.pedestalLevelVariance * varrms2

        bads = 0
    
        # Blind the Bad Pixels
        for i in range(self.n_camera_pixels):    
            if ((self.pedestalLevel <= 0             or (charge_std[i] > lolim1 and charge_std[i] <= uplim1))
                and (self.pedestalLevelVariance <= 0 or (charge_std[i] > lolim2 and charge_std[i] <= uplim2))):
                continue

            self.badrmspixels[i] = True
            if (charge_std[i] <= lolim1 or charge_std[i] <= lolim2):
                self.coldpixels[i] = True
            elif (charge_std[i] > uplim1 or charge_std[i] > uplim2):
                self.hotpixels[i] = True
            bads += 1

        return True;

    def _getpixratiosqrt(self, i_pix):
#         i_pixzero = np.where(self.geom.pix_id == 0)[0][0]
#         return np.sqrt(self.geom.pix_area[i_pix] / self.geom.pix_area[i_pixzero])
        return 1.

    def get_badrmspixel_mask(self, event):
        badrmspixel_mask = [[None],[None]]
        event_time = event.trig.gps_time.unix
        subrun_start_time = event.meta['subrun_start_unixtime']

        for tel_id in event.mon.tels_with_data:

            self._check_pedvar_fields(tel_id, event)

            # now find monitoring data sample matching to this event by time stamp:
            if event_time <= self.sample_times_hot[tel_id - 1][0]:
                i_min = 0
            else:
                i_min = np.where(event_time > self.sample_times_hot[tel_id - 1])[0][-1]
                #if self.sample_times_hot[tel_id - 1][i_min] > subrun_start_time and i_min != self.n_samples_hot[tel_id - 1] - 1:
                #    i_min += 1
                #if event.meta['number_subrun'] == 2: 
                #    i_min -= 1
                #if self.sample_times_hot[tel_id - 1][i_min]- subrun_start_time < 0:
                #print(event.meta['number_subrun'], event_time - self.sample_times_hot[tel_id - 1][i_min], self.sample_times_hot[tel_id - 1][i_min]- subrun_start_time)
            badrmspixel_mask[tel_id - 1] = event.mon.tel[tel_id].pedestal.charge_std_outliers[i_min]

        return badrmspixel_mask

    def get_hotpixel_mask(self, event):
        hotpixel_mask = [[None],[None]]
        event_time = event.trig.gps_time.unix

        for tel_id in event.mon.tels_with_data:
#
            self._check_pedvar_fields(tel_id, event)
            i_min = np.where(event_time > self.sample_times_hot[tel_id - 1])[0][-1]
            hotpixel_mask[tel_id - 1] = event.mon.tel[tel_id].pedestal.charge_std_outliers_hot[i_min]

        return hotpixel_mask
    
    def get_coldpixel_mask(self, event):
        coldpixel_mask = [[None],[None]]
        event_time = event.trig.gps_time.unix

        for tel_id in event.mon.tels_with_data:
#
            self._check_pedvar_fields(tel_id, event)

            i_min = np.where(event_time > self.sample_times_hot[tel_id - 1])[0][-1]
            coldpixel_mask[tel_id - 1] = event.mon.tel[tel_id].pedestal.charge_std_outliers_cold[i_min]

        return coldpixel_mask


    def get_badrmspixelindices(self, event):
        badrmspixelindices = [[None],[None]]
        badrmspixelmask = self.get_badrmspixelmask(self, event)
        for i, badrmspixelmask_tel_i in enumerate(badrmspixelmask):
            if badrmspixelmask_tel_i != [None]:
                badrmspixelindices[i] = np.where(badrmspixelmask_tel_i)[0]
        return badrmspixelindices

    def _check_pedvar_fields(self, tel_id, event):
        if self.n_samples_hot[tel_id - 1] == -1 or np.isscalar(event.mon.tel[tel_id].pedestal.charge_std_outliers) or self.is_update[tel_id - 1] == True:
            self.n_samples_hot[tel_id - 1] = len(event.mon.tel[tel_id].pedestal.sample_time)
            self.sample_times_hot[tel_id - 1] = np.zeros(self.n_samples_hot[tel_id - 1])
            for i in range(self.n_samples_hot[tel_id - 1]):
                self.sample_times_hot[tel_id - 1][i] = event.mon.tel[tel_id].pedestal.sample_time[i].unix
            
            # calculate only once the hot pixel array of the monitoring data:
            print("Update hot pixels for M%d..." % tel_id, end =" ")
            event.mon.tel[tel_id].pedestal.charge_std_outliers = []
            event.mon.tel[tel_id].pedestal.charge_std_outliers_hot = []
            event.mon.tel[tel_id].pedestal.charge_std_outliers_cold = []
            for i_sample in range(self.n_samples_hot[tel_id - 1]):
                self.badrmspixels = np.zeros(self.n_camera_pixels, dtype=np.bool)
                self.hotpixels = np.zeros(self.n_camera_pixels, dtype=np.bool)
                self.coldpixels = np.zeros(self.n_camera_pixels, dtype=np.bool)
                charge_std = event.mon.tel[tel_id].pedestal.charge_std[self.pedestalType][i_sample]
                self._check_pedestal_rms(charge_std)
                event.mon.tel[tel_id].pedestal.charge_std_outliers.append(self.badrmspixels)
                event.mon.tel[tel_id].pedestal.charge_std_outliers_hot.append(self.hotpixels)
                event.mon.tel[tel_id].pedestal.charge_std_outliers_cold.append(self.coldpixels)
            event.mon.tel[tel_id].pedestal.charge_std_outliers = np.array(event.mon.tel[tel_id].pedestal.charge_std_outliers, dtype=np.bool)
            event.mon.tel[tel_id].pedestal.charge_std_outliers_hot = np.array(event.mon.tel[tel_id].pedestal.charge_std_outliers_hot, dtype=np.bool)
            event.mon.tel[tel_id].pedestal.charge_std_outliers_cold = np.array(event.mon.tel[tel_id].pedestal.charge_std_outliers_cold, dtype=np.bool)
            self.is_update[tel_id - 1] = False
            print("done.")

    def get_deadpixel_mask(self, event):
        deadpixel_mask = [[None],[None]]
        event_time = event.trig.gps_time.unix

        for tel_id in event.mon.tels_with_data:

            if self.n_samples_dead[tel_id - 1] == -1:
                self.n_samples_dead[tel_id - 1] = len(event.mon.tel[tel_id].pixel_status.sample_time_range)
                self.sample_ranges_dead[tel_id - 1] = np.zeros(shape=(self.n_samples_dead[tel_id - 1],2))
                for i in range(self.n_samples_dead[tel_id - 1]):
                    self.sample_ranges_dead[tel_id - 1][i,0] = event.mon.tel[tel_id].pixel_status.sample_time_range[i][0].unix
                    self.sample_ranges_dead[tel_id - 1][i,1] = event.mon.tel[tel_id].pixel_status.sample_time_range[i][1].unix

            # now find sample:
            i_min_dead = np.where(event_time >= self.sample_ranges_dead[tel_id - 1][:,0])[0][-1]

            deadpixel_mask[tel_id - 1] = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[i_min_dead]


        return deadpixel_mask
    
#     def get_deadpixel_mask(self, event):
#         deadpixel_mask = [[None],[None]]
#         event_time = event.trig.gps_time.unix
# 
#         for tel_id in event.mon.tels_with_data:
# 
#             if self.n_samples_dead[tel_id - 1] == -1:
#                 self.n_samples_dead[tel_id - 1] = len(event.mon.tel[tel_id].pixel_status.sample_time_range)
#                 self.sample_ranges_dead[tel_id - 1] = np.zeros(shape=(self.n_samples_dead[tel_id - 1],2))
#                 for i in range(self.n_samples_dead[tel_id - 1]):
#                     self.sample_ranges_dead[tel_id - 1][i,0] = event.mon.tel[tel_id].pixel_status.sample_time_range[i][0].unix
#                     self.sample_ranges_dead[tel_id - 1][i,1] = event.mon.tel[tel_id].pixel_status.sample_time_range[i][1].unix
# 
#             if self.n_samples_hot[tel_id - 1] == -1:
#                 self.n_samples_hot[tel_id - 1] = len(event.mon.tel[tel_id].pedestal.sample_time)
#                 self.sample_times_hot[tel_id - 1] = np.zeros(self.n_samples_hot[tel_id - 1])
#                 for i in range(self.n_samples_hot[tel_id - 1]):
#                     self.sample_times_hot[tel_id - 1][i] = event.mon.tel[tel_id].pedestal.sample_time[i].unix
# 
#             # now find sample:
#             i_min_dead = np.where(event_time >= self.sample_ranges_dead[tel_id - 1][:,0])[0][-1]
#             i_min_hot = np.where(event_time > self.sample_times_hot[tel_id - 1])[0][-1]
#             # find end of hot interval:
#             try:
#                 time_hot_max = self.sample_times_hot[tel_id - 1][i_min_hot + 1]
#                 if i_min_dead ==0 and time_hot_max >= self.sample_ranges_dead[tel_id - 1][i_min_dead, 1]:
#                     deadpixel_mask[tel_id - 1] = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[i_min_dead + 1]
#                 elif i_min_dead !=0 and time_hot_max >= (self.sample_ranges_dead[tel_id - 1][i_min_dead, 0] + self.sample_ranges_dead[tel_id - 1][i_min_dead, 1])/2:
#                     deadpixel_mask[tel_id - 1] = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[i_min_dead + 1]
#                 else:
#                     deadpixel_mask[tel_id - 1] = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[i_min_dead]
#             except:
#                 deadpixel_mask[tel_id - 1] = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[i_min_dead]
# 
# 
#         return deadpixel_mask

    def get_charge_std(self, event):
        charge_std = [[None],[None]]
        event_time = event.trig.gps_time.unix

        for tel_id in event.mon.tels_with_data:

            if self.n_samples_hot[tel_id - 1] == -1:
                self.n_samples_hot[tel_id - 1] = len(event.mon.tel[tel_id].pedestal.sample_time)
                self.sample_times_hot[tel_id - 1] = np.zeros(self.n_samples_hot[tel_id - 1])
                for i in range(self.n_samples_hot[tel_id - 1]):
                    self.sample_times_hot[tel_id - 1][i] = event.mon.tel[tel_id].pedestal.sample_time[i].unix

            i_min = np.where(event_time > self.sample_times_hot[tel_id - 1])[0][-1]
            charge_std[tel_id - 1] = event.mon.tel[tel_id].pedestal.charge_std[self.pedestalType][i_min]

        return charge_std
