import numpy as np
from ctapipe.instrument import CameraGeometry

__all__ = [
    "MAGICBadPixelsCalc",
]

class MAGICBadPixelsCalc():

    def __init__(self, is_simulation, camera=None, config=None, tool=None, **kwargs):

        # MAGIC telescope description
        if camera == None:
            camera = CameraGeometry.from_name('MAGICCam')

        self.n_camera_pixels = camera.n_pixels

        # initialize bad pixel mask. Will updated for each telescope/sample by
        # self._check_pedvar_fields()
        self.badrmspixel_mask = np.zeros(self.n_camera_pixels, dtype=np.bool)

        self.config = config

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

        self.current_obs_id = -1

        # Pedestal sample times and outlier masks are reduced to the unique
        # outlier masks: In MARS files, mayn duplicate entries are present,
        # and removing them onces significantly speeds up searching the correct
        # mask for a given event.
        self.sample_times_ped = [[], []]
        self.n_samples_ped = np.zeros(2, dtype=np.int16) - 1
        self.charge_std_outliers = [[], []]
        #self.charge_std = [[], []]

        # Dead pixels masks change for every subrun and are directly used from
        # the MARS data
        self.sample_ranges_dead = [None, None]
        self.n_samples_dead = np.zeros(2, dtype=np.int16) - 1

        # Allow processing of MCs (do nothing, but also don't crash)
        self.is_mc = is_simulation

    def _check_new_run(self, event):
        """
        Initializes or resets for each new run subrun-wise dead pixel samples
        or pedestal info with computed outlier masks.
        """

        if event.index.obs_id != self.current_obs_id:

            self.sample_times_ped = [[], []]
            self.n_samples_ped = np.zeros(2, dtype=np.int16) - 1
            self.charge_std_outliers = [[], []]
            #self.charge_std = [[], []]

            self.sample_ranges_dead = [None, None]
            self.n_samples_dead = np.zeros(2, dtype=np.int16) - 1

            self.current_obs_id = event.index.obs_id

    def _check_pedestal_rms(self, charge_std):
        """
        This internal method calculates the pedestal outlier pixels depending on the
        values in self.config. Corresponds to mbadpixels/MBadPixelsCalc::CheckPedestalRms()

        Returns
        -------
        self.badrmspixel_mask: Mask with the Pedestal RMS outliers.
        """

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

            self.badrmspixel_mask[i] = True
#             if (charge_std[i] <= lolim1 or charge_std[i] <= lolim2):
#                 self.coldpixels[i] = True
#             elif (charge_std[i] > uplim1 or charge_std[i] > uplim2):
#                 self.hotpixels[i] = True
            bads += 1

        return True;

    def _getpixratiosqrt(self, i_pix):
#         i_pixzero = np.where(self.geom.pix_id == 0)[0][0]
#         return np.sqrt(self.geom.pix_area[i_pix] / self.geom.pix_area[i_pixzero])
        return 1.

    def get_badrmspixel_mask(self, event):
        """
        Fetch the bad RMS pixel mask for a given event, that is the event time.

        Returns
        -------
        badrmspixel_mask: has two dimensions: Masks for M1 and/or M2.
        """

        badrmspixel_mask = [None, None]

        if self.is_mc:
            for tel_id in event.trigger.tels_with_trigger:
                badrmspixel_mask[tel_id - 1] = np.zeros(self.n_camera_pixels, dtype=np.bool)
            return badrmspixel_mask

        self._check_new_run(event)

        event_time = event.trigger.time.unix

        for tel_id in event.trigger.tels_with_trigger:

            self._check_pedvar_fields(tel_id, event)

            # now find monitoring data sample matching to this event by time stamp:
            if event_time <= self.sample_times_ped[tel_id - 1][0]:
                i_min = 0
            else:
                i_min = np.where(event_time > self.sample_times_ped[tel_id - 1])[0][-1]

            badrmspixel_mask[tel_id - 1] = self.charge_std_outliers[tel_id - 1][i_min]

        return badrmspixel_mask

    def get_badrmspixel_indices(self, event):
        """
        Quick workaround to get the pixel IDs and not the mask

        Returns
        -------
        badrmspixel_indices
        """
        badrmspixel_indices = [[None],[None]]

        if self.is_mc:
            return badrmspixel_indices

        badrmspixel_mask = self.get_badrmspixel_mask(self, event)
        for i, badrmspixelmask_tel_i in enumerate(badrmspixel_mask):
            if badrmspixelmask_tel_i != [None]:
                badrmspixel_indices[i] = np.where(badrmspixelmask_tel_i)[0]
        return badrmspixel_indices

    def _check_pedvar_fields(self, tel_id, event):
        """
        Update the pedestal RMS outliers. Does it only once after initializing the class.
        and reduces the samples to the unique ones.

        Returns
        -------
         self.sample_times_ped, self.charge_std_outliers
        """

        if self.n_samples_ped[tel_id - 1] == -1:

            self.n_samples_ped[tel_id - 1] = len(event.mon.tel[tel_id].pedestal.sample_time)

            # calculate only once the hot pixel array of the monitoring data:
            print("Update hot pixels for M%d..." % tel_id, end =" ")
            event.mon.tel[tel_id].pedestal.charge_std_outliers = []
            for i_sample in range(self.n_samples_ped[tel_id - 1]):
                charge_std = event.mon.tel[tel_id].pedestal.charge_std[self.pedestalType][i_sample]
                if i_sample == 0:
                    charge_std_last = None
                else:
                    charge_std_last = event.mon.tel[tel_id].pedestal.charge_std[self.pedestalType][i_sample -1]

                if not np.array_equal(charge_std, charge_std_last):
                    self.sample_times_ped[tel_id - 1].append(event.mon.tel[tel_id].pedestal.sample_time[i_sample].unix)
                    self.badrmspixel_mask = np.zeros(self.n_camera_pixels, dtype=np.bool)
                    self._check_pedestal_rms(charge_std)
                    self.charge_std_outliers[tel_id - 1].append(self.badrmspixel_mask)
                    #self.charge_std[tel_id - 1].append(charge_std)

            self.sample_times_ped[tel_id - 1] = np.array(self.sample_times_ped[tel_id - 1])
            self.charge_std_outliers[tel_id - 1] = np.array(self.charge_std_outliers[tel_id - 1], dtype=np.bool)
            #self.charge_std[tel_id - 1] = np.array(self.charge_std[tel_id - 1])
            print("done.")

    def get_deadpixel_mask(self, event):
        """
        Fetch the subrun-wise defined dead pixels for a given event, that is the event time.

        Returns
        -------
        deadpixel_mask: has two dimensions: Masks for M1 and/or M2.
        """

        deadpixel_mask = [[None],[None]]

        if self.is_mc:
            for tel_id in event.trigger.tels_with_trigger:
                deadpixel_mask[tel_id - 1] = np.zeros(self.n_camera_pixels, dtype=np.bool)
            return deadpixel_mask

        self._check_new_run(event)

        event_time = event.trigger.time.unix

        for tel_id in event.trigger.tels_with_trigger:

            if self.n_samples_dead[tel_id - 1] == -1:
                self.n_samples_dead[tel_id - 1] = len(event.mon.tel[tel_id].pixel_status.sample_time_range)
                self.sample_ranges_dead[tel_id - 1] = np.zeros(shape=(self.n_samples_dead[tel_id - 1],2))
                for i in range(self.n_samples_dead[tel_id - 1]):
                    self.sample_ranges_dead[tel_id - 1][i,0] = event.mon.tel[tel_id].pixel_status.sample_time_range[i][0].unix
                    self.sample_ranges_dead[tel_id - 1][i,1] = event.mon.tel[tel_id].pixel_status.sample_time_range[i][1].unix

            # now find sample:
            indices_time_dead = np.where(event_time >= self.sample_ranges_dead[tel_id - 1][:,0])[0]
            if indices_time_dead.size:
                i_min_dead = indices_time_dead[-1]
            else:
                i_min_dead = 0

            deadpixel_mask[tel_id - 1] = event.mon.tel[tel_id].pixel_status.hardware_failing_pixels[i_min_dead]

        return deadpixel_mask

    def get_badpixel_mask(self, event):
        """
        Fetch the union of bad RMS and bad pixels for a given event, that is the event time.

        Returns
        -------
        badpixel_mask: has two dimensions: Masks for M1 and/or M2.
        """
        badpixel_mask = [[None],[None]]

        if self.is_mc:
            for tel_id in event.trigger.tels_with_trigger:
                badpixel_mask[tel_id - 1] = np.zeros(self.n_camera_pixels, dtype=np.bool)
            return badpixel_mask

        badrmspixel_mask = self.get_badrmspixel_mask(event)
        deadpixel_mask = self.get_deadpixel_mask(event)

        for tel_id in event.trigger.tels_with_trigger:
            badpixel_mask[tel_id - 1] = np.logical_or(badrmspixel_mask[tel_id - 1], deadpixel_mask[tel_id - 1])

        return badpixel_mask
#     def get_charge_std(self, event):
#         """
#         Fetch the pedestal RMS pixel values for a given event, that is the event time.
# 
#         Returns
#         -------
#         charge_std: has two dimensions: M1 and/or M2.
#         """
# 
#         charge_std = [None, None]
#         event_time = event.trigger.time.unix
# 
#         for tel_id in event.trigger.tels_with_trigger:
# 
#             self._check_pedvar_fields(tel_id, event)
# 
#             # now find monitoring data sample matching to this event by time stamp:
#             if event_time <= self.sample_times_ped[tel_id - 1][0]:
#                 i_min = 0
#             else:
#                 i_min = np.where(event_time > self.sample_times_ped[tel_id - 1])[0][-1]
# 
#             charge_std[tel_id - 1] = self.charge_std[tel_id - 1][i_min]
# 
#         return charge_std
