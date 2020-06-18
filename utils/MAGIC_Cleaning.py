import glob, sys, re
import itertools
import copy
import numpy as np
from scipy.sparse.csgraph import connected_components

class magic_clean():

    def __init__(self, camera, configuration):

        self.configuration = configuration
        self.camera = camera

        if configuration['usesum']:
            self.NN2, self.groups2NN = self.GetListOfNN(NN_size = 2)
            self.NN3, self.groups3NN = self.GetListOfNN(NN_size = 3)
            self.NN4, self.groups4NN = self.GetListOfNN(NN_size = 4)

    def GetListOfNN(self,NN_size = 2):

        NN = []
        pixels = list(range(self.camera.n_pixels))
        for pixel in pixels:

            neighbors = self.camera.neighbor_matrix_sparse[pixel].indices

            if len(neighbors) < NN_size:
                continue

            combos = list(itertools.combinations(neighbors,NN_size-1))
            for combo in combos:

                arr = list(combo)
                arr.append(pixel)

                if NN_size == 2:
                    NN.append(sorted(arr))

                if NN_size == 3:
                    neigh0 = self.camera.neighbor_matrix_sparse[arr[0]].indices
                    neigh1 = self.camera.neighbor_matrix_sparse[arr[1]].indices
                    neigh_set = np.asarray(list(set(neigh0) & set(neigh1)))
                    if len(neigh_set) != 2:
                        continue
                    shared_pixel = neigh_set[neigh_set!=pixel]
                    if shared_pixel in neighbors:
                        continue

                    NN.append(sorted(arr))

                if NN_size == 4:

                    neigh0 = self.camera.neighbor_matrix_sparse[arr[0]].indices
                    neigh1 = self.camera.neighbor_matrix_sparse[arr[1]].indices
                    neigh2 = self.camera.neighbor_matrix_sparse[arr[2]].indices

                    if ((arr[0] in neigh1) or (arr[0] in neigh2)) and ((arr[1] in neigh0) or (arr[1] in neigh2)) and ((arr[2] in neigh0) or (arr[2] in neigh1)):
                        pass
                    else:
                        continue

                    NN.append(sorted(arr))

        return np.unique(NN,axis=0), len(np.unique(NN,axis=0))

    def clean_image(self,event_image,event_pulse_time):

        self.event_image = event_image
        self.event_pulse_time = event_pulse_time

        if self.configuration['usesum']:
            clean_mask = self.magic_clean_step1Sum()
        else:
            clean_mask = self.magic_clean_step1()

        clean_mask = self.magic_clean_step2(clean_mask)

        clean_mask = self.magic_clean_step3(clean_mask)

        return clean_mask

    def group_calculation(self, mask, NN, clipNN, windowNN, thresholdNN):

        meantime = 0.0
        totcharge = 0.0

        charge = self.event_image[NN]
        charge[charge > clipNN] = clipNN

        totcharge = np.sum(charge,axis=1)
        meantime = np.sum(charge * self.event_pulse_time[NN],axis=1)/totcharge

        test = np.tile(meantime,(len(NN[0]),1)).transpose()

        timeok = np.all(np.fabs(test - self.event_pulse_time[NN]) < windowNN,axis=1)

        selection = (timeok) * (totcharge > thresholdNN)
        mask[NN[selection]] = True

        return mask

    def magic_clean_step1Sum(self):

        fSumThresh2NNPerPixel = 2.0
        fSumThresh3NNPerPixel = 1.5
        fSumThresh4NNPerPixel = 1.0

        fWindow2NN            = 2.0
        fWindow3NN            = 2.0
        fWindow4NN            = 4.0

        sumthresh2NN = fSumThresh2NNPerPixel * 2 * self.configuration['picture_thresh'];
        sumthresh3NN = fSumThresh3NNPerPixel * 3 * self.configuration['picture_thresh'];
        sumthresh4NN = fSumThresh4NNPerPixel * 4 * self.configuration['picture_thresh'];

        clip2NN = 2.2  * sumthresh2NN/2.
        clip3NN = 1.05 * sumthresh3NN/3.
        clip4NN = 1.05 * sumthresh4NN/4.

        mask = np.asarray([False]*len(self.event_image))
        mask = self.group_calculation(mask,self.NN2, clip2NN,fWindow2NN, sumthresh2NN)
        mask = self.group_calculation(mask,self.NN3, clip3NN,fWindow3NN, sumthresh3NN)
        mask = self.group_calculation(mask,self.NN4, clip4NN,fWindow4NN, sumthresh4NN)

        return np.asarray(mask)

    def magic_clean_step1(self):
        mask = self.event_image <= self.configuration['picture_thresh']
        return ~mask

    def magic_clean_step2(self, mask):
        
        if np.sum(mask) == 0:
            return mask

        else:
            pixels_to_remove = []

            neighbors = self.camera.neighbor_matrix_sparse
            clean_neighbors = neighbors[mask][:, mask]
            num_islands, labels = connected_components(clean_neighbors, directed=False)

            island_ids = np.zeros(self.camera.n_pixels)
            island_ids[mask] = labels + 1

            # Finding the islands "sizes" (total charges)
            island_sizes = np.zeros(num_islands)
            for i in range(num_islands):
                island_sizes[i] = self.event_image[mask][labels == i].sum()
              
            # Disabling pixels for all islands save the brightest one
            brightest_id = island_sizes.argmax() + 1

            if self.configuration['usetime']:
                brightest_pixel_times = self.event_pulse_time[mask & (island_ids == brightest_id)]
                brightest_pixel_charges = self.event_image[mask & (island_ids == brightest_id)]

                brightest_time = np.sum(brightest_pixel_times * brightest_pixel_charges**2) / np.sum(brightest_pixel_charges**2)

                time_diff = np.abs(self.event_pulse_time - brightest_time)

                mask[(self.event_image > 2*self.configuration['picture_thresh']) & (time_diff > 2*self.configuration['max_time_off'])] = False
                mask[(self.event_image < 2*self.configuration['picture_thresh']) & (time_diff > self.configuration['max_time_off'])] = False

            mask = self.single_island(self.camera,mask,self.event_image)
            
        return mask

    def magic_clean_step3(self, mask):

        thing = []
        core_mask = mask.copy()
        
        pixels_with_picture_neighbors_matrix = self.camera.neighbor_matrix_sparse

        for pixel in np.where(self.event_image)[0]:
            
            if pixel in np.where(core_mask)[0]:
                continue

            if self.event_image[pixel] <= self.configuration['boundary_thresh']:
                continue
            
            hasNeighbor = False
            if self.configuration['usetime']:
                
                neighbors = pixels_with_picture_neighbors_matrix[pixel].indices
                
                for neighbor in neighbors:
                    if neighbor not in np.where(core_mask)[0]:
                        continue
                    time_diff = np.abs(self.event_pulse_time[neighbor] - self.event_pulse_time[pixel])
                    if time_diff < self.configuration['max_time_diff']:
                        hasNeighbor = True
                        break
                if not hasNeighbor:
                    continue
                
            if not pixels_with_picture_neighbors_matrix.dot(core_mask)[pixel]:
                continue
            
            thing.append(pixel)
            
        mask[thing] = True

        return mask

    def single_island(self, camera, mask, image):
        pixels_to_remove = []
        neighbors = camera.neighbor_matrix
        for pix_id in np.where(mask)[0]:
            if len(set(np.where(neighbors[pix_id] & mask)[0])) == 0:
                pixels_to_remove.append(pix_id)
        mask[pixels_to_remove] = False
        return mask
