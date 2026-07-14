import numpy as np
import cv2
from skimage import measure




class PD_FA():
    def __init__(self, ):
        super(PD_FA, self).__init__()
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0

    def update(self, preds, labels, size):
        if np.max(preds) > 1 or np.min(preds) < 0:
            # return
            preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds))  # normalize output to 0-1
            # print('normalize output to 0-1')
        predits = np.array(preds > 0.5).astype('int64')
        labelss = np.array(labels).astype('int64')

        # predits = preds> 0.5
        # labelss = labels

        image = measure.label(predits, connectivity=2)  # Label 8-connected regions
        coord_image = measure.regionprops(image)  # Operate on different connected regions
        label = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label)

        self.target += len(coord_label)
        self.image_area_total = []
        self.image_area_match = []
        self.distance_match = []
        self.dismatch = []

        for K in range(len(coord_image)):
            area_image = np.array(coord_image[K].area)  # Predicted map's number of pixels in different connected regions
            self.image_area_total.append(area_image)  # Sequence of number of pixels in different connected regions of predicted map

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))  # Mask's coordinates of pixel centroids in different connected regions
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))  # Predicted map's coordinates of pixel centroids in different connected regions
                distance = np.linalg.norm(centroid_image - centroid_label)  # Distance between the two centroids
                area_image = np.array(coord_image[m].area)
                if distance < 3:
                    self.distance_match.append(distance)  # Sequence of distances less than 3
                    self.image_area_match.append(area_image)  # Sequence of predicted image pixel count for distance less than 3

                    del coord_image[m]
                    break

        self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]  #
        self.dismatch_pixel += np.sum(self.dismatch)
        self.all_pixel += size[0] * size[1]
        self.PD += len(self.distance_match)

    def get(self):
        Final_FA = self.dismatch_pixel / self.all_pixel
        Final_PD = self.PD / self.target
        return Final_PD, float(Final_FA)

    def reset(self):
        self.image_area_total = []
        self.image_area_match = []
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0


# class PD_FA():
#     def __init__(self,):
#         super(PD_FA, self).__init__()
#         self.image_area_total = []
#         self.image_area_match = []
#         self.dismatch_pixel = 0
#         self.all_pixel = 0
#         self.PD = 0
#         self.target= 0
#     def update(self, preds, labels, size):
#         predits  = np.array((preds).cpu()).astype('int64')
#         labelss = np.array((labels).cpu()).astype('int64')
#
#         image = measure.label(predits, connectivity=2)
#         coord_image = measure.regionprops(image)
#         label = measure.label(labelss , connectivity=2)
#         coord_label = measure.regionprops(label)
#
#         self.target    += len(coord_label)
#         self.image_area_total = []
#         self.distance_match   = []
#         self.dismatch         = []
#
#         for K in range(len(coord_image)):
#             area_image = np.array(coord_image[K].area)
#             self.image_area_total.append(area_image)
#
#         true_img = np.zeros(predits.shape)
#         for i in range(len(coord_label)):
#             centroid_label = np.array(list(coord_label[i].centroid))
#             for m in range(len(coord_image)):
#                 centroid_image = np.array(list(coord_image[m].centroid))
#                 distance = np.linalg.norm(centroid_image - centroid_label)
#                 area_image = np.array(coord_image[m].area)
#                 if distance < 3:
#                     self.distance_match.append(distance)
#                     true_img[coord_image[m].coords[:,0], coord_image[m].coords[:,1]] = 1
#                     del coord_image[m]
#                     break
#
#         self.dismatch_pixel += (predits - true_img).sum()
#         self.all_pixel +=size[0]*size[1]
#         self.PD +=len(self.distance_match)
#
#     def get(self):
#         Final_FA =  self.dismatch_pixel / self.all_pixel
#         Final_PD =  self.PD /self.target
#         return Final_PD, float(Final_FA.cpu().detach().numpy())
#
#     def reset(self):
#         self.FA  = np.zeros([self.bins+1])
#         self.PD  = np.zeros([self.bins+1])
