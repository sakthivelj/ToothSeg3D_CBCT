import sys
import os
from typing import Optional, Union
import numpy as np
import torch
from raster_geometry import sphere
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.measure import label, regionprops

if __name__ == "__main__":
    sys.path.insert(1, os.path.join(os.getcwd(), 'ToothSwinUNETR'))
from utils.data_augmentation import erosion3d, dilation3d


def morph_open(tensor : torch.tensor, e_k : tuple, d_k : Optional[Union[tuple, None]] = None,
               kernel_geom : str = 'cube', device='cpu'):
    if d_k is None:
        d_k = e_k
    if kernel_geom == 'cube':
        e_kernel = torch.ones((1,1)+e_k, device=device)
        d_kernel = torch.ones((1,1)+d_k, device=device)
    if kernel_geom == 'sphere':
        sphere_e = sphere(shape=e_k, radius = e_k[0]//2, smoothing=False, position=0.5).astype(np.float32)
        sphere_d = sphere(shape=e_k, radius = d_k[0]//2, smoothing=False, position=0.5).astype(np.float32)
        sphere_e = torch.from_numpy(sphere_e).reshape((1,1)+e_k)
        sphere_d = torch.from_numpy(sphere_d).reshape((1,1)+d_k)
        e_kernel = sphere_e.to(device)
        d_kernel = sphere_d.to(device)

    eroded = erosion3d(tensor, e_kernel)
    opened = dilation3d(eroded, d_kernel)
    return opened

def post_processing_segmentation(model_output : np.array, instance_ground_truth : Optional[Union[np.array, None]] = None,
                                 distance_treshold : float = 0.2, opening_kernel : int = 5, 
                                 instance_min_volume : int = 1000, sampling = [1,1,1], device : str ='cpu'):
    
    #binarize output
    binary_array = np.copy(model_output)
    binary_array[binary_array>1]=1
    
    #calculate distance transform with reduced z-axis sampling to emphasise scans with teeth in bite position
    distance_matrix = distance_transform_edt(binary_array, sampling = sampling)

    #normalize to 0-1
    distance_matrix = distance_matrix/distance_matrix.max()
    
    #treshold distance to get seeds/markers
    distance_matrix[distance_matrix<=distance_treshold]=0
    distance_matrix[distance_matrix>distance_treshold]=1

    #eliminate outliers, make sure to separate teeth with morphological opening
    #recommended to perform dillation and erosion on cuda
    kernel_shape =  (opening_kernel,)*3
    distance_matrix_tensor = torch.from_numpy(distance_matrix).unsqueeze(0).to(dtype=torch.float32, device=device)
    distance_matrix_opened = morph_open(distance_matrix_tensor, kernel_shape, kernel_geom='sphere', device=device).squeeze().cpu().numpy().astype(np.int16)

    #label potential seeds  and filter them based on volume 
    labeled_image, count = label(distance_matrix_opened, connectivity=1, return_num=True)
    objects = regionprops(labeled_image)
    filtered_objects = [obj for obj in objects if obj['area']< instance_min_volume]

    #zero values inside of bounding boxes of objects not meeting volume criteria
    for i in filtered_objects:
        distance_matrix_opened[i.bbox[0]:i.bbox[3], i.bbox[1]:i.bbox[4], i.bbox[2]:i.bbox[5]]=0

    #relabel tooth instances and calculate distance matrix with equal sampling for all axes
    instances = label(distance_matrix_opened)
    dst = distance_transform_edt(binary_array, sampling = sampling)

    #apply watershed algorithm based on negative distance using instances as seeds and masked by binary segmentation
    instance_masks = watershed(-dst, instances, mask=binary_array)

    #calculate properties of instance masks and perform majority voting based on original model output
    output = np.zeros_like(instance_masks)
    instance_masks_props = regionprops(instance_masks)

    for idx, i in enumerate(instance_masks_props):
   
        if instance_ground_truth is None:
            #get tooth instance voxels based on region of interest from original model output
            pred_instance  = model_output[i.bbox[0]:i.bbox[3], i.bbox[1]:i.bbox[4], i.bbox[2]:i.bbox[5]][i.image].astype(np.int8)
            #get class with the most votes (ignore background class)
            votes = np.bincount(pred_instance)
            if len(votes)>1:
                majority_class = np.argmax(votes[1:])+1
            else:
                majority_class = 0
        else:
            gt_instance  = instance_ground_truth[i.bbox[0]:i.bbox[3], i.bbox[1]:i.bbox[4], i.bbox[2]:i.bbox[5]][i.image].astype(np.int8)
            votes = np.bincount(gt_instance)
            if len(votes)>1:
                majority_class = np.argmax(votes[1:])+1
            else:
                majority_class = 0
        #relabel instance voxels based on winner
        output[i.bbox[0]:i.bbox[3], i.bbox[1]:i.bbox[4], i.bbox[2]:i.bbox[5]][i.image] = majority_class

    return output

