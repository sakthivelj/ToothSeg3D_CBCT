import sys
import tqdm
import glob
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import pandas as pd
from natsort import natsorted
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from monai.data import DataLoader, Dataset, decollate_batch, ThreadDataLoader
from monai.inferers import sliding_window_inference
from monai.networks.nets.unet import UNet
from monai.networks.nets import AttentionUnet
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import (HausdorffDistanceMetric, SurfaceDistanceMetric, MeanIoU, DiceMetric)
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score 
from monai.networks.utils import one_hot
from monai.config import print_config
from monai.utils import set_determinism
from monai.transforms import Compose, AsDiscrete
from monai.inferers.inferer import SlidingWindowInferer
from monai.transforms import (
    Compose,
    AsDiscreteD,
    ActivationsD,
    CropForegroundD,
    InvertD,
    EnsureChannelFirstD,
    SaveImageD,
    OrientationD,
    LoadImageD,
    ScaleIntensityRangeD,
    KeepLargestConnectedComponentD,
    SpacingD,
    ToTensorD,
    EnsureTypeD,
    ResizeD,
    Resize,
    ThresholdIntensityD,
    ToDeviceD,
    SaveImage
)
if __name__ == "__main__":
    sys.path.insert(1, os.path.join(os.getcwd(), 'ToothSwinUNETR'))
from models.swin_unetr_mlt import SwinUNETR
from monai.networks.nets import UNet
from utils.postprocessing import post_processing_segmentation
from utils.data_augmentation import CropForegroundFixedD

#CUDA
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.set_num_threads(24)

#TRANSFORM
# when data are preprocessed
inference_transform = Compose(
            [
                LoadImageD(keys=("image")),
                EnsureChannelFirstD(keys="image"),
                OrientationD(keys=["image"], axcodes="RAS"),
                ScaleIntensityRangeD(
                    keys=["image"], a_min=0, a_max=5000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                EnsureTypeD(keys=["image"], data_type="tensor"),
                ToDeviceD(keys=["image"], device='cuda:0')
            ]
        )

#inference with preprocessing
keys = ["image", 'label']
inference_transform_pre = Compose(
            [
                LoadImageD(keys=keys, reader='NibabelReader'),
                EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                OrientationD(keys=keys, axcodes="RAS"),
                ToDeviceD(keys=keys, device=device),
                EnsureTypeD(keys=keys, data_type="tensor", device=device),
                SpacingD(keys=keys,
                         pixdim=(0.4, 0.4, 0.4),
                         mode=("bilinear", 'nearest'),
                         recompute_affine=True),
                ScaleIntensityRangeD(
                    keys=["image"], a_min=0, a_max=5000,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                ThresholdIntensityD(keys=["label"], above=False, threshold=32, cval=32), #clip to number of classes - clip value equall to max class value
                ThresholdIntensityD(keys=["label"], above=True, threshold=0, cval=0),
            ]
        )

#crop preprocessing
test_crop_transform = Compose(
            [
                LoadImageD(keys=keys, reader='NibabelReader'),
                EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                OrientationD(keys=keys, axcodes="RAS"),
                ToDeviceD(keys=keys, device=device),
                EnsureTypeD(keys=keys, data_type="tensor", device=device),
                SpacingD(keys=keys, pixdim=(0.4,0.4,0.4), mode=("bilinear", "nearest")),
                CropForegroundD(keys=["image", "label"],
                                source_key="image",
                                select_fn=lambda x: x > 0,
                                margin=(32, 32, 32),
                                k_divisible=32,
                                mode='constant',
                                constant_values=(-1000, 0)),
                ScaleIntensityRangeD(keys="image",
                                    a_min=0,
                                    a_max=5000,
                                    b_min=0.0,
                                    b_max=1.0,
                                    clip=True),
                ThresholdIntensityD(keys=["label"], above=False, threshold=32, cval=32),
                ThresholdIntensityD(keys=["label"], above=True, threshold=0, cval=0),
                ToDeviceD(keys=keys, device=device),
                EnsureTypeD(keys=keys, data_type="tensor", device=device)
            ]
        )   
                     
#PATHS
model = 'swin'
data_root_dir = 'data'
binary_output = f"multimedia/inference/{model}_33class"
if not os.path.exists(binary_output):
    os.mkdir(binary_output)


#TRANSFORMS
post_transform_binary = Compose([
    ActivationsD(keys="pred", sigmoid=True),
    AsDiscreteD(keys="pred", threshold=0.5),
    # SaveImageD(keys="pred", meta_keys="pred_meta_dict", data_root_dir=data_root_dir, separate_folder=False, output_dir=binary_output, output_postfix="roi.seg", resample=False, print_log=True),
])
post_transform_multiclass = Compose([
    ActivationsD(keys="pred", softmax=True),
    AsDiscreteD(keys="pred", argmax=True),
    # SaveImageD(keys="pred", meta_keys="pred_meta_dict", output_dir=f"multimedia/inference/{model}_multiclass", output_postfix="seg2", resample=False),
])

#METRICS
reduction = 'mean_batch'
dice = DiceMetric(include_background=False, reduction=reduction)
jacc = MeanIoU(include_background=False, reduction=reduction)
hausdorf = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=None, get_not_nans=False, directed=True, reduction=reduction)
hausdorf95 = HausdorffDistanceMetric(include_background=False, distance_metric='euclidean', percentile=95, get_not_nans=False, directed=True, reduction=reduction)
asd = SurfaceDistanceMetric(include_background=False, distance_metric='euclidean', reduction=reduction)

metrics_list = [dice, jacc, hausdorf, hausdorf95, asd]

ablation = 0
checkpoint_path = ''

#MODEL
patch_size = (96,96,128)
roi_transform = ResizeD(keys=['image'], spatial_size=patch_size, size_mode="all", mode="trilinear")
strides = list((5-1)*(2,))
feature_maps = tuple(2**i*32 for i in range(0, 5))
model = AttentionUnet(spatial_dims=3, in_channels=1, out_channels=1, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2))
model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'], strict=False)
model = model.to(device)
model.eval()

#DATA
datasets = ['testset']
datalist = []
for dataset in datasets:
    nifti_paths_scans = natsorted(glob.glob(os.path.join(data_root_dir, dataset, 'scans', '*.nii.gz'), recursive=True))
    nifti_paths_labels = natsorted(glob.glob(os.path.join(data_root_dir, dataset, 'labels', '*.nii.gz'), recursive=True))
    # nifti_paths_labels = natsorted(glob.glob(os.path.join(data_root_dir, dataset, '**', '*.nii.gz'), recursive=True))
    nifti_list = [{'image' : scan, 'label': label} for scan, label in zip(nifti_paths_scans, nifti_paths_labels)]
    datalist.extend(nifti_list)

test_dataset = Dataset(data=datalist, transform=inference_transform_pre)
test_loader = ThreadDataLoader(test_dataset, use_thread_workers=True, buffer_size=1, batch_size=1)

#SAVE 
def save_nifti(array, path, folder_suffix="ablation"):
    affine_pixdim_04 = np.eye(4) * 0.4
    affine_pixdim_04[3][3]=1.0
    nib_roi = nib.Nifti1Image(array.astype(np.int16), affine=affine_pixdim_04)
    new_dir = f'multimedia/inference/visual_{folder_suffix}/ablation_{ablation}'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    # new_path = os.path.join(new_dir, path.split('/')[-3], path.split('/')[-1])
    new_path = os.path.join(new_dir, path.split('/')[-1])
    nib.save(nib_roi, new_path)

#missing_teeth is 0
all_teeth = torch.arange(1,33).long()     
target_names = [f'tooth_{i}' for i in range(1,33)] 
all_metrics = []
calculate_metrics=True
save_raw = False
save_postprocessed = False

with torch.no_grad():
    for idx, test_data in enumerate(tqdm.tqdm(test_loader)):
        crops = []
        bounding_boxes = []
        bbox_shapes = []
        scan = test_data['image'].squeeze().cpu().numpy()
        label = test_data['label'].squeeze().cpu().numpy()
        classes_in_scan = np.unique(label)
        for lbl in classes_in_scan[1:]:
            coords = np.argwhere(label == lbl)
            min_slice = coords.min(axis=0)
            max_slice = coords.max(axis=0)  
            new_shape = max_slice-min_slice
            margins = (0.5, 0.5, 0.5) #loose
            # margins = (0.1,0.1,0.1) #tight
            crop_margins = tuple(int(r*x) for x, r in zip(new_shape, margins))
            min_slice -= crop_margins
            max_slice += crop_margins
            # edge conditions - limit to boundaries of original shape
            min_slice[min_slice < 0] = 0
            max_slice = tuple(map(lambda a, da: min(a,da), tuple(max_slice), label.shape))
            #cut roi
            tresh_slices = tuple(map(slice, min_slice, max_slice))
            roi = scan[tresh_slices].copy()
            bbox_shapes.append(roi.shape)
            roi_label = label[tresh_slices].copy()
            roi_label[roi_label!=lbl]=0
            crops.append({'image':roi, 'label':roi_label})
            bounding_boxes.append(tresh_slices)

        container = torch.zeros_like(test_data['label'])
        for data, bbox, bbox_shape in zip(crops, bounding_boxes, bbox_shapes):
            roi_input = torch.from_numpy(data['image'])
            roi_input = roi_input.to(device=device).unsqueeze(0)
            roi_input = roi_transform({'image':roi_input})['image'].unsqueeze(0)
            output = model(roi_input)
            output = post_transform_binary({'pred':output})
            output = output['pred']
            output = Resize(spatial_size=bbox_shape, size_mode="all", mode="nearest")(output[0])
            container[0][0][bbox][output[0]>0]=1

        model_output = container.squeeze().cpu().numpy()
        gt_image = test_data["label"].squeeze().cpu().numpy()
        output_postprocessed = post_processing_segmentation(model_output, instance_ground_truth=gt_image, device=device, distance_treshold=0.2, 
                                                            opening_kernel=3, instance_min_volume=900)
        
        if calculate_metrics:
            #B,C,H,W,D
            output_torch = torch.from_numpy(output_postprocessed).unsqueeze(0).unsqueeze(0)

            #classification metrics
            output_classes = torch.unique(output_torch)[1:]
            gt_classes = torch.unique(test_data["label"])[1:].long()
            #fill with zeros missing teeth
            output_classes_miss = np.array([tooth_id.item() if tooth_id in output_classes else 0 for tooth_id in all_teeth])
            gt_classes_miss = np.array([tooth_id.item() if tooth_id in gt_classes else 0 for tooth_id in all_teeth])
            #binarise to encode missing or present tooth
            output_classes_miss[output_classes_miss>1]=1
            gt_classes_miss[gt_classes_miss>1]=1
            f1_report = classification_report(y_true=gt_classes_miss, y_pred=output_classes_miss, output_dict=True)
            f1_result = f1_report['weighted avg']['f1-score']

            metrics_values = [f1_result]
            print(test_data["image_meta_dict"]['filename_or_obj'])
            #multiclass
            pred_one_hot = one_hot(output_torch, num_classes=33, dim=1).long()
            gt_one_hot = one_hot(test_data["label"].cpu(), num_classes=33, dim=1).long()
            #binary
            output_torch[output_torch>1]=1
            output_torch = output_torch.long()
            gt_binary = test_data["label"].cpu().long()
            gt_binary[gt_binary>1]=1

            for func in metrics_list:
                func(y_pred=pred_one_hot,
                    y=gt_one_hot)
                # print(f"metric: {func.__class__}")
                results = func.aggregate().cpu().numpy()
                # print(results)
                if any(~np.isfinite(results)):
                    mean_metric = results[np.isfinite(results)].mean()
                else:
                    mean_metric = results[results.nonzero()].mean()
                metrics_values.append(mean_metric)
                func.reset()
                # print(metrics_values[-1])
            metrics=np.array(metrics_values)*[1,1,1,0.4,0.4,0.4] #pidim 0.4 mm/px
            # print(metrics)
            all_metrics.append(metrics.tolist())
        if save_raw:
            save_nifti(model_output, test_data["image_meta_dict"]['filename_or_obj'], "ablation_raw")
        if save_postprocessed:
            save_nifti(output_postprocessed, test_data["image_meta_dict"]['filename_or_obj'], "ablation")

if calculate_metrics:
    _=[print (' '.join([f"{i:.3f}" for i in m])) for m in all_metrics]
    all_metrics_np = np.array(all_metrics)
    metrics_mean = all_metrics_np.mean(axis=0)
    metrics_std = all_metrics_np.std(axis=0)
    metrics_df = pd.DataFrame(all_metrics, columns=["F1", "Dice", "mIoU", "HD", "HD95", "ASSD"])
    metrics_df.loc[metrics_df.shape[0]]=metrics_mean
    metrics_df.loc[metrics_df.shape[0]]=metrics_std
    metrics_df.to_csv(f'csv_files/final_testset/inference_results_att_unet_roi_tight.csv', sep='\t', index=False)
    print(' '.join([f"{m:.4f}±{s:.4f}" for m,s in zip(metrics_mean, metrics_std)]))