import numpy as np
import torch

from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
    ToDevice
)

from monai.transforms import (
    Compose,
    ActivationsD,
    AddChannelD,
    AsDiscreteD,
    CropForegroundD,
    CenterScaleCropD,
    CenterSpatialCropD,
    CropForegroundD,
    EnsureChannelFirstD,
    EnsureTypeD,
    FgBgToIndicesD,
    LoadImageD,
    MeanEnsembleD,
    NormalizeIntensityD,
    OrientationD,
    ResizeD,
    ResizeWithPadOrCropD,
    RandAdjustContrastD,
    Rand3DElasticD,
    RandFlipD,
    RandGaussianNoiseD,
    RandGaussianSmoothD,
    RandCropByPosNegLabelD,
    RandCoarseShuffleD,
    RandRotateD,
    RandRotate90D,
    RandAffineD,
    RandSpatialCropD,
    RandScaleIntensityD,
    RandShiftIntensityD,
    RandZoomD,
    ScaleIntensityD,
    ScaleIntensityRangeD,
    SpacingD,
    SpatialCropD,
    ThresholdIntensityD,
    ToTensorD,
    ToDeviceD,
    VoteEnsembleD
)

# NEW CROP FOREGROUND

from monai.transforms import (
    BorderPad,
    Crop,
    Cropd,
    Pad
)
from itertools import chain
from monai.config import KeysCollection
from typing import Sequence, Union, Optional, Mapping, Hashable, Callable, Dict
from monai.transforms.utils import (
    compute_divisible_spatial_size,
    generate_spatial_bounding_box,
    is_positive,
)
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.utils import PytorchPadMode,TraceKeys, ensure_tuple, ensure_tuple_rep, convert_data_type
from monai.config import IndexSelection, SequenceStr

#Augmentation logic
class CropForegroundFixed(Crop):
    def __init__(
        self,
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        margin: Union[Sequence[int], int] = 0,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        allow_smaller: bool = True,
        return_coords: bool = False,
        k_divisible: Union[Sequence[int], int] = 1,
        mode: str = PytorchPadMode.CONSTANT,
        **pad_kwargs,
    ) -> None:

        self.select_fn = select_fn
        self.channel_indices = ensure_tuple(channel_indices) if channel_indices is not None else None
        self.margin = margin
        self.spatial_size = spatial_size
        self.allow_smaller = allow_smaller
        self.return_coords = return_coords
        self.k_divisible = k_divisible
        self.padder = Pad(mode=mode, **pad_kwargs)

    def compute_bounding_box(self, img: torch.Tensor):
        """
        Compute the start points and end points of bounding box to crop.
        And adjust bounding box coords to be divisible by `k`.

        """
        box_start, box_end = generate_spatial_bounding_box(
            img, self.select_fn, self.channel_indices, self.margin, self.allow_smaller
        )
        box_start_, *_ = convert_data_type(box_start, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        box_end_, *_ = convert_data_type(box_end, output_type=np.ndarray, dtype=np.int16, wrap_sequence=True)
        # make the spatial size divisible by `k`
        # spatial_size = np.asarray(compute_divisible_spatial_size(orig_spatial_size.tolist(), k=self.k_divisible))
        # update box_start and box_end
        center_ = np.floor_divide(box_start_ + box_end_, 2)
        box_start_ = center_ - np.floor_divide(np.asarray(self.spatial_size), 2)
        box_end_ = box_start_ + self.spatial_size

        return box_start_, box_end_

    def crop_pad(
        self, img: torch.Tensor, box_start: np.ndarray, box_end: np.ndarray, mode: Optional[str] = None, **pad_kwargs
    ):
        """
        Crop and pad based on the bounding box.

        """        
        slices = self.compute_slices(roi_start=box_start, roi_end=box_end)
        cropped = super().__call__(img=img, slices=slices)
        pad_to_start = np.maximum(-box_start, 0)
        pad_to_end = np.maximum(box_end - np.asarray(img.shape[1:]), 0)
        pad = list(chain(*zip(pad_to_start.tolist(), pad_to_end.tolist())))
        pad_width = BorderPad(spatial_border=pad).compute_pad_width(cropped.shape[1:])
        ret = self.padder.__call__(img=cropped, to_pad=pad_width, mode=mode, **pad_kwargs)
        # combine the traced cropping and padding into one transformation
        # by taking the padded info and placing it in a key inside the crop info.
        if get_track_meta():
            ret_: MetaTensor = ret  # type: ignore
            app_op = ret_.applied_operations.pop(-1)
            ret_.applied_operations[-1][TraceKeys.EXTRA_INFO]["pad_info"] = app_op
        return ret

    def __call__(self, img: torch.Tensor, mode: Optional[str] = None, **pad_kwargs):  # type: ignore
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't change the channel dim.
        """
        box_start, box_end = self.compute_bounding_box(img)
        cropped = self.crop_pad(img, box_start, box_end, mode, **pad_kwargs)

        if self.return_coords:
            return cropped, box_start, box_end
        return cropped

    def inverse(self, img: MetaTensor) -> MetaTensor:
        transform = self.get_most_recent_transform(img)
        # we moved the padding info in the forward, so put it back for the inverse
        pad_info = transform[TraceKeys.EXTRA_INFO].pop("pad_info")
        img.applied_operations.append(pad_info)
        # first inverse the padder
        inv = self.padder.inverse(img)
        # and then inverse the cropper (self)
        return super().inverse(inv)

#Augmentation dictionary wrapper
class CropForegroundFixedd(Cropd):

    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        margin: Union[Sequence[int], int] = 0,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        allow_smaller: bool = True,
        k_divisible: Union[Sequence[int], int] = 1,
        mode: SequenceStr = PytorchPadMode.CONSTANT,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        allow_missing_keys: bool = False,
        **pad_kwargs,
    ) -> None:

        self.source_key = source_key
        self.start_coord_key = start_coord_key
        self.end_coord_key = end_coord_key
        cropper = CropForegroundFixed(
            select_fn=select_fn,
            channel_indices=channel_indices,
            margin=margin,
            spatial_size=spatial_size,
            allow_smaller=allow_smaller,
            k_divisible=k_divisible,
            **pad_kwargs,
        )
        super().__init__(keys, cropper=cropper, allow_missing_keys=allow_missing_keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        self.cropper: CropForegroundFixed
        box_start, box_end = self.cropper.compute_bounding_box(img=d[self.source_key])
        if self.start_coord_key is not None:
            d[self.start_coord_key] = box_start
        if self.end_coord_key is not None:
            d[self.end_coord_key] = box_end
        for key, m in self.key_iterator(d, self.mode):
            d[key] = self.cropper.crop_pad(img=d[key], box_start=box_start, box_end=box_end, mode=m)
        return d

CropForegroundFixedD = CropForegroundFixedDict = CropForegroundFixedd

class Transforms():
    def __init__(self,
                 args,
                 device : str = 'cpu' 
                ) -> None:

        self.is_h5 = 'h5' in args.data
        self.pixdim = (args.pixdim,)*3
        self.class_treshold = args.classes if args.classes == 1 else args.classes-1
        keys = args.keys

        if args.patch_mode == "global":
            self.train_transform = Compose(
                [
                    #INITAL SETUP
                    LoadImageD(keys=keys, reader='NibabelReader'),
                    EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                    OrientationD(keys=keys, axcodes="RAS"),
                    ToDeviceD(keys=keys, device=device),
                    EnsureTypeD(keys=keys, data_type="tensor", device=device),
                    #GEOMETRIC - NON-RANDOM - PREPROCESING
                    SpacingD(keys=keys, pixdim=self.pixdim, mode=("bilinear", "nearest")),
                    CropForegroundFixedD(keys=keys,
                                        source_key="label",
                                        select_fn=lambda x: x > 0,
                                        margin=args.spatial_crop_margin,
                                        spatial_size=args.spatial_crop_size,
                                        mode='constant',
                                        return_coords=True,
                                        constant_values=(-1000, 0)),
                    #NON-RANDOM - perform on GPU

                    #INTENSITY - NON-RANDOM - PREPROCESING
                    ##image
                    ScaleIntensityRangeD(keys="image",
                                        a_min=0,
                                        a_max=args.houndsfield_clip,
                                        b_min=0.0,
                                        b_max=1.0,
                                        clip=True),
                    ##label
                    ThresholdIntensityD(keys=["label"], above=False, threshold=self.class_treshold, cval=self.class_treshold), #clip to number of classes - clip value equall to max class value
                    ThresholdIntensityD(keys=["label"], above=True, threshold=0, cval=0), #clip from below to 0 - all values smaler than 0 are replaces with zeros
                    #######################################
                    #GEOMETRIC - RANDOM - DATA AUGMENTATION 
                    ToDeviceD(keys=keys, device=device),
                    RandSpatialCropD(keys=keys,
                                        roi_size=args.patch_size,
                                        random_center=True,
                                        random_size=False),
                    # Do not use flips - because mirror reflection changes classes - tooth eg. 44 and 34 is identical
                    # RandFlipD(keys=["image", "label"], prob=0.25, spatial_axis=0), 
                    # RandFlipD(keys=["image", "label"], prob=0.25, spatial_axis=1), # axis=2 is z axis and patient is always upright so no need for this augmentation
                    # RandZoomD(keys=["image", "label"], prob=0.25, min_zoom=0.8, max_zoom=1.2, mode=("trilinear", "nearest"), padding_mode='constant', constant_values=(0, 0), keep_size=True),
                    # RandRotate90D(keys, max_k=1, spatial_axes=(0,1), prob=0.1),
                    # RandRotate90D(keys, max_k=1, spatial_axes=(1,0), prob=0.1),
                    #INTENSITY - RANDOM - DATA AUGMENTATION
                    RandAdjustContrastD(keys="image",
                                        gamma=(0.5, 2.0),
                                        prob=0.25),
                    RandShiftIntensityD(keys="image", offsets=0.20, prob=0.5),
                    RandScaleIntensityD(keys="image", factors=0.15, prob=0.5),
                    # RandCoarseShuffleD(keys="image", holes=8, max_holes=16, spatial_size=(16,16,16), max_spatial_size=(32,32,32), prob = 1.0),
                    # RandCoarseShuffleD(keys="image", holes=8, max_holes=20, spatial_size=(22,22,22), max_spatial_size=(44,44,44), prob = 0.5), #it gives dropout from 0.5% to 10% of the patch size volume
                    #FINAL CHECK
                    EnsureTypeD(keys=keys, data_type="tensor", device=device)
                ]
            )   

            self.val_transform = Compose(
                [
                    #INITAL SETUP
                    LoadImageD(keys=keys, reader='NibabelReader'),
                    EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                    OrientationD(keys=keys, axcodes="RAS"),
                    ToDeviceD(keys=keys, device=device),
                    #GEOMETRIC - NON-RANDOM - PREPROCESING
                    SpacingD(keys=keys, pixdim=self.pixdim, mode=("bilinear", "nearest")),
                    # SpatialPadD(keys=["image", "label"],
                    #             spatial_size = args.padding_size, # pad z axis if smaller than X
                    #             method='symmetric',
                    #             mode='constant',
                    #             constant_values=(-1000, 0)),
                    CropForegroundFixedD(keys=keys,
                                        source_key="label",
                                        select_fn=lambda x: x > 0,
                                        margin=args.spatial_crop_margin,
                                        spatial_size=args.spatial_crop_size,
                                        mode='constant',
                                        return_coords=True,
                                        constant_values=(-1000, 0)),
                    # CenterSpatialCropD(keys=["image", "label"],
                    #                    roi_size=(384,384,256)), # perform it in case some scans where bigger than padding, crop based on padding size
                    #NON-RANDOM - perform on GPU
                    EnsureTypeD(keys=keys, data_type="tensor"),
                    ToDeviceD(keys=keys, device=device),
                    #INTENSITY - NON-RANDOM - PREPROCESING
                    ##image
                    ScaleIntensityRangeD(keys="image",
                                        a_min=0,
                                        a_max=args.houndsfield_clip,
                                        b_min=0.0,
                                        b_max=1.0,
                                        clip=True),
                    ##label
                    ThresholdIntensityD(keys=["label"], above=False, threshold=self.class_treshold, cval=self.class_treshold), #clip to number of classes - clip value equall to max class value
                    ThresholdIntensityD(keys=["label"], above=True, threshold=0, cval=0), #clip from below to 0 - all values smaler than 0 are replaces with zeros
                    # EnsureTypeD(keys=keys, data_type="tensor"),
                    # ToDeviceD(keys=keys, device=device)
                ]
            )   
        elif args.patch_mode == "local":
            keys = ['image', 'label']
            self.train_transform = Compose(
                        [
                        #INITAL SETUP
                        LoadImageD(keys=keys, reader='NibabelReader'),
                        EnsureChannelFirstD(keys=keys),
                        OrientationD(keys=keys, axcodes="RAS"),
                        ToDeviceD(keys=keys, device=device),
                        EnsureTypeD(keys=keys, data_type="tensor", device=device),
                        #GEOMETRIC - NON-RANDOM - PREPROCESING
                        SpacingD(keys=keys, pixdim=self.pixdim, mode=("bilinear", "nearest")),
                        # ResizeWithPadOrCropD(keys=keys, spatial_size=(64,64,96), method="symmetric", mode = "constant", constant_values=(-1000, 0)),
                        ResizeD(keys=keys, spatial_size=args.patch_size, size_mode="all", mode=("trilinear", "nearest")),
                        #NON-RANDOM - perform on GPU
                        ToDeviceD(keys=keys, device=device),# run if persistent datat were generated on different gpu
                        #INTENSITY - NON-RANDOM - PREPROCESING
                        ##image
                        ScaleIntensityRangeD(keys="image",
                                            a_min=0,
                                            a_max=args.houndsfield_clip,
                                            b_min=0.0,
                                            b_max=1.0,
                                            clip=True),
                        ##label
                        ThresholdIntensityD(keys=["label"], above=False, threshold=1, cval=1), #clip to number of classes - clip value equall to max class value
                        ThresholdIntensityD(keys=["label"], above=True, threshold=0, cval=0), #clip from below to 0 - all values smaler than 0 are replaces with zeros
                        #######################################
                        #GEOMETRIC - RANDOM - DATA AUGMENTATION 
                        ToDeviceD(keys=keys, device=device),
                        RandZoomD(keys=keys, min_zoom=1.5, max_zoom=1.5, mode=("trilinear", "nearest"), prob=0.5),
                        # RandSpatialCropD(keys=keys,
                        #                     roi_size=args.patch_size,
                        #                     random_center=True,
                        #                     random_size=False),
                        # Do not use flips - because mirror reflection changes classes - tooth eg. 44 and 34 is identical
                        # RandFlipD(keys=["image", "label"], prob=0.25, spatial_axis=0), 
                        # RandFlipD(keys=["image", "label"], prob=0.25, spatial_axis=1), # axis=2 is z axis and patient is always upright so no need for this augmentation
                        # RandZoomD(keys=["image", "label"], prob=0.25, min_zoom=0.8, max_zoom=1.2, mode=("trilinear", "nearest"), padding_mode='constant', constant_values=(0, 0), keep_size=True),
                        # RandRotate90D(keys, max_k=1, spatial_axes=(0,1), prob=0.1),
                        # RandRotate90D(keys, max_k=1, spatial_axes=(1,0), prob=0.1),
                        #INTENSITY - RANDOM - DATA AUGMENTATION
                        RandAdjustContrastD(keys="image",
                                            gamma=(0.5, 2.0),
                                            prob=0.5),
                        RandShiftIntensityD(keys="image", offsets=0.20, prob=0.5),
                        RandScaleIntensityD(keys="image", factors=0.15, prob=0.5),
                        #FINAL CHECK
                        EnsureTypeD(keys=keys, data_type="tensor", device=device)
                        ]
                    )   
            self.val_transform = Compose(
                        [
                        #INITAL SETUP
                        LoadImageD(keys=keys, reader='NibabelReader'),
                        EnsureChannelFirstD(keys=keys),
                        OrientationD(keys=keys, axcodes="RAS"),
                        ToDeviceD(keys=keys, device=device),
                        EnsureTypeD(keys=keys, data_type="tensor", device=device),
                        #GEOMETRIC - NON-RANDOM - PREPROCESING
                        SpacingD(keys=keys, pixdim=self.pixdim, mode=("bilinear", "nearest")),
                        # ResizeWithPadOrCropD(keys=keys, spatial_size=(64,64,96), method="symmetric", mode = "constant", constant_values=(-1000, 0)),
                        ResizeD(keys=keys, spatial_size=args.patch_size, size_mode="all", mode=("trilinear", "nearest")),
                        #NON-RANDOM - perform on GPU
                        ToDeviceD(keys=keys, device=device),# run if persistent datat were generated on different gpu
                        #INTENSITY - NON-RANDOM - PREPROCESING
                        ##image
                        ScaleIntensityRangeD(keys="image",
                                            a_min=0,
                                            a_max=args.houndsfield_clip,
                                            b_min=0.0,
                                            b_max=1.0,
                                            clip=True),
                        ##label
                        ThresholdIntensityD(keys=["label"], above=False, threshold=1, cval=1), #clip to number of classes - clip value equall to max class value
                        ThresholdIntensityD(keys=["label"], above=True, threshold=0, cval=0), #clip from below to 0 - all values smaler than 0 are replaces with zeros
                        #######################################
                        #GEOMETRIC - RANDOM - DATA AUGMENTATION 
                        ToDeviceD(keys=keys, device=device),
                        EnsureTypeD(keys=keys, data_type="tensor", device=device)
                        ]
                    )   

        ##############################################################################################################################################################
        if args.multitask:
            self.train_rec_transform = Compose(
                [
                    #INITAL SETUP
                    LoadImageD(keys=["image", "label"], reader='NibabelReader'),
                    EnsureChannelFirstD(keys=["image", "label"], channel_dim='no_channel'),
                    OrientationD(keys=["image", "label"], axcodes="RAS"),
                    ToDeviceD(keys=["image", "label"], device=device),
                    #GEOMETRIC - NON-RANDOM - PREPROCESING
                    SpacingD(keys=["image", "label"], pixdim=self.pixdim, mode=("bilinear", "nearest")),
                    # CropForegroundFixedD(keys=["image", "label"],
                    #                     source_key="label",
                    #                     select_fn=lambda x: x > 0,
                    #                     margin=args.spatial_crop_margin,
                    #                     spatial_size=args.spatial_crop_size,
                    #                     mode='constant',
                    #                     return_coords=True,
                    #                     constant_values=(-1000, 0)),
                    CropForegroundD(keys=["image", "label"],
                                    source_key="image",
                                    select_fn=lambda x: x > 0,
                                    margin=(32, 32, 32),
                                    k_divisible=32,
                                    mode='constant',
                                    constant_values=(-1000, 0)),
                    #NON-RANDOM - perform on GPU
                    EnsureTypeD(keys=["image", "label"], data_type="tensor"),
                    ToDeviceD(keys=["image", "label"], device=device),
                    #INTENSITY - NON-RANDOM - PREPROCESING
                    ThresholdIntensityD(keys=["image", "label"], above=False, threshold=self.class_treshold, cval=self.class_treshold), #clip to number of classes - clip value equall to max class value
                    ThresholdIntensityD(keys=["image", "label"], above=True, threshold=0, cval=0), #clip from below to 0 - all values smaler than 0 are replaces with zeros
                    #######################################
                    #GEOMETRIC - RANDOM - DATA AUGMENTATION 
                    ToDeviceD(keys=["image", "label"], device=device),
                    RandSpatialCropD(keys=["image", "label"],
                                        roi_size=args.patch_size,
                                        random_center=True,
                                        random_size=False),
                    # Do not use flips - because mirror reflection changes classes - tooth eg. 44 and 34 is identical
                    # RandFlipD(keys=["image", "label"], prob=0.25, spatial_axis=0), 
                    # RandFlipD(keys=["image", "label"], prob=0.25, spatial_axis=1), # axis=2 is z axis and patient is always upright so no need for this augmentation
                    # RandZoomD(keys=["image", "label"], prob=0.25, min_zoom=0.8, max_zoom=1.2, mode=("trilinear", "nearest"), padding_mode='constant', constant_values=(0, 0), keep_size=True),
                    #INTENSITY - RANDOM - DATA AUGMENTATION
                    RandCoarseShuffleD(keys="image", holes=8, max_holes=16, spatial_size=(16,16,16), max_spatial_size=(32,32,32), prob = 1.0), #it gives dropout from 0.5% to 10% of the patch size volume
                    #FINAL CHECK
                    EnsureTypeD(keys=["image", "label"], data_type="tensor", device=device)
                ]
            )   

            self.val_rec_transform = Compose(
                [
                    #INITAL SETUP
                    LoadImageD(keys=["image", "label"], reader='NibabelReader'),
                    EnsureChannelFirstD(keys=["image", "label"], channel_dim='no_channel'),
                    OrientationD(keys=["image", "label"], axcodes="RAS"),
                    #GEOMETRIC - NON-RANDOM - PREPROCESING
                    SpacingD(keys=["image", "label"], pixdim=self.pixdim, mode=("bilinear", "nearest"),),
                    # SpatialPadD(keys=["image", "label"],
                    #             spatial_size = args.padding_size, # pad z axis if smaller than X
                    #             method='symmetric',
                    #             mode='constant',
                    #             constant_values=(-1000, 0)),
                    CropForegroundFixedD(keys=["image", "label"],
                                        source_key="label",
                                        select_fn=lambda x: x > 0,
                                        margin=args.spatial_crop_margin,
                                        spatial_size=args.spatial_crop_size,
                                        mode='constant',
                                        return_coords=True,
                                        constant_values=(-1000, 0)),
                    # CenterSpatialCropD(keys=["image", "label"],
                    #                    roi_size=(384,384,256)), # perform it in case some scans where bigger than padding, crop based on padding size
                    #NON-RANDOM - perform on GPU
                    EnsureTypeD(keys=["image", "label"], data_type="tensor"),
                    ##label
                    ThresholdIntensityD(keys=["image", "label"], above=False, threshold=args.classes, cval=args.classes-1), #clip to number of classes - clip value equall to max class value
                    ThresholdIntensityD(keys=["image", "label"], above=True, threshold=0, cval=0), #clip from below to 0 - all values smaler than 0 are replaces with zeros
                    ToDeviceD(keys=["image", "label"], device=device)
                ]
            )   

        ##############################################################################################################################################################

        self.binarize_transform = ThresholdIntensityD(keys="label", above=False, threshold=1, cval=1)

        if args.classes > 1:
            self.post_pred = Compose([Activations(softmax=True, dim=0),
                                      AsDiscrete(argmax=True,
                                                 dim=0,
                                                 keepdim=True),
                                      ToDevice(device=device)
                                    ])
            self.post_pred_labels = Compose([AsDiscrete(argmax=False,
                                                        to_onehot=args.classes,
                                                        dim=0),
                                             ToDevice(device=device)
                                            ])
        elif args.classes == 1:
            self.post_pred = Compose([Activations(sigmoid=True),
                                      AsDiscrete(threshold=0.5)],
                                      ToDevice(device=device))

def dilation2d(image : torch.tensor, kernel : torch.tensor, border_type: str = 'constant', border_value: int = 0):

    _, _, se_h, se_w = kernel.shape
    origin = [se_h // 2, se_w // 2]
    pad_margin = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]

    volume_pad = torch.nn.functional.pad(image, pad_margin, mode=border_type, value=border_value)
    out = torch.nn.functional.conv2d(volume_pad, kernel, padding=0).to(torch.int)
    dilation_out = torch.clamp(out,0,1)
    return dilation_out


def dilation3d(volume : torch.tensor, kernel : torch.tensor = torch.ones((1,1,3,3,3)), border_type: str = 'constant', border_value: int = 0):

    if len(kernel.shape) == 5:
        _, _, se_h, se_w, se_d = kernel.shape
    elif len(kernel.shape) == 4:
        _, se_h, se_w, se_d = kernel.shape

    origin = [se_h // 2, se_w // 2, se_d // 2]
    pad_margin = [origin[2], se_d - origin[2] - 1, origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]

    volume_pad = torch.nn.functional.pad(volume, pad_margin, mode=border_type, value=border_value)
    out = torch.nn.functional.conv3d(volume_pad, kernel, padding=0)
    dilation_out = torch.clamp(out,0,1)
    return dilation_out


def erosion2d(image : torch.tensor, kernel : torch.tensor, border_type: str = 'constant', border_value: int = 0):

    _, _, se_h, se_w = kernel.shape
    origin = [se_h // 2, se_w // 2]
    pad_margin = [origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]
   
    volume_pad = torch.nn.functional.pad(image, pad_margin, mode=border_type, value=border_value)
    if torch.is_tensor(kernel):
        bias=-kernel.sum().unsqueeze(0)
    else:
        bias=torch.tensor(-kernel.sum()).unsqueeze(0)
    out = torch.nn.functional.conv2d(volume_pad, kernel, padding=0, bias=bias).to(torch.int)
    erosion_out = torch.add(torch.clamp(out,-1,0),1)
    return erosion_out


def erosion3d(volume : torch.tensor, kernel : torch.tensor = torch.ones((1,1,3,3,3)), border_type: str = 'constant', border_value: int = 0):

    if len(kernel.shape) == 5:
        _, _, se_h, se_w, se_d = kernel.shape
    elif len(kernel.shape) == 4:
        _, se_h, se_w, se_d = kernel.shape

    origin = [se_h // 2, se_w // 2, se_d // 2]
    pad_margin = [origin[2], se_d - origin[2] - 1, origin[1], se_w - origin[1] - 1, origin[0], se_h - origin[0] - 1]

    volume_pad = torch.nn.functional.pad(volume, pad_margin, mode=border_type, value=border_value)
    if torch.is_tensor(kernel):
        bias=-kernel.sum().unsqueeze(0)
    else:
        bias=torch.tensor(-kernel.sum()).unsqueeze(0)
    out = torch.nn.functional.conv3d(volume_pad, kernel, padding=0, stride=1, bias=bias)
    erosion_out = torch.add(torch.clamp(out,-1,0),1)
    return erosion_out
