"""Train script for CBCT segmentation using dentition-based learning."""
import time
import glob
import os
import warnings
from utils.parser import args
from natsort import natsorted

task = 'binary' if args.classes == 1 else 'multiclass'
args.cache_dir = os.path.join(args.cache_dir, f"{task}_{args.classes}_{args.patch_size[0]}_{args.spatial_crop_size[0]}_{args.spatial_crop_size[1]}_{args.spatial_crop_size[2]}_ablation_local_roi")

# Persistent Dataset cache for time and energy save
if args.clear_cache:
    print("Clearning cache...")
    train_cache = glob.glob(os.path.join(args.cache_dir, 'train/*.pt'))
    val_cache = glob.glob(os.path.join(args.cache_dir, 'val/*.pt'))
    if len(train_cache) != 0:
        for file in train_cache:
            os.remove(file)
    if len(val_cache) != 0:
        for file in val_cache:
            os.remove(file)
    print(f"Cleared cache in dir: {args.cache_dir}, train: {len(train_cache)} files, val: {len(val_cache)} files.")

if args.comet:
    from comet_ml import Experiment
    # replace API_KEY with our own key
    experiment = Experiment("API_KEY", project_name="CBCT_seg")
    tags = args.tags.split('#')
    tags += [args.model_name]
    experiment.add_tags(tags)
    experiment.log_asset('ToothSwinUNETR/utils/data_augmentation.py')
    experiment.log_asset('ToothSwinUNETR/utils/parser.py')
else:
    from utils.dummy_logger import DummyExperiment
    experiment = DummyExperiment()

# TORCH modules
import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from torch.optim.lr_scheduler import CosineAnnealingLR
from optimizers.scheduler import CosineAnnealingWarmupRestarts
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.nn import MSELoss, BCEWithLogitsLoss

#MONAI modules
from monai.networks.nets import UNet, VNet, AttentionUnet, UNETR
from models.swin_unetr import SwinUNETR
from monai.networks.utils import one_hot
from monai.losses import DiceLoss, DiceFocalLoss
from monai.metrics import MeanIoU, DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.optimizers import WarmupCosineSchedule
from monai.config import print_config
from monai.utils import set_determinism
from monai.data import set_track_meta, ThreadDataLoader, decollate_batch
from monai.data.dataset import PersistentDataset
from monai.inferers import sliding_window_inference
if args.print_config:
    print_config() # print monai config for the used environment    

from sklearn.model_selection import KFold

#external modules
import sys
sys.path.insert(1, os.getcwd())
from utilities.cuda_stats import setup_cuda
from utilities.log_image import Logger

from losses.loss import GWDLCELoss, get_tooth_dist_matrix, DiceCELoss
from utils.data_augmentation import Transforms
from utilities.log_image import Logger

#config
if args.seed != -1:
    set_determinism(seed=args.seed)
    warnings.warn(
        "You have chosen to seed training. "
        "This will turn on the CUDNN deterministic setting, "
        "which can slow down your training considerably! "
        "You may see unexpected behavior when restarting "
        "from checkpoints."
    )
experiment.log_parameters(vars(args))

# use amp to accelerate training
scaler = None
if args.use_scaler:
    scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

#LOGGER
log = Logger(args.classes, args.is_log_3d)

#CUDA
setup_cuda(args.gpu_frac, num_threads=args.num_threads, device=args.device, visible_devices=args.visible_devices, use_cuda_with_id=args.cuda_device_id)
if args.device == 'cuda':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=int(args.cuda_device_id))
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(int(args.cuda_device_id))

#TRANSFORMS
trans = Transforms(args, device)
set_track_meta(True)

#DATA
#global approach
if args.patch_mode == "global":
    data_root_dir = args.data
    datasets = ['scans']
    labels = ['labels_american']
    datalist =[]
    for dataset, labels in zip(datasets, labels):
        nifti_paths_scans = natsorted(glob.glob(os.path.join(data_root_dir, dataset, '**', '*.nii.gz'), recursive=True))
        nifti_paths_labels = natsorted(glob.glob(os.path.join(data_root_dir, labels, '**', '*.nii.gz'), recursive=True))
        nifti_list = [{'image': scan, 'label': label} for (scan, label) in zip(nifti_paths_scans, nifti_paths_labels)]
        datalist.extend(nifti_list)
    datalist = datalist[0:97]
#local approach
elif args.patch_mode == "local":
    data_root_dir = args.data
    datalist =[]
    scan_paths_roi = natsorted(glob.glob('data/china/roi_data_loose/scans/**/*.nii.gz'))
    label_paths_roi = natsorted(glob.glob('data/china/roi_data_loose/labels/**/*.nii.gz'))
    for scan, label in zip(scan_paths_roi, label_paths_roi):
        datalist.append({'image':scan,'label':label})

if not os.path.exists(args.cache_dir):
    os.makedirs(os.path.join(args.cache_dir, 'train'))
    os.makedirs(os.path.join(args.cache_dir, 'val'))

#DATASET
train_dataset = PersistentDataset(datalist, trans.train_transform, cache_dir=os.path.join(args.cache_dir, 'train'))
val_dataset = PersistentDataset(datalist, trans.val_transform, cache_dir=os.path.join(args.cache_dir, 'val'))
kfold = KFold(n_splits=args.split, shuffle=False)

#LOSS FUNCTION
# cross-entropy weights based on chinese public dataset for classes: 0-32 (background 0 and 32 classes {1,2,3,...,31,32})
# inverse frequency: n_samples / (n_classes * np.bincount(y))
if args.weighted_ce:
    weights = torch.from_numpy(numpy.load('ToothSwinUNETR/losses/ce_weights.npy')).to(dtype=torch.float32, device=device)
    weights[0]=args.background_weight
    assert(len(weights) == args.classes)

ls_weights = args.loss_weights
if args.loss_name == "DiceLoss" and args.classes == 1:
    criterion_seg = DiceLoss(sigmoid=True)
elif args.loss_name == "DiceCELoss":
    if args.weighted_ce:
        criterion_seg = DiceCELoss(include_background=args.include_background, to_onehot_y=True, softmax=True, lambda_ce=1.0, lambda_dice=1.0, ce_weight=weights)
    else:
        criterion_seg = DiceCELoss(include_background=args.include_background, to_onehot_y=True, softmax=True, lambda_ce=1.0, lambda_dice=1.0)
elif args.loss_name == "WasserDiceLoss":
    dist_matrix = get_tooth_dist_matrix(device, quarter_penalty=args.inter_quarter_penalty)
    criterion_seg = GWDLCELoss(dist_matrix, weights, lambda_dice=1.0, lambda_ce=1.0)
else:
    criterion_seg = DiceCELoss(include_background=args.include_background, to_onehot_y=True, softmax=True, lambda_ce=1.0, lambda_dice=1.0, ce_weight=weights)

### TRAINING STEP ###
def training_step(batch_idx, train_data, args):
    #scaler used only for debug, models for all final results are trained in FP32 precision
    if args.use_scaler and scaler is not None:
        with torch.cuda.amp.autocast():
            output = model(train_data["image"])
            dice_loss, ce_loss = criterion_seg(output, train_data["label"].long())
            loss = ls_weights['dice_w'] * dice_loss + ls_weights['ce_w'] * ce_loss
            loss = loss / accum_iter 
        scaler.scale(loss).backward()
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    else:
        output_dict = model(train_data["image"])
       
        if args.classes > 1:
            dice_loss, ce_loss = criterion_seg(output_dict, train_data["label"].long())
            loss = ls_weights['dice_w'] * dice_loss + ls_weights['ce_w'] * ce_loss
        else:
            #binary global approach - only dice loss
            dice_loss = criterion_seg(output_dict, train_data["label"].long())
            loss = ls_weights['dice_w'] * dice_loss

        loss = loss / accum_iter
        loss.backward()
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()

    if args.classes == 1:
        pred = trans.post_pred(output_dict).long()
        dice_metric(y_pred=pred, y=train_data["label"])
        jaccard_metric(y_pred=pred, y=train_data["label"])
    else:
        pred = torch.softmax(output_dict, dim=1).argmax(dim=1, keepdim=True).long()
        for func in [dice_metric, jaccard_metric]:
            func(y_pred=one_hot(pred, num_classes=args.classes, dim=1),
                y=one_hot(train_data["label"], num_classes=args.classes, dim=1))
    
    dice = dice_metric.aggregate()
    jac = jaccard_metric.aggregate()
    
    if args.classes > 1:
        dice = dice[dice.nonzero()]
        jac = jac[jac.nonzero()]
        
    epoch_time=time.time() - start_time_epoch

    if (batch_idx+1) % args.log_batch_interval == 0:
        print(" ", end="")
        print(f"Batch: {batch_idx + 1}/{len(train_loader)}"
            f" Loss: {loss.item():.4f} - ce:{ce_loss.item():.4f}, dice(gwdl): {dice_loss.item():.4f}."
            f" *** Jaccard: {jac.mean().item():.4f}"
            f" Dice Index: {dice.mean().item():.4f}"
            f" Time: {epoch_time:.2f}s")
    
    #log visual results to comet.ml
    if (args.is_log_image or args.is_log_3d) and batch_idx == 9:
        if (epoch+1) % args.log_slice_interval == 0 or (epoch+1) % args.log_3d_scene_interval_training == 0:
            pred_np = pred[0].squeeze().detach().cpu().numpy()
            label_np = train_data["label"][0].long().squeeze().detach().cpu().numpy()
            if (epoch+1) % args.log_slice_interval == 0:
                image = train_data["image"][0].squeeze().detach().cpu().numpy()
                image_log_out = log.log_image(pred_np, label_np, image)
                experiment.log_image(image_log_out, name=f'img_{(epoch+1):04}_{batch_idx+1:02}')
            if (epoch+1) % args.log_3d_scene_interval_training == 0:
                scene_log_out = log.log_3dscene_comp(pred_np, label_np, args.classes, scene_size=1024)
                experiment.log_image(scene_log_out, name=f'scene_{(epoch+1):04}_{batch_idx+1:02}')

    return jac.mean().item(), dice.mean().item(), loss.item()


### VALIDATION STEP ###
def validation_step(batch_idx, val_data, args):


    with torch.cuda.amp.autocast(args.use_scaler):
        val_output = sliding_window_inference(val_data["image"], roi_size=args.patch_size, sw_batch_size=8, predictor=model, overlap=0.6, sw_device=device,
                                            device=device, mode='gaussian', sigma_scale=0.125, padding_mode='constant', cval=0, progress=True)

    if args.classes == 1:
        val_preds = [trans.post_pred(i).long() for i in decollate_batch(val_output)]
        val_labels = [i for i in decollate_batch(val_data["label"])]
        dice_metric(y_pred=val_preds, y=val_labels)
        jaccard_metric(y_pred=val_preds, y=val_labels)
    else:
        val_preds_argmax = [trans.post_pred(i).long() for i in decollate_batch(val_output)]
        val_labels = [trans.post_pred_labels(i) for i in decollate_batch(val_data["label"])]
        for func in [dice_metric, jaccard_metric]:
            func(y_pred=[one_hot(i, args.classes, dim=0)for i in val_preds_argmax],
                 y=val_labels)
        
    jac = jaccard_metric.aggregate()
    dice = dice_metric.aggregate()
    
    if args.is_log_3d and epoch % args.log_3d_scene_interval_validation == 0 and batch_idx==0:
        if args.classes >1:
            pred=val_preds_argmax[0].squeeze().detach().cpu().numpy()
        else:
            pred=val_preds[0].squeeze().detach().cpu().numpy()
        label=val_data["label"][0].squeeze().detach().cpu().numpy()
        scene_log = log.log_3dscene_comp(pred,label, args.classes, scene_size=1024)
        experiment.log_image(scene_log, name=f'val_scene_{(epoch+1):04}_{batch_idx:02}')  

    return jac.mean().item(), dice.mean().item()


### CROSS VALIDATION LOOP ###
print("--------------------")
for fold, (train_ids, test_ids) in enumerate(kfold.split(train_dataset)):
    print(f"FOLD {fold}")
    print("-------------------")
    if fold == 1:
        break
    train_subsampler = SubsetRandomSampler(train_ids)
    test_subsampler = SubsetRandomSampler(test_ids)

    train_loader = ThreadDataLoader(train_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size, sampler=train_subsampler)
    test_loader = ThreadDataLoader(val_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val, sampler=test_subsampler)
    
    #UNET params
    feature_maps = tuple(2**i*args.n_features for i in range(0, args.unet_depth))
    strides = list((args.unet_depth-1)*(2,))
    
    #MODEL INIT
    # SOTA architectures
    if args.model_name == "SwinUNETR":
        model = SwinUNETR(spatial_dims=3, in_channels=1, out_channels=args.classes, img_size=args.patch_size,
                          feature_size=args.feature_size, use_checkpoint=args.activation_checkpoints)
    elif args.model_name == "UNet":
        model = UNet(spatial_dims=3, in_channels=1, out_channels=args.classes, channels=feature_maps, strides=strides, norm="instance")
    elif args.model_name == "AttUnet":
        model = AttentionUnet(spatial_dims=3, in_channels=1, out_channels=args.classes, channels=feature_maps, strides=strides)
    elif args.model_name == "VNet":
        model = VNet(spatial_dims=3, in_channels=1, out_channels=args.classes, dropout_prob=0, bias=False)
    elif args.model_name == "UNETR":
        model = UNETR(spatial_dims=3, in_channels=1, out_channels=args.classes, img_size=args.patch_size,  feature_size=args.feature_size, norm_name="instance")
    else:
        raise NotImplementedError(f"There are no implementation of: {args.model_name}")

    if args.parallel:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    # Optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.adam_ams, eps=args.adam_eps)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps)
    else:
        raise NotImplementedError(f"There are no implementation of: {args.optimizer}")

    if args.continue_training:
        model.load_state_dict(torch.load(args.trained_model, map_location=device)['model_state_dict'])
        optimizer.load_state_dict(torch.load(args.trained_model, map_location=device)['optimizer_state_dict'])
        args.start_epoch = torch.load(args.trained_model)['epoch']
        print(f'Loaded model, optimizer, starting with epoch: {args.start_epoch}')

    # Scheduler
    if args.scheduler_name == 'annealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=True)
    elif args.scheduler_name == 'warmup':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, warmup_multiplier=0.01, t_total=args.epochs, verbose=False)
    elif args.scheduler_name == "warmup_restarts":
        scheduler = CosineAnnealingWarmupRestarts(optimizer, warmup_steps=args.warmup_steps, first_cycle_steps=int(args.epochs * (2/3)), cycle_mult=(1/2), gamma=args.scheduler_gamma, max_lr=args.lr, min_lr=1e-6) 
    
    # Metrics
    if args.classes > 1:
        dice_metric = DiceMetric(
            include_background=args.include_background, reduction='mean_batch')
        jaccard_metric = MeanIoU(
            include_background=args.include_background, reduction='mean_batch')
    else:
        dice_metric = DiceMetric(
            include_background=True, reduction='mean_batch')
        jaccard_metric = MeanIoU(
            include_background=True, reduction='mean_batch')


    with experiment.train():

        best_dice_score = 0.0
        best_dice_val_score = 0.0
        accum_iter = args.gradient_accumulation
        for epoch in range(args.start_epoch, args.epochs):
            start_time_epoch = time.time()
            print(f"Starting epoch {epoch + 1}")
            
            running_loss = 0.0
            running_jaccard = 0.0
            running_dice = 0.0
            epoch_time=0.0

            # dice_per_class_train = [0 for i in range(args.classes)]
            # dice_per_class_val = [0 for i in range(args.classes)]

            model.train()
            for batch_idx, train_data in enumerate(train_loader):
                jac, dice, loss = training_step(batch_idx, train_data, args)   
                running_jaccard += jac
                running_dice += dice
                running_loss += loss

            epoch_time=time.time() - start_time_epoch

            dice_metric.reset()
            jaccard_metric.reset()
            print("Training process has finished. Starting testing...")

            val_running_jac = 0.0
            val_running_dice = 0.0

            model.eval()
            with torch.no_grad():
                if epoch % args.validation_interval == 0 and epoch != 0:
                    start_time_validation = time.time()
                    for batch_idx, val_data in enumerate(test_loader):
                        jac, dice = validation_step(batch_idx, val_data, args)
                        val_running_jac += jac
                        val_running_dice += dice
                    val_time=time.time() - start_time_validation
                    print( f"Validation time: {val_time:.2f}s")

                dice_metric.reset()
                jaccard_metric.reset()

                train_loss = running_loss / len(train_loader)

                train_jac = running_jaccard / len(train_loader)
                test_jac = val_running_jac / len(test_loader)

                train_dice = running_dice / len(train_loader)
                test_dice = val_running_dice / len(test_loader)

                scheduler.step()
                experiment.log_metric("lr_rate", scheduler.get_last_lr(), epoch=epoch)
                
                # CHECKPOINTS SAVE
                directory = f"checkpoints/{args.checkpoint_dir}/classes_{str(args.classes)}"
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save best TRAIN model
                if best_dice_score < train_dice:
                    save_path = f"{directory}/model-{args.model_name}-{args.classes}class-fold-{fold}_current_best_train.pt"
                    torch.save({
                            'epoch': (epoch),
                            'model_state_dict': model.state_dict(),
                            'model_val_dice': train_dice,
                            'model_val_jac': train_jac
                            }, save_path)
                    best_dice_score = train_dice
                    print(f"Current best train dice score {best_dice_score:.4f}. Model saved!")
                                
                # save best VALIDATION score
                if best_dice_val_score < test_dice:
                    save_path = f"{directory}/model-{args.model_name}-{args.classes}class-fold-{fold}_current_best_val.pt"
                    torch.save({
                        'epoch': (epoch),
                        'model_state_dict': model.state_dict(),
                        'model_val_dice': test_dice,
                        'model_val_jac': test_jac 
                        }, save_path)
                    best_dice_val_score = test_dice
                    print(f"Current best validation dice score {best_dice_val_score:.4f}. Model saved!")

                #save based on SAVE INTERVAL
                if epoch % args.save_interval == 0 and epoch != 0:
                    save_path = f"{directory}/model-{args.model_name}-{args.classes}class-fold-{fold}_val_{test_dice:.4f}_train_{train_dice:.4f}_epoch_{(epoch+1):04}.pt"
                    #save based on optimiser save interval
                    if args.save_optimizer and epoch % args.save_optimiser_interval == 0 and epoch != 0:
                        torch.save({
                            'epoch': (epoch),
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler' : scheduler.state_dict(),
                            'model_train_dice': train_dice,
                            'model_train_jac': train_jac,
                            'model_val_dice': test_dice,
                            'model_val_jac': test_jac
                            }, save_path)
                        print("Saved optimizer and scheduler state dictionaries.")
                    else:
                        torch.save({
                            'epoch': (epoch),
                            'model_state_dict': model.state_dict(),
                            'model_train_dice': train_dice,
                            'model_train_jac': train_jac,
                            'model_val_dice': test_dice,
                            'model_val_jac': test_jac
                            }, save_path)
                    print(f"Interval model saved! - train_daice: {train_dice:.4f}, val_dice: {test_dice:.4f}, best_val_dice: {best_dice_val_score:.4f}.")

                experiment.log_current_epoch(epoch)
                experiment.log_metric("train_jac", train_jac, epoch=epoch)
                experiment.log_metric("val_jac", test_jac, epoch=epoch)
                experiment.log_metric("train_dice", train_dice, epoch=epoch)
                experiment.log_metric("val_dice", test_dice, epoch=epoch)
                experiment.log_metric("train_loss", train_loss, epoch=epoch)

                print('    ', end='')

                print(f"Joint Loss: {train_loss:.4f} -"
                      f" Train Jaccard: {train_jac:.4f},"
                      f" Test Jaccard: {test_jac:.4f}."
                      f" Train Dice: {train_dice:.4f},"
                      f" Test Dice: {test_dice:.4f}."
                      f" - Total epoch time: {epoch_time:.2f}s")

        print(f"Training finished!")
