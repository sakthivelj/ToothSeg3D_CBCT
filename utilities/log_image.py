from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
import itertools
import subprocess
import glob 
import os
import h5py
import cv2
import pyvista as pv
from matplotlib import colors
from matplotlib import pyplot as plt

pv.global_theme.font.size = 26
pv.global_theme.font.label_size = 22
pv.global_theme.font.color = 'black'

def start_xvfb(display : int = 99, is_jupyter : bool = False):
    print("Starting pyvista xvfb server") 
    xvfb = subprocess.check_output("ps -ef | grep Xvfb | grep screen", shell=True)
    display = f':{display}'
    if display in str(xvfb):
        os.environ['DISPLAY'] = display
        print(f"Xvfb process was working, using DISPLAY={display})")
    else:
        pv.start_xvfb()
        print(f"Xvfb started, using DISPLAY={display}")
    if is_jupyter:
        pv.set_jupyter_backend('panel')

class Logger():

    def __init__(self,
                 num_classes : int = -1,
                 is_log_3d : bool = False,
                 camera_views : list[int] = [3,5,6,7]) -> None:
        
        self.classes_num = num_classes
        camera_positions = list(map(list, itertools.product([-1, 1], repeat=3)))
        self.camera_positions = [ camera_positions[i] for i in camera_views]
        self.camera_positions_LR = [[1,0,0],[-1,0,0]]
        self.camera_positions_AP = [[0,1,0],[0,-1,0]]

        if is_log_3d:
            print("Starting pyvista xvfb server") 
            xvfb = subprocess.check_output("ps -ef | grep Xvfb | grep screen", shell=True)
            if ':99' in str(xvfb):
                os.environ['DISPLAY'] = ':99'
                print("Xvfb process was working, using DISPLAY=:99")
            else:
                pv.start_xvfb()
                print("Xvfb started, using DISPLAY=:99")
            pv.set_jupyter_backend('panel')
            

        tooth_colors = pd.read_csv(
            'csv_files/ToothSegmentColors.txt', delimiter=" ", header=None)
        tooth_colors_df = tooth_colors.iloc[:,2:5]
        tooth_colors_df.columns = ['r', 'g', 'b']
        slicer_colorspace = tooth_colors_df.to_numpy()/255

        if self.classes_num == -1:
            self.color_map = slicer_colorspace
        else:
            self.color_map = slicer_colorspace[:(self.classes_num+1)]
        self.slicer_map = colors.ListedColormap(self.color_map, 'slicer_colors')


    def log_image(self, prediction: torch.tensor, label: torch.tensor, image: torch.tensor) -> np.array:

        x,y,z = prediction, label, image

        w, h, d = x.shape[0]//2-15, x.shape[1]//2, x.shape[2]//2
        slices = []

        #labels
        for img in [x, y]:
            w_sl = np.rot90(img[w, :, :])
            h_sl = np.rot90(img[:, h, :])
            d_sl = img[:, :, d]
            if self.classes_num > 1:
                slices.extend([self.color_map[w_sl], self.color_map[h_sl], self.color_map[d_sl]])
            else:
                slices.extend([w_sl, h_sl, d_sl])
        #source image
        w_sl = np.rot90(z[w, :, :])
        h_sl = np.rot90(z[:, h, :])
        d_sl = z[:, :, d]
        slices.extend([w_sl, h_sl, d_sl])

        slices_norm = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1) for img in slices]
        if self.classes_num > 1:
            horizontal_imgs = [cv2.hconcat([slices_norm[0], slices_norm[3], cv2.cvtColor(slices_norm[6], cv2.COLOR_GRAY2RGB)]),
                               cv2.hconcat([slices_norm[1], slices_norm[4], cv2.cvtColor(slices_norm[7], cv2.COLOR_GRAY2RGB)]),
                               cv2.hconcat([slices_norm[2], slices_norm[5], cv2.cvtColor(slices_norm[8], cv2.COLOR_GRAY2RGB)])]
        else:
            horizontal_imgs = [cv2.hconcat([slices_norm[0], slices_norm[3], slices_norm[6]]),
                               cv2.hconcat([slices_norm[1], slices_norm[4], slices_norm[7]]),
                               cv2.hconcat([slices_norm[2], slices_norm[5], slices_norm[8]])]

        img_log = cv2.vconcat(horizontal_imgs)
        return img_log

    def log_binary(pred : torch.tensor, mask : torch.tensor, image : torch.tensor) -> np.array:
        
        x,y,z = pred, mask, image
        w,h,d = x.shape[0]//2, x.shape[1]//2, x.shape[2]//2

        #prediction
        w_sl = np.rot90(x[w,:,:])
        h_sl = np.rot90(x[:,h,:])
        d_sl = x[:,:,d]
        
        #ground truth
        w_sl_lbl = np.rot90(y[w,:,:])
        h_sl_lbl = np.rot90(y[:,h,:])
        d_sl_lbl = y[:,:,d]

        #source scan
        w_sl_im = np.rot90(z[w,:,:])
        h_sl_im = np.rot90(z[:,h,:])
        d_sl_im = z[:,:,d]

        slices = [w_sl, h_sl, d_sl, w_sl_lbl, h_sl_lbl, d_sl_lbl, w_sl_im, h_sl_im, d_sl_im]
        slices_norm = [cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1) for img in slices]
        dims = [cv2.hconcat([slices_norm[0],slices_norm[3], slices_norm[6]]),
                cv2.hconcat([slices_norm[1],slices_norm[4], slices_norm[7]]),
                cv2.hconcat([slices_norm[2],slices_norm[5], slices_norm[8]])]
        img_log = cv2.vconcat(dims)     
        return img_log  
        # experiment.log_image(img_log, name=f'{epoch:03}_{batch_idx:02}')

    def log_scene(self, volume: np.array, num_classes: int = -1, add_volume_outline=False, scene_size: int = 480, is_zoom: bool = False, val: Optional[Union[str, int]] = 2.0, view: int = 3) -> np.array:
        scene_size = (scene_size,) * 2
        zoomed = None
        labels = dict(xlabel='R', ylabel='P', zlabel='S')

        if num_classes == -1:
            num_classes = self.classes_num
        data = pv.UniformGrid()

        data.dimensions = np.array(volume.shape) + 1
        data.cell_data['values'] = volume.ravel(order='F')
        tresh_data = data.threshold(1, scalars='values')

        p = pv.Plotter(window_size=scene_size, off_screen=True, lighting='three lights')
        p.set_background('#c1c3e8', top='#7579be')
        # p.enable_shadows()
        p.add_axes(line_width=6, ambient=0.5, **labels)

        sargs = dict(
            title='tooth_class',
            title_font_size=16,
            label_font_size=12,
            shadow=True,
            n_labels=self.classes_num+1,
            italic=False,
            fmt="%.0f",
            font_family="arial",
            )

        p.add_mesh(tresh_data, cmap=self.slicer_map, scalars="values", clim=[-.5, num_classes + 0.5], 
                   scalar_bar_args=sargs, smooth_shading=False)

        # ANGLED VIEWS FROM LIST
        views = []
        p.camera.zoom(1.0)
        if add_volume_outline:
            #bounds, center, faces - EMPTY, points - vertices
            outline = data.outline()
            p.add_mesh(outline, color="k")
        for camera_pos in self.camera_positions:
            p.camera_position = camera_pos
            views.append(p.screenshot(return_img=True))

        # LR VIEWS
        for camera_pos in self.camera_positions_LR:
            p.camera_position = camera_pos
            views.append(p.screenshot(return_img=True))

        # AP VIEWS
        for camera_pos in self.camera_positions_AP:
            p.camera_position = camera_pos
            views.append(p.screenshot(return_img=True))

        out_image = cv2.vconcat(
            [cv2.hconcat([views[0], views[1], views[4], views[5]]), cv2.hconcat([views[2], views[3], views[6], views[7]])])

        # ZOOMED CHOSEN VIEW
        if is_zoom:
            # edges = tresh_data.extract_feature_edges(0)
            # p.add_mesh(edges, color="black", line_width=1)
            p.camera_position = self.camera_positions[view]
            p.camera.zoom(val)
            zoomed = p.screenshot(return_img=True)
            return out_image, zoomed

        return out_image

    def log_3dscene_comp(self, volume: np.array, volume_gt: np.array, num_classes: int = -1, scene_size: int = 480, camera_pos : list = [0,-1,0]) -> np.array:
            
            scene_size = (scene_size,) * 2
            labels = dict(xlabel='R', ylabel='P', zlabel='S')

            if num_classes == -1:
                num_classes = self.classes_num-1
            
            data = pv.UniformGrid()
            data_gt = pv.UniformGrid()

            #prediction
            data.dimensions = np.array(volume.shape) + 1
            data.cell_data['values'] = volume.ravel(order='F')
            tresh_data = data.threshold(1, scalars='values')

            #ground_truth
            data_gt.dimensions = np.array(volume_gt.shape) + 1
            data_gt.cell_data['values'] = volume_gt.ravel(order='F')
            tresh_data_gt = data_gt.threshold(1, scalars='values')

            #plotter
            p = pv.Plotter(window_size=scene_size, off_screen=True, lighting='three lights')
            p.set_background('#c1c3e8', top='#7579be')
            p.add_axes(line_width=6, ambient=0.5, **labels)

            sargs = dict(
                title='tooth_class',
                title_font_size=16,
                label_font_size=12,
                shadow=True,
                n_labels=self.classes_num,
                italic=False,
                fmt="%.0f",
                font_family="arial",
                )
            
            #PLOT SCENES
            # prediction
            pred = p.add_mesh(tresh_data, cmap=self.slicer_map, scalars="values", clim=[-0.5, num_classes + 0.5], 
                            scalar_bar_args=sargs, smooth_shading=False)

            p.camera_position= [0,-1,0]
            pred_scene_PA = p.screenshot(return_img=True)
            p.camera_position= [0,1,0]
            pred_scene_AP = p.screenshot(return_img=True)
            _ = p.remove_actor(pred)
            pred_image = cv2.hconcat([pred_scene_PA, pred_scene_AP])
            
            # ground_truth
            gt = p.add_mesh(tresh_data_gt, cmap=self.slicer_map, scalars="values", clim=[-0.5, num_classes + 0.5], 
                            scalar_bar_args=sargs, smooth_shading=False)
            p.camera_position= [0,-1,0]
            gt_scenePA = p.screenshot(return_img=True)    
            p.camera_position= [0,1,0]
            gt_sceneAP = p.screenshot(return_img=True)
            _ = p.remove_actor(gt)
            gt_image = cv2.hconcat([gt_scenePA, gt_sceneAP])

            out_image = cv2.vconcat([pred_image, gt_image])
            
            return out_image
    
def log_simple(volume: np.array, color_map : colors.ListedColormap, clim : list = [-0.5, 32.5], tresh_background : int = 1, original_volume: np.array = None, draw_volume : bool = True, draw_bounding : bool = False, bounding_margin :float = 0.1, camera_pos : list= None) -> pv.Plotter:

    data = pv.UniformGrid()
    data.dimensions = np.array(volume.shape) + 1
    data.cell_data['values'] = volume.ravel(order='F')
    p = pv.Plotter(window_size=(500,500), off_screen=True, lighting='three lights')

    if draw_bounding:
        p.add_mesh(data.outline(), color="b")

    if draw_volume:
        if original_volume is not None:
            data = pv.UniformGrid()
            data.dimensions = np.array(original_volume.shape) + 1
            data.cell_data['values'] = original_volume.ravel(order='F')
                                            
        tresh_data = data.threshold(tresh_background, scalars='values')
        p.add_mesh(tresh_data, scalars="values", cmap=color_map, clim=clim, smooth_shading=True)
    if camera_pos is not None:
        p.camera_position = camera_pos
    return p



