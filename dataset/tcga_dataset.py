
import time
import skimage.io
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
# from efficientnet_pytorch import model as enet
import albumentations
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
import os
from openslide import OpenSlide
import random
import pickle

from  albumentations import (HueSaturationValue, ShiftScaleRotate, OpticalDistortion, PadIfNeeded,
                                RandomBrightnessContrast, GridDistortion, RandomGridShuffle,
                             RandomSizedCrop, Normalize, Compose,
                            Rotate)

class TCGADataset(Dataset):
    def __init__(self, is_train=True, valid_fold=0, patch_mode=True, expand_mod=False, distile_mod=False):#size=512, 
        self.data_dir = '/mnt/data/tcga/dataset/'
        self.image_folder = self.data_dir + 'imgs'
        self.patch_folder = self.data_dir + 'img_patch_2'
        self.distile_mod = distile_mod
        self.expand_mod =  expand_mod
        df = pd.read_csv(os.path.join(self.data_dir, 'train_detail_expand.csv'), sep='\t')
        # if expand_mod:
        #     df = pd.read_csv(os.path.join(self.data_dir, 'train_detail_expand.csv'), sep='\t')
        # else:    
        #     df = pd.read_csv(os.path.join(self.data_dir, 'train_detail.csv'), sep='\t')
        df_train = df[df['fold']!=valid_fold]
        df_valid = df[df['fold']==valid_fold]
        small_patch_size = 768
        self.small_patch_size = (small_patch_size, small_patch_size)
        if is_train:
            self.df = df_train.reset_index(drop=True) 
        else:
            self.df = df_valid.reset_index(drop=True) 

        self.names = self.df['FILE_NAME'].unique()
        self.patch_mode = patch_mode
        self.is_train = is_train
        if is_train:
            self.albumentation_crop = Compose([
                HueSaturationValue(),
                Rotate(),
                RandomBrightnessContrast(),
                OpticalDistortion(), 
                GridDistortion(),
                PadIfNeeded(int(small_patch_size*1.1)+1, int(small_patch_size*1.1)+1, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
                RandomSizedCrop((int(small_patch_size*0.9)-1, int(small_patch_size*1.1)+1), small_patch_size, small_patch_size),#, always_apply=True
                Normalize(),
            ])
        else:
            self.albumentation_crop = Compose([
                PadIfNeeded(small_patch_size, small_patch_size, border_mode=cv2.BORDER_CONSTANT, value=(255,255,255)),
                RandomSizedCrop((small_patch_size, small_patch_size), small_patch_size, small_patch_size),#, always_apply=True
                Normalize(),
            ])
        self.albumentation_all = Compose([
            Normalize()#always_apply=True
        ])


    def __getitem__(self, i):
        

#         file_names = row.paths
#         tiff_file = os.path.join(self.image_folder, file_name)
#         image = OpenSlide(tiff_file)
        if self.expand_mod:
            row = self.df.iloc[i]
            # img_fn = row.patch_name
            # label = row.patch_label
        else:
            wsl_name = self.names[i]
            row = random.choice(self.df[self.df['FILE_NAME']==wsl_name])
        img_fn = row.patch_name
        label = eval(row.patch_label)
        pic_name = os.path.join(self.patch_folder, img_fn)
#         if not os.path.exists(pic_name):       
#             self.save_patchs(row, patch_size=self.small_patch_size, read_level=2, i=None)
        img = cv2.imread(pic_name)
        
        if img is None:
               print(pic_name)
#         patch = self.get_patchs(image, bbox)
        # i1 = image.read_region((0,0), 2, image.level_dimensions[2])
        # scale = image.level_dimensions[target_level][1] / image.level_dimensions[2][1]
#         print(img.shape)
        img = self.albumentation_crop(image=img)['image']
        img = img.transpose((2,0,1))
        return img, np.array(label)
    
    def __len__(self):
        if self.expand_mod:
            return len(self.df)
        else:
            return len(self.names)
        # return 40
    @staticmethod
    def random_box_center(xywh, target_shape):
        xy, wh = xywh[:2], xywh[2:]
        xc, yc = (xy+wh) / 2
        xy_min_range, xy_max_range = (xy+target_shape).astype(np.int), (xy+wh-target_shape).astype(np.int)
        x_range = list(range(xy_min_range[0], xy_max_range[0])) if xy_max_range[0] > xy_min_range[0] else [xc]
        y_range = list(range(xy_min_range[1], xy_max_range[1])) if xy_max_range[1] > xy_min_range[1] else [yc]
        x_c, y_c = random.choice(x_range), random.choice(y_range)
        return np.array((x_c-target_shape[0]//2, y_c-target_shape[1]//2, target_shape[1], target_shape[1]))

    @staticmethod
    def grid_box_cut(xywh, target_shape):
        xy, wh = xywh[:2], xywh[2:]
        n = wh / target_shape
        margin = ((n - n.astype(np.int))/2 * target_shape).astype(np.int)
        coords = []
        n = n.astype(np.int)
        if (n==0).any():
            return [xywh]
        # if n[0]==0:
        #     n[0] = 1
        #     margin[0] = -margin[0]
        # if n[1]==0:
        #     n[1] = 1
        #     margin[1] = -margin[1]
        for i in range(n[0]):
            for j in range(n[1]):
                coords.append(np.array([margin[0] + xy[0] + i*target_shape[0], margin[1] + xy[1] + j*target_shape[1], target_shape[0], target_shape[1]]).astype(np.int)) 
        return coords

    @staticmethod
    def get_statics(patch):
        # black = (255,255,255) - patch
        black_binary = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        black_binary = black_binary < 240 
        pixel_percent = black_binary.sum() / black_binary.shape[0] / black_binary.shape[1]
        # black = (255,255,255) - patch
        # print(black.sum()/)
        return pixel_percent

    def filter_img(self, imgs0, forground_percentage_thresh=0.3):
        imgs = list(filter(lambda x:self.get_statics(x)>forground_percentage_thresh, imgs0))
        if len(imgs)==0:
            return imgs0
        return imgs

    def get_patchs_patch(self, img, bbox, read_level, patch_size=(1000,1000), i=None):
        '''
        read_level only in [1 2]
        '''
        appmag = img.properties['aperio.AppMag']
        level_count = img.level_count
        if level_count == 2:
            target_level = 1
        elif level_count == 3 or level_count == 4:
            target_level = 2
        else:
            print('level count not in [2,3,4]')
        target_scale = 4**read_level
        # target_level = 1
        if appmag == '40':
            # if level_count==4:
                # scale_level = img.level_dimensions[0][0] / img.level_dimensions[1][0]# 最后两个scale是2 和之前的不同
                # if scale_level<3: print(i, 'badimg', appmag, level_count, scale_level)
            scale_level = 0.25 ** (target_level-read_level)
            scale_resize = 1.
            # elif level_count==3:
            #     scale_level = 0.25 ** (target_level-read_level)
            #     scale_resize = 1.0
            if level_count not in [3,4]:
                print(i, 'badimg', appmag, level_count)
                return None
        elif appmag == '20':

            if level_count==3 or level_count==4:
                _scale = img.level_dimensions[target_level][0] / img.level_dimensions[read_level][0]
                scale_level = _scale ** (target_level-read_level)
                scale_resize = (2. * (img.level_dimensions[0][0] / img.level_dimensions[read_level][0])) / target_scale
            elif level_count==2:
                scale_level = 1.
                scale_resize = (2. * (img.level_dimensions[0][0] / img.level_dimensions[1][0])) / target_scale
            else:
                print(i, 'badimg', appmag, level_count)
                return None
        else:
            print(i, 'badimg', appmag, level_count)
            return None
        scale_0 = img.level_dimensions[0][1] / img.level_dimensions[target_level][1]
        cut_box = np.array(bbox, dtype=np.float)
        cut_boxes = self.grid_box_cut(cut_box/scale_level, np.array(patch_size))
        patches = []
        # cut_boxes = random.sample(cut_boxes, 8)
        self.down_sample=1
        for cut_box in cut_boxes:
            cut_box_xy = (np.round(cut_box[:2] * scale_0 / self.down_sample, 0)).astype(np.int)
            cut_box_wh = (np.round(cut_box[2:] / self.down_sample , 0)).astype(np.int)
            p = img.read_region(cut_box_xy, read_level, cut_box_wh)
            patch = np.array(p)[:,:,:3]

            w, h = img.level_dimensions[target_level]
            x2, y2 = cut_box[:2] + cut_box[2:]
            if x2>w:
                patch[int(x2/scale_level):,:,:] = (255,255,255)
            if y2>h:
                patch[:, int(y2/scale_level):,:] = (255,255,255)

            if scale_resize!=1.0:
                patch = cv2.resize(patch, dsize=(0,0), fx=scale_resize, fy=scale_resize)
            patches.append(patch)
        
        return self.filter_img(patches)


    def get_patchs_whole(self, img, bbox, i=None):

        appmag = img.properties['aperio.AppMag']
        level_count = img.level_count
        if level_count < 3:
            target_level = level_count-1
        else:
            target_level = 2
        if appmag == '40':
            if level_count==4:
                scale_level = img.level_dimensions[2][0] / img.level_dimensions[3][0]# 最后两个scale是2 和之前的不同
                read_level = 3
                scale_resize = scale_level/4
            elif level_count==3:
                scale_level = 1
                read_level = target_level
                scale_resize = 0.25
            else:
                print(i, 'badimg', appmag, level_count)
                return None
        elif appmag == '20':
            if level_count==3:
                scale_level = 1.
                read_level = target_level
                scale_resize = 0.50
            elif level_count==2:
                scale_level = 1
                read_level = target_level
                scale_resize = 1/2
            else:
                print(i, 'badimg', appmag, level_count)
                return None
        else:
            print(i, 'badimg', appmag, level_count)
            return None
        scale_0 = img.level_dimensions[0][1] / img.level_dimensions[target_level][1]
        cut_box = np.array(bbox, dtype=np.float)
        cut_box_xy = (np.round(cut_box[:2] * scale_0 / self.down_sample, 0)).astype(np.int)
        cut_box_wh = (np.round(cut_box[2:] / self.down_sample / scale_level, 0)).astype(np.int)
        p = img.read_region(cut_box_xy, read_level, cut_box_wh)
        patch = np.array(p)[:,:,:3]

        w, h = img.level_dimensions[target_level]
        x2, y2 = cut_box[:2] + cut_box[2:]
        if x2>w:
            patch[x2//scale_0:] = 255
        if y2>h:
            patch[:, y2//scale_0:] = 255

        if scale_resize!=1.0:
            patch = cv2.resize(patch, dsize=(0,0), fx=scale_resize, fy=scale_resize)
        return patch

    def show_patchs(self, i, read_level=2):
        row = self.df.iloc[i] 
        file_name = row.FILE_NAME
        tiff_file = os.path.join(self.image_folder, file_name)
        image = OpenSlide(tiff_file)
        level_count =  image.level_count    
        appmag = image.properties['aperio.AppMag']
        bboxes = eval(row.bboxes)
        imgs = []
        print(image.level_dimensions[read_level])
        aa = image.read_region((0,0), level_count-1, image.level_dimensions[read_level-1])
        aa.show()
        for b in bboxes:
            print(bboxes)
            if b[2]>750 and b[3]>750:
                if self.patch_mode:
                    p = self.get_patchs_patch(image, b, read_level, patch_size=(1000,1000), i=i)
                else:
                    p = self.get_patchs_whole(image, b, i)
                if p is not None:
                    imgs.extend(p)
        for img in imgs:
            plt.figure()
            plt.imshow(img)

    def save_patchs(self, row, patch_size, read_level, i=None):
#         row = self.df.iloc[i] 
        file_name = row.FILE_NAME
        tiff_file = os.path.join(self.image_folder, file_name)
        
        image = OpenSlide(tiff_file)
        level_count =  image.level_count    
        appmag = image.properties['aperio.AppMag']
        bboxes = eval(row.bboxes)
        imgs = []

        for b in bboxes:
            if b[2]>750 and b[3]>750:
                if self.patch_mode:
                    p = self.get_patchs_patch(image, b, read_level=read_level, patch_size=patch_size, i=i)
                else:
                    p = self.get_patchs_whole(image, b, i)
                if p is not None:
                    imgs.extend(p)
        # imgs = random.sample(imgs, 6)if len(imgs)>6 else imgs  

        for i, img in enumerate(imgs):
            cv2.imwrite(os.path.join(self.patch_folder, file_name.replace('.svs', f'_{i}.png')), img)

