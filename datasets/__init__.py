# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F
from importlib import import_module
import misc.transforms as own_transforms
from  misc.transforms import  check_image
import torchvision.transforms as standard_transforms
from . import dataset
from . import setting
from . import samplers
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from config import  cfg
from PIL import Image
import random
from torch.utils.data.distributed import DistributedSampler
class train_pair_transform(object):
    def __init__(self, cfg_data, check_dim = True):
        self.cfg_data = cfg_data
        self.pair_flag = 0
        self.scale_factor = 1
        self.last_cw_ch =(0, 0)
        self.crop_left = (0, 0)
        self.last_crop_left = (0, 0)
        self.rate_range = (0.8, 1.2)
        self.resize_and_crop= own_transforms.RandomCrop()
        self.scale_to_setting = own_transforms.ScaleByRateWithMin(cfg_data.TRAIN_SIZE[1], cfg_data.TRAIN_SIZE[0])

        self.flip_flag = 0
        self.horizontal_flip = own_transforms.RandomHorizontallyFlip()

        self.last_frame_size = (0,0)

        self.check_dim = check_dim
    def __call__(self, img, target):
        self.scale_factor = random.uniform(self.rate_range[0], self.rate_range[1])
        self.c_h, self.c_w = int(self.cfg_data.TRAIN_SIZE[0]/self.scale_factor), int(self.cfg_data.TRAIN_SIZE[1]/self.scale_factor)
        img, target = check_image(img, target, (self.c_h, self.c_w), 
                                  (self.cfg_data.TRAINING_MAX_LONG, self.cfg_data.TRAINING_MAX_SHORT))  # make sure the img size is large than we needed
        w, h = img.size
        if self.pair_flag % 2 == 0:
            self.last_cw_ch = (self.c_w, self.c_h)
            self.pair_flag = 0

            x1 = random.randint(0, w - self.c_w)
            y1 = random.randint(0, h - self.c_h)
            self.last_crop_left = (x1,y1)

        if self.pair_flag % 2 == 1:
            if self.check_dim:
                x1 = max(0, int(self.last_crop_left[0] + (self.last_cw_ch[0]-self.c_w)))
                y1 = max(0, int(self.last_crop_left[1] + (self.last_cw_ch[1]-self.c_h)))
            else:   # for pre_training on other dataset
                x1 = random.randint(0, w - self.c_w)
                y1 = random.randint(0, h - self.c_h)
        self.crop_left = (x1, y1)

        img, target = self.resize_and_crop(img, target, self.crop_left, crop_size=(self.c_h, self.c_w))
        img, target = self.scale_to_setting(img, target)

        target["points"][:, 0] = torch.clamp(target["points"][:, 0], min=0, max=img.size[0]-1)
        target["points"][:, 1] = torch.clamp(target["points"][:, 1], min=0, max=img.size[1]-1)
        self.flip_flag = round(random.random())
        img, target = self.horizontal_flip(img, target, self.flip_flag)
        self.pair_flag += 1

        return img, target

class test_transform(object):
    def __init__(self, cfg_data):
        self.cfg_data = cfg_data
    def __call__(self, img, target):
        w, h = img.size
        long_side = max(w, h)
        short_side = min(w, h)
        if self.cfg_data.TEST_MAX_LONG is not None and self.cfg_data.TEST_MAX_SHORT is not None:
            max_long_side, max_short_side = self.cfg_data.TEST_MAX_LONG, self.cfg_data.TEST_MAX_SHORT
            scale_long = max_long_side / long_side
            scale_short = max_short_side / short_side
            if scale_long < 1 or scale_short < 1:
                scale = min(scale_long, scale_short)
                new_width = int(w * scale)
                new_height = int(h * scale)
                target['points'] = target['points'] * scale
                img = img.resize((new_width, new_height), Image.LANCZOS)

        target["points"][:, 0] = torch.clamp(target["points"][:, 0], min=0, max=img.size[0]-1)
        target["points"][:, 1] = torch.clamp(target["points"][:, 1], min=0, max=img.size[1]-1)

        return img, target

class train_resize_transform(object):
    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w
        self.horizontal_flip = own_transforms.RandomHorizontallyFlip()

    def __call__(self, img, target):
        w, h = img.size
        img = img.resize((self.new_w, self.new_h), Image.LANCZOS)

        rate_w = self.new_w / w
        rate_h = self.new_h / h
        target['points'][:, 0] =  target['points'][:, 0]  * rate_w
        target['points'][:, 1] =  target['points'][:, 1]  * rate_h
        target["points"][:, 0] = torch.clamp(target["points"][:, 0], min=0, max=self.new_w-1)
        target["points"][:, 1] = torch.clamp(target["points"][:, 1], min=0, max=self.new_h-1)

        self.flip_flag = round(random.random())
        img, target = self.horizontal_flip(img, target, self.flip_flag)
        return img, target

class train_transform(object):
    def __init__(self, cfg_data, check_dim = True):
        self.cfg_data = cfg_data
        self.crop = own_transforms.RandomCrop()
        self.scale_to_setting = own_transforms.Scale(cfg_data.TRAIN_SIZE[1], cfg_data.TRAIN_SIZE[0])
        self.check_dim = check_dim
        self.final_h = cfg_data.TRAIN_SIZE[0]
        self.final_w = cfg_data.TRAIN_SIZE[1]

    def __call__(self, img, target):
        img, target = check_image(img, target, (self.final_h, self.final_w))  # make sure the img size is large than we needed
        w_ori, h_ori = img.size
        points = target['points']
        if len(points) > 0:
            x_mean = int(points[:, 0].mean().item())
            y_mean = int(points[:, 1].mean().item())
        else:
            x_mean = w_ori // 2
            y_mean = h_ori // 2
            
        x_range_start = max(0, x_mean - self.final_w // 2)
        x_range_end = min(x_mean + self.final_w // 2, w_ori)
        y_range_start = max(0, y_mean - self.final_h // 2)
        y_range_end = min(y_mean + self.final_h // 2, h_ori)

        x_crop_center = random.randint(x_range_start, x_range_end-1)
        y_crop_center = random.randint(y_range_start, y_range_end-1)

        y1, y2 = max(0, y_crop_center - self.final_h // 2), min(h_ori, y_crop_center + self.final_h // 2)
        x1, x2 = max(0, x_crop_center - self.final_w // 2), min(w_ori, x_crop_center + self.final_w // 2)
        self.crop_left = (x1, y1)

        img, target = self.crop(img, target, self.crop_left, crop_size=(y2-y1, x2-x1))
        w, h = img.size
        if w != self.final_w or h != self.final_h:
            img, target = self.scale_to_setting(img, target)
        
        target["points"][:, 0] = torch.clamp(target["points"][:, 0], min=0, max=self.final_w-1)
        target["points"][:, 1] = torch.clamp(target["points"][:, 1], min=0, max=self.final_h-1)

        return img, target
    
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    data = list(zip(*batch))
    img_tensors = []
    labels = []
    for i in range(len(data[0])):
        try:
            img_tensors.append(torch.stack(data[0][i], dim=0))
        except:
            pass

        try:
            labels += data[1][i]
        except:
            pass

    try:
        img_tensors = torch.cat(img_tensors, dim=0)
    except:
        pass
    return img_tensors, labels

def createTrainData(datasetname, Dataset, cfg_data, distributed):
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])

    pair_transform = train_pair_transform(cfg_data)
    resize_transform = train_resize_transform(cfg_data.TRAIN_SIZE[0], cfg_data.TRAIN_SIZE[1])
    train_set = Dataset(cfg_data.TRAIN_LST,
                        cfg_data.DATA_PATH,
                        main_transform= resize_transform if datasetname == "UAVVIC" else pair_transform,
                        img_transform=img_transform,
                        train=True,
                        datasetname=datasetname,
                        frame_intervals=cfg_data.TRAIN_FRAME_INTERVALS)
    sampler_train = DistributedSampler(train_set) if distributed else None
    train_loader = DataLoader(train_set, 
                              batch_size=cfg_data.TRAIN_BATCH_SIZE, 
                              sampler=sampler_train, 
                              shuffle=False,
                              num_workers=8, 
                              collate_fn=collate_fn, 
                              pin_memory=True)
    print('dataset is {}, training images num is {}'.format(datasetname, train_set.__len__()))

    return  train_loader, sampler_train
def createValData(datasetname, Dataset, cfg_data):


    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*cfg_data.MEAN_STD)
    ])

    val_loader = []
    with open(os.path.join( cfg_data.DATA_PATH, cfg_data.VAL_LST), 'r') as txt:
        scene_names = txt.readlines()
    for scene in scene_names:
        sub_val_dataset = Dataset([scene.strip()],
                                  cfg_data.DATA_PATH,
                                  main_transform=None,
                                  img_transform= img_transform ,
                                  train=False,
                                  datasetname=datasetname)
        sub_val_loader = DataLoader(sub_val_dataset, batch_size=cfg_data.VAL_BATCH_SIZE, num_workers=4,collate_fn=collate_fn,pin_memory=False )
        val_loader.append(sub_val_loader)

    return  val_loader
def createRestore(mean_std):
    return standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

def loading_data(datasetname, val_interval, distributed, is_main):
    if datasetname != "MovingDroneCrowd":
        datasetname = datasetname.upper()
    cfg_data = getattr(setting, datasetname).cfg_data

    Dataset = dataset.Dataset
    train_loader, sampler_train = createTrainData(datasetname, Dataset, cfg_data, distributed)
    restore_transform = createRestore(cfg_data.MEAN_STD)

    Dataset = dataset.TestDataset
    val_loader = createValTestData(datasetname, Dataset, cfg_data, val_interval, True, is_main, mode ='val')

    return train_loader, sampler_train, val_loader, restore_transform

def createValTestData(datasetname, Dataset, cfg_data, frame_interval, skip_flag, is_main, mode ='val'):
    if is_main:
        img_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*cfg_data.MEAN_STD)
        ])
        main_transform = test_transform(cfg_data)
        with open(os.path.join(cfg_data.DATA_PATH, eval('cfg_data.{}_LST'.format(mode.upper()))), 'r') as txt:
            scene_names = txt.readlines()
            scene_names = [i.strip() for i in scene_names]
        target = True
        if datasetname == 'MovingDroneCrowd':
            last_scene_names = []
            for scene_name in scene_names:
                root  = os.path.join(cfg_data.DATA_PATH, 'frames', scene_name)
                if '/' in scene_name:
                    scene_name, clip_names = scene_name.split('/')
                    clip_names = [clip_names]
                else:
                    clip_names = [clip_name for clip_name in os.listdir(root) if not '.' in clip_name]
                    clip_names.sort()
                
                for clip_name in clip_names:
                    scene_clip_name = "{}/{}".format(scene_name, clip_name)
                    last_scene_names.append(scene_clip_name)
            scene_names = last_scene_names
        data_loader = []
        for scene_name in scene_names:
            print(scene_name)
            sub_dataset = Dataset(scene_name = scene_name,
                                base_path=cfg_data.DATA_PATH,
                                main_transform=main_transform,
                                img_transform=img_transform,
                                interval=frame_interval,
                                skip_flag=skip_flag,
                                target=target,
                                datasetname = datasetname)
            sub_loader = DataLoader(sub_dataset, 
                                    batch_size=cfg_data.VAL_BATCH_SIZE, 
                                    shuffle=False,
                                    collate_fn=collate_fn, 
                                    num_workers=0, 
                                    pin_memory=True)
            data_loader.append([scene_name, sub_loader])
        return data_loader

    else:
        return None
    
def loading_testset(datasetname, test_interval, skip_flag, mode='test'):
    if datasetname != "MovingDroneCrowd":
        datasetname = datasetname.upper()
    cfg_data = getattr(setting, datasetname).cfg_data

    Dataset = dataset.TestDataset

    test_loader = createValTestData(datasetname, Dataset, cfg_data, test_interval, skip_flag, True, mode=mode)

    restore_transform = createRestore(cfg_data.MEAN_STD)
    return test_loader, restore_transform