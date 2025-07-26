#!/usr/bin/env python
# coding: utf-8

import os.path as osp
import os
from collections import defaultdict
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import json
from torchvision.ops.boxes import clip_boxes_to_image
from PIL import Image
import re
from copy import deepcopy
import random
class Dataset(data.Dataset):
    """
    Dataset class.
    """
    def __init__(self, txt_path, base_path, main_transform=None, img_transform=None, train=True, datasetname='Empty', frame_intervals=(3,6)):
        self.base_path = base_path
        self.bboxes = defaultdict(list)
        self.imgs_path = []
        self.labels = []
        self.datasetname = datasetname
        with open(osp.join(base_path, txt_path), 'r') as txt:
            scene_names = txt.readlines()
            scene_names = [i.strip() for i in scene_names]
        if datasetname == 'MovingDroneCrowd':
            last_scene_names = []
            for scene_name in scene_names:
                root  = os.path.join(base_path, 'frames', scene_name)
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

        for scene_name in scene_names:
            if datasetname == 'MovingDroneCrowd':
                img_path, label = MDC_ImgPath_and_Target(base_path, scene_name.strip())
            elif datasetname == 'UAVVIC':
                img_path, label = UAVVIC_ImgPath_and_Target(base_path, scene_name.strip())
            else:
                raise NotImplementedError
            self.imgs_path += img_path
            self.labels += label

        self.scenes = []
        self.scene_id = {}
        for idx, label in enumerate(self.labels):
            scene_name = label['scene_name']
            if scene_name not in self.scene_id.keys():
                self.scene_id.update({scene_name:0})
            self.scene_id[scene_name]+=1
            self.scenes.append(scene_name)

        self.n_sample = len(self.imgs_path)
        self.is_train = train
        self.main_transforms = main_transform
        self.img_transforms = img_transform
        self.frame_intervals = frame_intervals
    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        c = index
        scene_name = self.scenes[c]
        tmp_intervals = random.randint(self.frame_intervals[0],
                                        min(self.scene_id[scene_name]//2, self.frame_intervals[1]))
        if c < self.n_sample - tmp_intervals:
            if self.scenes[c + tmp_intervals] == scene_name:
                pair_c = c + tmp_intervals
            else:
                pair_c = c
                c = c- tmp_intervals
        else:
            pair_c = c
            c = c - tmp_intervals
        assert self.scenes[c] == self.scenes[pair_c]

        img0 = Image.open(self.imgs_path[c])
        img1 = Image.open(self.imgs_path[pair_c])
        if img0.mode is not 'RGB':
            img0 = img0.convert('RGB')

        if img1.mode is not 'RGB':
            img1 = img1.convert('RGB')

        target0 = deepcopy(self.labels[c])
        target1 = deepcopy(self.labels[pair_c])

        img0, target0 = self.main_transforms(img0, target0)
        img1, target1 = self.main_transforms(img1, target1)

        if 'person_id' in target0:
            ids0 = target0['person_id']
            ids1 = target1['person_id']

            share_mask0 = torch.isin(ids0, ids1)
            share_mask1 = torch.isin(ids1, ids0)

            outflow_mask = torch.logical_not(share_mask0)
            inflow_mask = torch.logical_not(share_mask1)

        elif 'inflow' in target0:
            outflow_mask = target0['outflow'].bool()
            inflow_mask = target1['inflow'].bool()
            share_mask0 = torch.logical_not(outflow_mask)
            share_mask1 = torch.logical_not(inflow_mask)
        
        count_in_pair=[target0['points'].size(0), target1['points'].size(0)]
        if not ((np.array(count_in_pair) > 0).all() and torch.sum(share_mask0) > 2):
            return self.__getitem__((index+1)%len(self))
        
        target0['share_mask0'] = share_mask0
        target0['outflow_mask'] = outflow_mask
        target1['share_mask1'] = share_mask1
        target1['inflow_mask'] = inflow_mask

        if self.img_transforms is not None:
            img0 = self.img_transforms(img0)
            img1 = self.img_transforms(img1)

        return  [img0, img1], [target0, target1]

def HT21_ImgPath_and_Target(base_path, i):
    img_path = []
    labels=[]
    root  = osp.join(base_path, i + '/img1')
    img_ids = os.listdir(root)
    img_ids = sorted(img_ids, key=lambda x:int(x.split('.')[0]))
    gts = defaultdict(list)
    with open(osp.join(root.replace('img1', 'gt'), 'gt.txt'), 'r') as f:
        lines = f.readlines()
        for lin in lines:
            lin_list = [float(i) for i in lin.rstrip().split(',')]
            ind = int(lin_list[0])
            gts[ind].append(lin_list)

    for img_id in img_ids:
        img_id = img_id.strip()
        single_path = osp.join(root, img_id)
        annotation  = gts[int(img_id.split('.')[0])]
        annotation = torch.tensor(annotation,dtype=torch.float32)
        box = annotation[:,2:6]
        points = box[:,0:2] + box[:,2:4]/2

        sigma = torch.min(box[:,2:4], 1)[0] / 2.
        ids = annotation[:,1].long()
        img_path.append(single_path)

        labels.append({'scene_name':i,'frame':int(img_id.split('.')[0]), 'person_id':ids, 'points':points,'sigma':sigma})
    return img_path, labels

def SENSE_ImgPath_and_Target(base_path, i):
    img_path = []
    labels=[]
    root  = osp.join(base_path, 'video_ori', i )
    img_ids = os.listdir(root)
    img_ids.sort()
    gts = defaultdict(list)
    with open(root.replace('video_ori', 'label_list_all')+'.txt', 'r') as f: #label_list_all_rmInvalid
        lines = f.readlines()
        for lin in lines:
            lin_list = [i for i in lin.rstrip().split(' ')]
            ind = lin_list[0]
            lin_list = [float(i) for i in lin_list[3:] if i != '']
            assert len(lin_list) % 7 == 0
            gts[ind] = lin_list

    for img_id in img_ids:
        img_id = img_id.strip()
        single_path = osp.join(root, img_id)
        label = gts[img_id]
        box_and_point = torch.tensor(label).view(-1, 7).contiguous()

        points = box_and_point[:, 4:6].float()
        ids = (box_and_point[:, 6]).long()

        if ids.size(0)>0:
            sigma = 0.6*torch.stack([(box_and_point[:,2]-box_and_point[:,0])/2,(box_and_point[:,3]-box_and_point[:,1])/2],1).min(1)[0]  #torch.sqrt(((box_and_point[:,2]-box_and_point[:,0])/2)**2 + ((box_and_point[:,3]-box_and_point[:,1])/2)**2)
        else:
            sigma = torch.tensor([])
        img_path.append(single_path)

        labels.append({'scene_name':i,'frame':int(img_id.split('.')[0]), 'person_id':ids, 'points':points, 'sigma':sigma})
    return img_path, labels

def UAVVIC_ImgPath(base_path, scene_name):
    img_path = []
    scene_img_root = os.path.join(base_path, "frames", scene_name)
    img_ids = os.listdir(scene_img_root)
    img_ids = sorted(img_ids, key=lambda x:int(x.split('.')[0]))
    for img_id in img_ids:
        img_id = img_id.strip()
        single_path = os.path.join(scene_img_root, img_id)
        img_path.append(single_path)

    return img_path

def UAVVIC_ImgPath_and_Target(base_path, scene_name):
    img_path = []
    labels=[]
    scene_img_root = os.path.join(base_path, "videos", scene_name)
    img_ids = os.listdir(scene_img_root)
    img_ids = sorted(img_ids, key=lambda x:int(x.split('.')[0]))
    ann_root = os.path.join(base_path, 'annotations', scene_name.replace('Video', ''))
    for img_id in img_ids:
        img_id = img_id.strip()
        single_path = os.path.join(scene_img_root, img_id)
        gt_path = os.path.join(ann_root, img_id.replace("jpg", "jsonl"))
        if not os.path.exists(gt_path):
            continue
        with open(gt_path, "r") as f:
            data = json.load(f)
        if len(data) > 0:
            data = torch.tensor(data)
        else:
            data = torch.empty((0, 7))
        data = data[data[:, 4] == 1]
        boxes = data[:, 0:4].float()

        points = torch.zeros((len(boxes), 2))
        points[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) 
        points[:, 1] = (boxes[:, 1] + boxes[:, 3] * 0.1) 

        inflow = data[:, 5].long()
        outflow = data[:, 6].long()
        img_path.append(single_path)

        labels.append({'scene_name':"{}".format(scene_name),'frame':int(img_id.split('.')[0]), 'inflow':inflow, 'outflow':outflow, 'points':points}) 

    return img_path, labels

def MDC_ImgPath_and_Target(base_path, scene_name):
    img_path = []
    labels=[]
    root  = osp.join(base_path, 'frames', scene_name)
    img_ids = os.listdir(root)
    img_ids = sorted(img_ids, key=lambda x:int(x.split('.')[0]))
    df = pd.read_csv(root.replace('frames', 'annotations') +'.csv', header=None)
    grouped = df.groupby(df[0])
    gts = {frame_id: group for frame_id, group in grouped}

    for img_id in img_ids:
        img_id = img_id.strip()
        single_path = osp.join(root, img_id)
        if int(img_id.split('.')[0])-1 < len(gts):
            label = gts[int(img_id.split('.')[0])-1]
            data =  torch.tensor(label.to_numpy())[:, 1:6].contiguous()
        else:
            label = []
            data = torch.empty((0, 5))
        
        boxes = data[:, 1:5].float()

        points = torch.zeros((len(boxes), 2))
        points[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) 
        points[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) 

        ids = (data[:, 0]).long()

        img_path.append(single_path)

        labels.append({'scene_name':scene_name,'frame':int(img_id.split('.')[0]), 'person_id':ids, 'points':points})
    return img_path, labels

def MDC_ImgPath(base_path, scene_name):
    img_path = []
    root  = osp.join(base_path, 'frames', scene_name)
    img_ids = os.listdir(root)
    img_ids = sorted(img_ids, key=lambda x:int(x.split('.')[0]))

    for img_id in img_ids:
        img_id = img_id.strip()
        single_path = osp.join(root, img_id)
        img_path.append(single_path)

    return img_path

class TestDataset(data.Dataset):
    """
    Dataset class.
    """
    def __init__(self, scene_name, base_path, main_transform=None, img_transform=None, interval=1, skip_flag=True, target=True, datasetname='Empty'):
        self.base_path = base_path
        self.target = target

        if self.target:
            if datasetname == 'MovingDroneCrowd':
                self.imgs_path, self.label = MDC_ImgPath_and_Target(self.base_path, scene_name)
            elif datasetname == 'UAVVIC':
                self.imgs_path, self.label = UAVVIC_ImgPath_and_Target(self.base_path, scene_name)
            else:
                raise NotImplementedError
        else:
            if datasetname == 'MovingDroneCrowd':
                self.imgs_path = MDC_ImgPath(self.base_path, scene_name)
            elif datasetname == 'UAVVIC':
                self.imgs_path = UAVVIC_ImgPath(self.base_path, scene_name)
            else:
                raise NotImplementedError

        self.main_transforms = main_transform
        self.img_transforms = img_transform
        self.length = len(self.imgs_path)

        self.interval = interval if interval < self.length else self.length // 2
        self.skip_flag = skip_flag
        self.valid = self.is_valid()

    def is_valid(self):
        if self.skip_flag:
            valid = torch.zeros((self.length))
            loop_idx_range = self.length - self.interval -1
            for i in range(self.length - self.interval):
                if i % self.interval == 0:
                    valid[i] = 1
                    if i + self.interval > loop_idx_range:
                        valid[i + self.interval] = 1
                elif i == loop_idx_range:
                    valid[i] = 1
                    valid[i + self.interval] = 1
        else:
            valid = torch.ones((self.length))
        
        return valid

    def __len__(self):
        return len(self.imgs_path) - self.interval


    def __getitem__(self, index):
        if self.valid[index]:
            index1 = index
            index2 = index + self.interval
            img1 = Image.open(self.imgs_path[index1])
            img2 = Image.open(self.imgs_path[index2])

            if img1.mode is not 'RGB':
                img1 = img1.convert('RGB')
            if img2.mode is not 'RGB':
                img2 = img2.convert('RGB')
                
            if self.target:
                target1 = deepcopy(self.label[index1])
                target2 = deepcopy(self.label[index2])

                if self.main_transforms is not None:
                    img1, target1 = self.main_transforms(img1, target1)
                    img2, target2 = self.main_transforms(img2, target2)

                if 'person_id' in target1:
                    ids0 = target1['person_id']
                    ids1 = target2['person_id']

                    share_mask0 = torch.isin(ids0, ids1)
                    share_mask1 = torch.isin(ids1, ids0)

                    outflow_mask = torch.logical_not(share_mask0)
                    inflow_mask = torch.logical_not(share_mask1)

                elif 'inflow' in target1:
                    outflow_mask = target1['outflow'].bool()
                    inflow_mask = target2['inflow'].bool()
                    share_mask0 = torch.logical_not(outflow_mask)
                    share_mask1 = torch.logical_not(inflow_mask)

                
                target1['share_mask0'] = share_mask0
                target1['outflow_mask'] = outflow_mask
                target2['share_mask1'] = share_mask1
                target2['inflow_mask'] = inflow_mask
                
                if self.img_transforms is not None:
                    img1 = self.img_transforms(img1)
                    img2 = self.img_transforms(img2)

                return  [img1, img2], [target1, target2]

            if self.img_transforms is not None:
                img1 = self.img_transforms(img1)
                img2 = self.img_transforms(img2)

            return [img1, img2], None
        else:
            return None, None

    def generate_imgPath_label(self, i):

        img_path = []
        root = osp.join(self.base_path, i +'/img1')
        img_ids = os.listdir(root)
        img_ids.sort(key=self.myc)


        for img_id in img_ids:
            img_id = img_id.strip()
            single_path = osp.join(root, img_id)
            img_path.append(single_path)

        return img_path

    def myc(self, string):
        p = re.compile("\d+")
        return int(p.findall(string)[0])