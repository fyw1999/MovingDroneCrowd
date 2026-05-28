import math
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
from functools import partial
from .gvt import pcvit_base, PosCNN
import torch.nn.functional as F
from model.points_from_den import *
from misc.layer import Gaussianlayer
from model.VGG.ResNet50_FPN import resnet50
from .optimal_transport_layer import Optimal_Transport_Layer
from model.VGG.VGG16_FPN import VGG16FPN_Stride8, VGG16FPN_Stride16
from model.ViT.models_crossvit import CrossAttentionBlock, FeatureFusionModule
from model.ResNet.ResNet50_FPN import ResNet50FPN_Stride8, ResNet50FPN_Stride16

from model.decoder import ShareDecoder, InOutDecoder, GlobalDecoder

import cv2
import misc.transforms as own_transforms
import torchvision.transforms as standard_transforms
restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*(
    [117/255., 110/255., 105/255.], [67.10/255., 65.45/255., 66.23/255.]
)),
        standard_transforms.ToPILImage()
    ])

visual_counter = 0
def visualize_and_save(features, images, restore_transform, coords1, coords2, scene_name):
    global visual_counter
    def tensor_to_cv2_img(img_tensor):
        img = restore_transform(img_tensor.cpu())
        img = np.array(img)  # PIL Image to numpy
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def feature_to_heatmap(feat):
        feat = feat.cpu().detach().numpy()
        feat = np.mean(feat, axis=0)  # average over channels, shape (h, w)
        feat = cv2.GaussianBlur(feat, (71, 71), 0)
        feat -= feat.min()
        feat /= (feat.max() + 1e-8)
        feat = (feat * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(feat, cv2.COLORMAP_JET)
        return heatmap

    imgs_with_kpts = []
    for i, (img_tensor, coord) in enumerate(zip(images, [coords1, coords2])):
        img = tensor_to_cv2_img(img_tensor)
        for x, y in coord:
            # cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.rectangle(img, (int(x-7), int(y-7)), (int(x+7), int(y+7)), (255, 255, 255), 1)
        imgs_with_kpts.append(img)

    heatmaps = []
    for i in range(2):
        feat = features[i]
        heatmap = feature_to_heatmap(feat)
        heatmaps.append(heatmap)

    results = []
    for img, heat in zip(imgs_with_kpts, heatmaps):
        heat_resized = cv2.resize(heat, (img.shape[1], img.shape[0]))
        blended = cv2.addWeighted(img, 0.6, heat_resized, 0.4, 0)
        results.append(blended)

    divider = np.ones((results[0].shape[0], 10, 3), dtype=np.uint8) * 255
    final_img = np.hstack([results[0], divider, results[1]])

    save_path = 'visual_results' + '/' + scene_name.replace('/', '_') + '_' + str(visual_counter) + '.jpg'
    cv2.imwrite(save_path, final_img)
    print(f"Saved to {save_path}")
    visual_counter += 1

def get_final_MatchInfo(a_pts, ori_matched_a_pts, b_pts, ori_matched_b_pts, shared_ids_exp):
    K = 100000
    a_pts_code = a_pts[:, 0] * K + a_pts[:, 1]
    ori_matched_a_pts_code = ori_matched_a_pts[:, 0] * K + ori_matched_a_pts[:, 1]

    a_pts_code_sorted, a_pts_sort_idx = torch.sort(a_pts_code)
    ori_matched_a_pts_pos = torch.searchsorted(a_pts_code_sorted, ori_matched_a_pts_code)
    ori_matched_a_pts_in_range = (ori_matched_a_pts_pos < len(a_pts_code))

    ori_matched_a_pts_valid = torch.zeros_like(ori_matched_a_pts_code, dtype=torch.bool)
    valid_indices = torch.nonzero(ori_matched_a_pts_in_range, as_tuple=True)[0]
    ori_matched_a_pts_valid[valid_indices] = (
    a_pts_code_sorted[ori_matched_a_pts_pos[valid_indices]] == ori_matched_a_pts_code[valid_indices]
    )

    b_pts_code = b_pts[:, 0] * K + b_pts[:, 1]
    ori_matched_b_pts_code = ori_matched_b_pts[:, 0] * K + ori_matched_b_pts[:, 1]

    b_pts_code_sorted, b_pts_sort_idx = torch.sort(b_pts_code)
    ori_matched_b_pts_pos = torch.searchsorted(b_pts_code_sorted, ori_matched_b_pts_code)
    ori_matched_b_pts_in_range = (ori_matched_b_pts_pos < len(b_pts_code))

    ori_matched_b_pts_valid = torch.zeros_like(ori_matched_b_pts_code, dtype=torch.bool)
    valid_indices = torch.nonzero(ori_matched_b_pts_in_range, as_tuple=True)[0]
    ori_matched_b_pts_valid[valid_indices] = (
    b_pts_code_sorted[ori_matched_b_pts_pos[valid_indices]] == ori_matched_b_pts_code[valid_indices]
    )


    ori_matched_a_b_pts_valid = ori_matched_a_pts_valid & ori_matched_b_pts_valid

    final_matched_a_pts_pos = ori_matched_a_pts_pos[ori_matched_a_b_pts_valid]
    final_matched_b_pts_pos = ori_matched_b_pts_pos[ori_matched_a_b_pts_valid]

    shared_ids_exp = shared_ids_exp[ori_matched_a_b_pts_valid]

    final_matched_a_pts_idx = a_pts_sort_idx[final_matched_a_pts_pos]
    final_matched_b_pts_idx = b_pts_sort_idx[final_matched_b_pts_pos]

    
    mask_a = torch.ones(a_pts.shape[0], dtype=torch.bool, device=a_pts.device)
    mask_a[final_matched_a_pts_idx] = False
    unmatched_a = torch.nonzero(mask_a, as_tuple=True)[0]

    mask_b = torch.ones(b_pts.shape[0], dtype=torch.bool, device=b_pts.device)
    mask_b[final_matched_b_pts_idx] = False
    unmatched_b = torch.nonzero(mask_b, as_tuple=True)[0]

    matched_a2b = torch.stack([final_matched_a_pts_idx, final_matched_b_pts_idx], 1)

    match_gt={'a2b': matched_a2b, 'un_a':unmatched_a, 'un_b':unmatched_b}

    a_ids = torch.full(a_pts.shape[0:1], -1, dtype=torch.int64, device=a_pts.device)
    b_ids = torch.full(b_pts.shape[0:1], -1, dtype=torch.int64, device=b_pts.device)
    a_ids[final_matched_a_pts_idx] = shared_ids_exp
    b_ids[final_matched_b_pts_idx] = shared_ids_exp

    return match_gt, a_ids, b_ids


class Video_Counter(nn.Module):
    def __init__(self, cfg, cfg_data):
        super(Video_Counter, self).__init__()
        self.cfg = cfg
        self.cfg_data = cfg_data
        if cfg.encoder == 'VGG16_FPN':
            if cfg.MODEL == "GD3A":
                self.Extractor = VGG16FPN_Stride8()
            elif cfg.MODEL == "SDNet":
                self.Extractor = VGG16FPN_Stride16()
            else:
                raise  Exception("The model is out of setting, Please chose GD3A or SDNet")
        elif cfg.encoder == 'PCPVT':
            if cfg.MODEL == "SDNet":
                self.Extractor = pcvit_base()
            else:
                raise  Exception("now the backbone is out of setting, Please chose SDNet")
        elif cfg.encoder == 'ResNet_50_FPN':
            if cfg.MODEL == "GD3A":
                self.Extractor = ResNet50FPN_Stride8()
            elif cfg.MODEL == "SDNet":
                self.Extractor = ResNet50FPN_Stride16()
            else:
                raise  Exception("The model is out of setting, Please chose GD3A or SDNet")
        else:
            raise  Exception("The backbone is out of setting")

        if cfg.MODEL == "GD3A":
            self.get_ori_MatchInfo = get_ori_MatchInfo(self.cfg_data.TRAIN_SIZE, self.cfg.ROI_RADIUS, self.cfg.down_scale)
            OT_config = {
                'feature_dim': cfg.FEATURE_DIM,
                'sinkhorn_iterations':cfg.sinkhorn_iterations,
                'matched_threshold': cfg.matched_thre
            }
            self.Matching_Layer = Optimal_Transport_Layer(OT_config)
            self.Gaussian = Gaussianlayer(sigma=[4], kernel_size=25)
        elif cfg.MODEL == "SDNet":
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            self.share_cross_attention = nn.ModuleList([nn.ModuleList([
            CrossAttentionBlock(cfg.cross_attn_embed_dim, cfg.cross_attn_num_heads, cfg.mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for _ in range(cfg.cross_attn_depth)])
            for _ in range(3)])
            
            self.share_cross_attention_norm = norm_layer(cfg.cross_attn_embed_dim)

            self.feature_fuse = FeatureFusionModule(self.cfg.FEATURE_DIM)
            self.global_decoder = GlobalDecoder()
            self.share_decoder = ShareDecoder()
            self.in_out_decoder = InOutDecoder()
            self.criterion = torch.nn.MSELoss()
            self.Gaussian = Gaussianlayer()
            
    def KPI_cal(self, match_gt, scores, a_ids, b_ids):
        self.match_pairs_cnt = match_gt['un_a'].size(0) + match_gt['un_b'].size(0) + match_gt['a2b'].size(0)
        scores[-1, -1] = 0
        max0_idx = scores.max(1).indices  # the points in a that have matched in b, return b's index,
        max1_idx = scores.max(0).indices  # the points in b that have matched in b, return a's index
        if match_gt['a2b'].size(0)>0:
            pred_a = max0_idx[match_gt['a2b'][:, 0]]
            pred_b = max1_idx[match_gt['a2b'][:, 1]]

            pred_a_mask = torch.isin(pred_a, match_gt['a2b'][:, 1])
            pred_b_mask = torch.isin(pred_b, match_gt['a2b'][:, 0])

            pred_mask = pred_a_mask & pred_b_mask
            pred_a = pred_a[pred_mask]
            pred_b = pred_b[pred_mask]
            
            gt_a = match_gt['a2b'][:, 1][pred_mask]
            gt_b = match_gt['a2b'][:, 0][pred_mask]
            
            a_id_exp = torch.cat([a_ids, torch.tensor([-2]).to(a_ids.device)])
            b_id_exp = torch.cat([b_ids, torch.tensor([-3]).to(b_ids.device)])
            
            TP_a = b_id_exp[gt_a] == b_id_exp[pred_a]
            TP_b = a_id_exp[gt_b] == a_id_exp[pred_b]
            
            TP_mask = TP_a & TP_b
            
            TP = TP_mask.sum()  # correct matched person pairs in two framess
        else:
            TP=0
        TN = (max0_idx[match_gt['un_a']] == scores.size(1) - 1).sum() + (max1_idx[match_gt['un_b']] == scores.size(0) - 1).sum()
        FP = (max0_idx[match_gt['un_a']] != scores.size(1) - 1).sum() + (max1_idx[match_gt['un_b']] != scores.size(0) - 1).sum()
        self.correct_pairs_cnt = TP + TN
        return  self.match_pairs_cnt, self.correct_pairs_cnt, TP, FP
    
    def forward(self, img, target, global_counter = None):
        if self.cfg.MODEL == "GD3A":
            return self.GD3A_forward(img, target, global_counter)
        elif self.cfg.MODEL == "SDNet":
            return self.SDNet_forward(img, target)
        else:
            raise  Exception("The model is out of setting, Please chose GD3A or SDNet")
        
    def GD3A_forward(self, img, target, global_counter = None):
        B, C, H, W = img.shape
        img_pair_num = img.size(0) // 2
        assert  img.size(0) % 2 == 0
        features = self.Extractor(img)
        loss_dict = {}
        gt_global_dot_map = torch.zeros((B, 1, H, W), device=img.device)
        for i, data in enumerate(target):
            points = data['points'].long()
            gt_global_dot_map[i, 0, points[:, 1], points[:, 0]] = 1
        gt_global_den = self.Gaussian(gt_global_dot_map)

        if global_counter:
            pre_global_den = global_counter(img)
        else:
            pre_global_den = torch.zeros_like(gt_global_den)

        gt_global_den = F.interpolate(gt_global_den, 
                                      scale_factor=1/self.cfg.down_scale, 
                                      mode='bilinear', 
                                      align_corners=False)*(self.cfg.down_scale**2)
        
        pre_global_den = F.interpolate(pre_global_den, 
                                       scale_factor=1/self.cfg.down_scale, 
                                       mode='bilinear', 
                                       align_corners=False)*(self.cfg.down_scale**2)

        gt_in_out_dot_map = torch.zeros_like(gt_global_dot_map)
        gt_share_dot_map = torch.zeros_like(gt_global_dot_map)

        pre_share_den = torch.zeros_like(gt_global_den)
        pre_in_out_den = torch.zeros_like(gt_global_den)

        if global_counter:
            global_den = pre_global_den
        else:
            global_den = gt_global_den

        filtered_masks = (global_den > self.cfg.filter_thre).float()
        matched_results = {'matches0': [], 'matches1': [],'matching_scores0': [],'matching_scores1': [], 'gt_matched': [],
                           'gt_count_diff': 0,'pre_count_diff': 0, "kpts0": [], "kpts1": []}
        matched_metrics = {}

        self.batch_match_loss = torch.tensor(0.).to(img.device)
        self.batch_hard_loss = torch.tensor(0.).to(img.device)

        match_loss =[]
        match_pairs_cnt=torch.tensor(0.).to(img.device)
        correct_pairs_cnt=torch.tensor(0.).to(img.device)
        TP_cnt = torch.tensor(0.).to(img.device)
        FP_cnt = torch.tensor(0.).to(img.device)

        for pair_idx in range(img_pair_num):
            if self.training:
                gt_ds_shared_pts_a_exp, gt_ds_shared_pts_b_exp, shared_ids_exp = self.get_ori_MatchInfo(target[pair_idx * 2], target[pair_idx * 2+1])
            
            filtered_mask0 = filtered_masks[pair_idx * 2]
            filtered_mask1 = filtered_masks[pair_idx * 2+1]

            filtered_idx0 = filtered_mask0[0].nonzero()
            filtered_idx1 = filtered_mask1[0].nonzero()
            
            match_gt = None
            a_ids = b_ids = None
            if self.training:
                match_gt, a_ids, b_ids = get_final_MatchInfo(filtered_idx0[:, [1, 0]], gt_ds_shared_pts_a_exp, filtered_idx1[:, [1, 0]], gt_ds_shared_pts_b_exp, shared_ids_exp)
            indices0 = -1 * torch.ones(filtered_idx0.size(0), dtype=torch.int64)
            indices1 = -1 * torch.ones(filtered_idx1.size(0), dtype=torch.int64)
            mscores0 = torch.zeros(filtered_idx0.size(0))
            mscores1 = torch.zeros(filtered_idx1.size(0))
            keypoints0 = torch.zeros((filtered_idx0.size(0), 2))
            keypoints1 = torch.zeros((filtered_idx1.size(0), 2))
            count_in_pair=[filtered_idx0.size(0), filtered_idx1.size(0)]
            if ((np.array(count_in_pair) > 0).all()):
                a = features[pair_idx * 2, :, filtered_idx0[:,0], filtered_idx0[:,1]]
                b = features[pair_idx * 2+1, :, filtered_idx1[:,0], filtered_idx1[:,1]]

                keypoints0 = filtered_idx0[:, [1, 0]]
                keypoints1 = filtered_idx1[:, [1, 0]]

                data = {
                    "descriptors0": a.unsqueeze(0),
                    "descriptors1": b.unsqueeze(0),
                    "keypoints0": keypoints0.unsqueeze(0),
                    "keypoints1": keypoints1.unsqueeze(0),
                    "shape": features.shape
                }
                scores, indices0, indices1, mscores0, mscores1 = self.Matching_Layer(data, self.cfg.top_k, match_gt)
                
                if self.training:
                    match_loss_ = self.Matching_Layer.loss
                    match_loss.append(match_loss_)

                    tmp_gt_person, tmp_correct_pairs, TP, FP = self.KPI_cal(match_gt, scores.clone(), a_ids, b_ids)
                    match_pairs_cnt += tmp_gt_person
                    correct_pairs_cnt += tmp_correct_pairs
                    TP_cnt += TP
                    FP_cnt += FP
                
            outflow_idxs = indices0 == -1
            outflow_pts = filtered_idx0[outflow_idxs]

            pre_in_out_den[pair_idx * 2, 0, outflow_pts[:, 0], outflow_pts[:, 1]] = global_den[pair_idx * 2, 0, outflow_pts[:, 0], outflow_pts[:, 1]]

            shared_idxs0 = indices0 != -1
            shared_pts0 = filtered_idx0[shared_idxs0]
            pre_share_den[pair_idx * 2, 0, shared_pts0[:, 0], shared_pts0[:, 1]] = global_den[pair_idx * 2, 0, shared_pts0[:, 0], shared_pts0[:, 1]]

            inflow_idxs = indices1 == -1
            inflow_pts = filtered_idx1[inflow_idxs]
            pre_in_out_den[pair_idx * 2+1, 0, inflow_pts[:, 0], inflow_pts[:, 1]] = global_den[pair_idx * 2+1, 0, inflow_pts[:, 0], inflow_pts[:, 1]]

            shared_idxs1 = indices1 != -1
            shared_pts1 = filtered_idx1[shared_idxs1]
            pre_share_den[pair_idx * 2+1, 0, shared_pts1[:, 0], shared_pts1[:, 1]] = global_den[pair_idx * 2+1, 0, shared_pts1[:, 0], shared_pts1[:, 1]]

            points0 = target[pair_idx * 2]['points']
            points1 = target[pair_idx * 2 + 1]['points']

            share_mask0 = target[pair_idx * 2]['share_mask0']
            outflow_mask = target[pair_idx * 2]['outflow_mask']
            share_mask1 = target[pair_idx * 2 + 1]['share_mask1']
            inflow_mask = target[pair_idx * 2 + 1]['inflow_mask']
            
            share_coords0 = points0[share_mask0].long()
            share_coords1 = points1[share_mask1].long()
            
            gt_share_dot_map[pair_idx * 2, 0, share_coords0[:, 1], share_coords0[:, 0]] = 1
            gt_share_dot_map[pair_idx * 2 + 1, 0, share_coords1[:, 1], share_coords1[:, 0]] = 1

            outflow_coords = points0[outflow_mask].long()
            inflow_coords = points1[inflow_mask].long()

            gt_in_out_dot_map[pair_idx * 2, 0, outflow_coords[:, 1], outflow_coords[:, 0]] = 1
            gt_in_out_dot_map[pair_idx * 2 + 1, 0, inflow_coords[:, 1], inflow_coords[:, 0]] = 1

            matched_results['matches0'].append(indices0)  # use -1 for invalid match
            matched_results['matches1'].append(indices1)  # use -1 for invalid match
            matched_results['kpts0'].append(keypoints0)
            matched_results['kpts1'].append(keypoints1)
            matched_results['matching_scores0'].append(mscores0)
            matched_results['matching_scores1'].append(mscores1)
            if match_gt is not None:
                matched_results['gt_matched'].append(match_gt['a2b']) 
            matched_results['gt_count_diff'] += torch.sum(gt_in_out_dot_map[pair_idx * 2+1]).item()
            matched_results['pre_count_diff'] += torch.sum(pre_in_out_den[pair_idx * 2+1]).item()
        
        if self.training:
            matched_metrics['recall'] = TP_cnt/(torch.cat(matched_results['gt_matched']).size(0) + 1e-6)
            matched_metrics['accuracy'] = correct_pairs_cnt/(match_pairs_cnt + 1e-6)
            matched_metrics['precision'] = TP_cnt/(TP_cnt + FP_cnt + 1e-6)
            matched_metrics['f1'] = (2*matched_metrics['recall'] * matched_metrics['precision'])/(matched_metrics['recall'] + matched_metrics['precision'] + 1e-6)
        
            if len(match_loss)>0:
                self.batch_match_loss =  torch.mean(torch.cat(match_loss))
            loss_dict['match'] = self.batch_match_loss

        gt_share_den = self.Gaussian(gt_share_dot_map)
        gt_in_out_den = self.Gaussian(gt_in_out_dot_map)

        pre_global_den = F.interpolate(pre_global_den, scale_factor=self.cfg.down_scale, mode='bilinear', align_corners=False)/(self.cfg.down_scale**2)
        gt_global_den = F.interpolate(gt_global_den, scale_factor=self.cfg.down_scale, mode='bilinear', align_corners=False)/(self.cfg.down_scale**2)

        
        pre_share_den = F.interpolate(pre_share_den, scale_factor=self.cfg.down_scale, mode='bilinear', align_corners=False)/(self.cfg.down_scale**2)
        pre_in_out_den = F.interpolate(pre_in_out_den, scale_factor=self.cfg.down_scale, mode='bilinear', align_corners=False)/(self.cfg.down_scale**2)
         
        return pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, loss_dict, matched_results, matched_metrics
    
    def SDNet_forward(self, img, target):
        features = self.Extractor(img)
        B, C, H, W = features[-1].shape
        self.feature_scale = H / img.shape[2] 
        pre_global_den = self.global_decoder(features[-1])
        all_loss = {}
        gt_in_out_dot_map = torch.zeros_like(pre_global_den)
        gt_share_dot_map = torch.zeros_like(pre_global_den)
        img_pair_num = img.size(0) // 2
        assert img.size(0) % 2 == 0
        share_features = None
        for l_num in range(len(self.share_cross_attention)):
            share_results = []
            if share_features is not None:
                feature_fused = self.feature_fuse(share_features, features[l_num])

            for pair_idx in range(img_pair_num):
                if share_features is not None:
                    q1 = feature_fused[pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                else:
                    q1 = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                kv = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                for i in range(len(self.share_cross_attention[l_num])):
                    q1 = self.share_cross_attention[l_num][i](q1, kv)
                    # if i == 0:
                    #     q1 = self.cross_pos_block(q1, H, W)
                
                q1 = self.share_cross_attention_norm(q1)

                if share_features is not None:
                    q2 = feature_fused[pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                else:
                    q2 = features[l_num][pair_idx * 2 + 1].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                kv = features[l_num][pair_idx * 2].unsqueeze(0).flatten(2).permute(0, 2, 1).contiguous() 
                for i in range(len(self.share_cross_attention[l_num])):
                    q2 = self.share_cross_attention[l_num][i](q2, kv)
                    # if i == 0:
                    #     q2 = self.cross_pos_block(q2, H, W)
                
                q2 = self.share_cross_attention_norm(q2)

                share_results.append(q1)
                share_results.append(q2)
                
            share_features = torch.cat(share_results, dim=0)
            share_features = share_features.permute(0, 2, 1).reshape(B, C, H, W).contiguous()

        for pair_idx in range(img_pair_num):
            points0 = target[pair_idx * 2]['points']
            points1 = target[pair_idx * 2 + 1]['points']
            
            share_mask0 = target[pair_idx * 2]['share_mask0']
            outflow_mask = target[pair_idx * 2]['outflow_mask']
            share_mask1 = target[pair_idx * 2 + 1]['share_mask1']
            inflow_mask = target[pair_idx * 2 + 1]['inflow_mask']
            
            share_coords0 = points0[share_mask0].long()
            share_coords1 = points1[share_mask1].long()
            
            gt_share_dot_map[pair_idx * 2, 0, share_coords0[:, 1], share_coords0[:, 0]] = 1
            gt_share_dot_map[pair_idx * 2 + 1, 0, share_coords1[:, 1], share_coords1[:, 0]] = 1

            outflow_coords = points0[outflow_mask].long()
            inflow_coords = points1[inflow_mask].long()

            gt_in_out_dot_map[pair_idx * 2, 0, outflow_coords[:, 1], outflow_coords[:, 0]] = 1
            gt_in_out_dot_map[pair_idx * 2 + 1, 0, inflow_coords[:, 1], inflow_coords[:, 0]] = 1


        pre_share_den = self.share_decoder(share_features)
        mid_pre_in_out_den = pre_global_den - pre_share_den
        pre_in_out_den = self.in_out_decoder(mid_pre_in_out_den)

        # ===================== density map loss =============================
        gt_global_dot_map = torch.zeros_like(pre_global_den)
        for i, data in enumerate(target):
            points = data['points'].long()
            gt_global_dot_map[i, 0, points[:, 1], points[:, 0]] = 1
        gt_global_den = self.Gaussian(gt_global_dot_map)

        assert pre_global_den.size() == gt_global_den.size()
        global_mse_loss = self.criterion(pre_global_den, gt_global_den * self.cfg_data.DEN_FACTOR)
        pre_global_den = pre_global_den.detach() / self.cfg_data.DEN_FACTOR
        all_loss['global'] = global_mse_loss


        gt_share_den = self.Gaussian(gt_share_dot_map)
        assert pre_share_den.size() == gt_share_den.size()
        share_mse_loss = self.criterion(pre_share_den, gt_share_den * self.cfg_data.DEN_FACTOR)
        pre_share_den = pre_share_den.detach() / self.cfg_data.DEN_FACTOR
        all_loss['share'] = share_mse_loss*10

        gt_in_out_den = self.Gaussian(gt_in_out_dot_map)
        assert pre_in_out_den.size() == gt_in_out_den.size()
        in_out_mse_loss = self.criterion(pre_in_out_den, gt_in_out_den * self.cfg_data.DEN_FACTOR)
        pre_in_out_den = pre_in_out_den.detach() / self.cfg_data.DEN_FACTOR
        all_loss['in_out'] = in_out_mse_loss

        return pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, all_loss
    
    def test_forward(self, img, global_counter):
        img_pair_num = img.size(0) // 2
        assert  img.size(0) % 2 ==0
        features = self.Extractor(img)
        pre_global_den = global_counter(img)
        
        pre_global_den = F.interpolate(pre_global_den, scale_factor=1/self.cfg.down_scale, mode='bilinear', align_corners=False)*(self.cfg.down_scale**2)

        pre_share_den = torch.zeros_like(pre_global_den)
        pre_in_out_den = torch.zeros_like(pre_global_den)

        filtered_masks = (pre_global_den > self.cfg.filter_thre).float()

        matched_results = {'matches0': [], 'matches1': [],'matching_scores0': [],'matching_scores1': [], 'gt_matched': [],
                           'gt_count_diff': 0,'pre_count_diff': 0}

        for pair_idx in range(img_pair_num):
            filtered_mask0 = filtered_masks[pair_idx * 2]
            filtered_mask1 = filtered_masks[pair_idx * 2+1]

            filtered_idx0 = filtered_mask0[0].nonzero()
            filtered_idx1 = filtered_mask1[0].nonzero()
            indices0 = -1 * torch.ones(filtered_idx0.size(0), dtype=torch.int64)
            indices1 = -1 * torch.ones(filtered_idx1.size(0), dtype=torch.int64)
            mscores0 = torch.zeros(filtered_idx0.size(0))
            mscores1 = torch.zeros(filtered_idx1.size(0))
            count_in_pair=[filtered_idx0.size(0), filtered_idx1.size(0)]
            if ((np.array(count_in_pair) > 0).all()):
                a = features[pair_idx * 2, :, filtered_idx0[:,0], filtered_idx0[:,1]]
                b = features[pair_idx * 2+1, :, filtered_idx1[:,0], filtered_idx1[:,1]]
                
                a_score = pre_global_den[pair_idx * 2, :, filtered_idx0[:,0], filtered_idx0[:,1]]
                b_score = pre_global_den[pair_idx * 2+1, :, filtered_idx1[:,0], filtered_idx1[:,1]]
                
                data = {
                    "descriptors0": a.unsqueeze(0),
                    "descriptors1": b.unsqueeze(0),
                    "keypoints0": filtered_idx0[:, [1, 0]].unsqueeze(0),
                    "keypoints1": filtered_idx1[:, [1, 0]].unsqueeze(0),
                    "scores0": a_score,
                    "scores1": b_score,
                    "shape": features.shape
                }
                scores, indices0, indices1, mscores0, mscores1 = self.Matching_Layer(data, self.cfg.top_k)

            outflow_idxs = indices0 == -1
            outflow_pts = filtered_idx0[outflow_idxs]
            pre_in_out_den[pair_idx * 2, 0, outflow_pts[:, 0], outflow_pts[:, 1]] = pre_global_den[pair_idx * 2, 0, outflow_pts[:, 0], outflow_pts[:, 1]]

            shared_idxs0 = indices0 != -1
            shared_pts0 = filtered_idx0[shared_idxs0]
            pre_share_den[pair_idx * 2, 0, shared_pts0[:, 0], shared_pts0[:, 1]] = pre_global_den[pair_idx * 2, 0, shared_pts0[:, 0], shared_pts0[:, 1]]

            inflow_idxs = indices1 == -1
            inflow_pts = filtered_idx1[inflow_idxs]
            pre_in_out_den[pair_idx * 2+1, 0, inflow_pts[:, 0], inflow_pts[:, 1]] = pre_global_den[pair_idx * 2+1, 0, inflow_pts[:, 0], inflow_pts[:, 1]]

            shared_idxs1 = indices1 != -1
            shared_pts1 = filtered_idx1[shared_idxs1]
            pre_share_den[pair_idx * 2+1, 0, shared_pts1[:, 0], shared_pts1[:, 1]] = pre_global_den[pair_idx * 2+1, 0, shared_pts1[:, 0], shared_pts1[:, 1]]

            matched_results['matches0'].append(indices0)  # use -1 for invalid match
            matched_results['matches1'].append(indices1)  # use -1 for invalid match
            matched_results['matching_scores0'].append(mscores0)
            matched_results['matching_scores1'].append(mscores1)

        
        pre_global_den = F.interpolate(pre_global_den, scale_factor=self.cfg.down_scale, mode='bilinear', align_corners=False)/(self.cfg.down_scale**2)

        pre_share_den = F.interpolate(pre_share_den, scale_factor=self.cfg.down_scale, mode='bilinear', align_corners=False)/(self.cfg.down_scale**2)
        pre_in_out_den = F.interpolate(pre_in_out_den, scale_factor=self.cfg.down_scale, mode='bilinear', align_corners=False)/(self.cfg.down_scale**2)
        return pre_global_den, pre_share_den, pre_in_out_den
