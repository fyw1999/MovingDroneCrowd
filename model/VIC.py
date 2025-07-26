import torch
import numpy as np
import torch.nn as nn
from functools import partial
from .gvt import pcvit_base, PosCNN
from model.points_from_den import *
from misc.layer import Gaussianlayer
from model.VGG.ResNet50_FPN import resnet50
from model.ViT.models_crossvit import CrossAttentionBlock, FeatureFusionModule
from model.VGG.VGG16_FPN import VGG16_FPN_Encoder
from model.ResNet.ResNet50_FPN import ResNet_50_FPN_Encoder
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

class Video_Counter(nn.Module):
    def __init__(self, cfg, cfg_data):
        super(Video_Counter, self).__init__()
        self.cfg = cfg
        self.cfg_data = cfg_data
        if cfg.encoder == 'VGG16_FPN':
            self.Extractor = VGG16_FPN_Encoder()
        elif cfg.encoder == 'PCPVT':
            self.Extractor = pcvit_base()
        elif cfg.encoder == 'ResNet_50_FPN':
            self.Extractor = ResNet_50_FPN_Encoder()
        else:
            raise  Exception("The backbone is out of setting, Please chose HR_Net or VGG16_FPN")

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # self.cross_pos_block = PosCNN(self.cfg.FEATURE_DIM, self.cfg.FEATURE_DIM)
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
        
    def forward(self, img, target):
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
    
    def test_forward(self, img):
        features = self.Extractor(img)
        B, C, H, W = features[-1].shape
        pre_global_den = self.global_decoder(features[-1])
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

        pre_share_den = self.share_decoder(share_features)
        mid_pre_in_out_den = pre_global_den - pre_share_den
        pre_in_out_den = self.in_out_decoder(mid_pre_in_out_den)

        pre_global_den = pre_global_den.detach() / self.cfg_data.DEN_FACTOR
        pre_share_den = pre_share_den.detach() / self.cfg_data.DEN_FACTOR
        pre_in_out_den = pre_in_out_den.detach() / self.cfg_data.DEN_FACTOR

        return pre_global_den, pre_share_den, pre_in_out_den
    
    
    # def con_loss(self, labels, features, pair_idx, share_mask0):
    #     count_in_pair=[labels[pair_idx * 2]['points'].size(0), labels[pair_idx * 2+1]['points'].size(0)]
    #     if (np.array(count_in_pair) > 0).all() and torch.sum(share_mask0) > 2:
    #         match_gt, pois = self.get_ROI_and_MatchInfo(labels[pair_idx * 2], labels[pair_idx * 2 + 1], noise='ab')
    #         poi_features = prroi_pool2d(features[pair_idx*2: pair_idx*2+2], pois, 1, 1, self.feature_scale)
    #         poi_features = poi_features.squeeze(2).squeeze(2)[None].transpose(1,2) # [batch, dim, num_features]
    #         mdesc0, mdesc1 = torch.split(poi_features, count_in_pair,dim=2)
    #         sim_matrix = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)  #inner product (n,m)
    #         # mdesc0(n,256) mdesc1(m,256) frame1 n peoples frames 2 m peoples
    #         m0 = torch.norm(mdesc0,dim = 1) #l2norm
    #         m1 = torch.norm(mdesc1,dim = 1)
    #         norm = torch.einsum('bn,bm->bnm',m0,m1) + 1e-7 # (n,m)
    #         exp_term = torch.exp(sim_matrix / (256 ** .5 )/ norm)[0]
    #         try:
    #             topk = torch.topk(exp_term[match_gt['a2b'][:,0]],50,dim = 1).values #(c,b) # # of negative 
    #         except:
    #             topk = exp_term[match_gt['a2b'][:,0]]
    #         denominator = torch.sum(topk,dim=1)   # denominator
    #         numerator = exp_term[match_gt['a2b'][:,0], match_gt['a2b'][:,1]]   # numerator 
    #         loss =  torch.sum(-torch.log(numerator / denominator +1e-7))
    #         return loss.sum()
    #     else:
    #         return torch.tensor(0., device=features[0].device)

    