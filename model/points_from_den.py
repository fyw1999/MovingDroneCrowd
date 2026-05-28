import torch
import  torch.nn as nn
import  torch.nn.functional as F

class get_ori_MatchInfo(object):
    def __init__(self, train_size, radius=8, down_scale=8):
        self.h = train_size[0]
        self.w = train_size[1]
        self.radius = radius
        self.down_scale = down_scale
        
    def mark_duplicate_points(self, a: torch.Tensor) -> torch.Tensor:
        """
        输入:
            a: (n, 2) 的张量，表示二维点
        输出:
            b: (n,) 的bool张量，True表示该点是重复出现的（非第一次出现）
        """
        assert a.ndim == 2 and a.shape[1] == 2, "输入必须是 (n, 2) 的张量"
        
        # 1️⃣ 先把二维坐标编码成一维唯一标识
        # 这里乘一个足够大的常数K保证不会冲突
        K = int(a[:, 1].max().item()) + 100000  # 防止溢出
        a_code = a[:, 0].long() * K + a[:, 1].long()
        
        # 2️⃣ 排序以便找到重复项
        sorted_code, sorted_idx = torch.sort(a_code)
        is_dup_sorted = torch.cat([
            torch.tensor([False], device=a.device),
            sorted_code[1:] == sorted_code[:-1]
        ])
        
        # 3️⃣ 还原到原始顺序
        b = torch.zeros_like(is_dup_sorted)
        b[sorted_idx] = is_dup_sorted
        
        return ~b
    
    def __call__(self, target_a, target_b):
        gt_pts_a, gt_pts_b = target_a['points'], target_b['points']
        a_ids = target_a['person_id']
        b_ids = target_b['person_id']

        dis = a_ids.unsqueeze(1).expand(-1,len(b_ids)) - b_ids.unsqueeze(0).expand(len(a_ids),-1)
        dis = dis.abs()
        shared_id_idx_a, shared_id_idx_b = torch.where(dis==0)
        shared_ids = a_ids[shared_id_idx_a]

        device = gt_pts_a.device
        H_ds, W_ds = self.h // self.down_scale, self.w // self.down_scale

        gt_ds_pts_a = (gt_pts_a // self.down_scale).long()
        gt_ds_pts_b = (gt_pts_b // self.down_scale).long()

        gt_ds_shared_pts_a = gt_ds_pts_a[shared_id_idx_a]     # (M, 2)
        gt_ds_shared_pts_b = gt_ds_pts_b[shared_id_idx_b]  # (M, 2)

        grid = torch.stack(torch.meshgrid(
            torch.arange(-self.radius, self.radius+1, device=device),
            torch.arange(-self.radius, self.radius+1, device=device),
            indexing='xy'), dim=-1).long()  # (2r+1, 2r+1, 2)
        grid = grid.view(-1, 2)  # (K, 2)
        circle_mask = (grid[:, 0]**2 + grid[:, 1]**2 <= self.radius**2)  # (K,)
        offsets = grid[circle_mask]  # (K', 2)

        K = offsets.shape[0]
        M = gt_ds_shared_pts_a.shape[0]
        offsets = offsets.unsqueeze(0).expand(M, -1, -1)  # (M, K, 2)

        gt_ds_shared_pts_a_exp = gt_ds_shared_pts_a.unsqueeze(1) + offsets  # (M, K, 2)
        gt_ds_shared_pts_b_exp = gt_ds_shared_pts_b.unsqueeze(1) + offsets  # (M, K, 2)
        shared_ids_exp = shared_ids.unsqueeze(1).expand(-1, K)
        
        gt_ds_shared_pts_a_exp = gt_ds_shared_pts_a_exp.view(-1, 2)  # (M*K', 2)
        gt_ds_shared_pts_b_exp = gt_ds_shared_pts_b_exp.view(-1, 2)  # (M*K', 2)
        shared_ids_exp = shared_ids_exp.flatten()  # (M*K',)
        
        valid0 = self.mark_duplicate_points(gt_ds_shared_pts_a_exp)
        valid1 = self.mark_duplicate_points(gt_ds_shared_pts_b_exp)
        valid = valid0 & valid1
        
        shared_ids_exp = shared_ids_exp[valid]
        gt_ds_shared_pts_a_exp = gt_ds_shared_pts_a_exp[valid]  # (?, 2)
        gt_ds_shared_pts_b_exp = gt_ds_shared_pts_b_exp[valid]  # (?, 2)
        
        valid0 = (gt_ds_shared_pts_a_exp[..., 0] >= 0) & (gt_ds_shared_pts_a_exp[..., 0] < W_ds) & \
             (gt_ds_shared_pts_a_exp[..., 1] >= 0) & (gt_ds_shared_pts_a_exp[..., 1] < H_ds)
        valid1 = (gt_ds_shared_pts_b_exp[..., 0] >= 0) & (gt_ds_shared_pts_b_exp[..., 0] < W_ds) & \
                (gt_ds_shared_pts_b_exp[..., 1] >= 0) & (gt_ds_shared_pts_b_exp[..., 1] < H_ds)
        valid = valid0 & valid1

        shared_ids_exp = shared_ids_exp[valid]
        gt_ds_shared_pts_a_exp = gt_ds_shared_pts_a_exp[valid]  # (?, 2)
        gt_ds_shared_pts_b_exp = gt_ds_shared_pts_b_exp[valid]  # (?, 2)

        return  gt_ds_shared_pts_a_exp, gt_ds_shared_pts_b_exp, shared_ids_exp

def local_maximum_points(sub_pre, gaussian_maximun,radius=8.):
    sub_pre = sub_pre.detach()
    _,_,h,w = sub_pre.size()
    kernel = torch.ones(3,3)/9.
    kernel =kernel.unsqueeze(0).unsqueeze(0).cuda()
    weight = nn.Parameter(data=kernel, requires_grad=False)
    sub_pre = F.conv2d(sub_pre, weight, stride=1, padding=1)

    keep = F.max_pool2d(sub_pre, (5, 5), stride=2, padding=2)
    keep = F.interpolate(keep, scale_factor=2)
    keep = (keep == sub_pre).float()
    sub_pre = keep * sub_pre

    sub_pre[sub_pre < 0.25*gaussian_maximun] = 0
    sub_pre[sub_pre > 0] = 1
    count = int(torch.sum(sub_pre).item())

    points = torch.nonzero(sub_pre)[:,[0,1,3,2]].float() # b,c,h,w->b,c,w,h
    rois = torch.zeros((points.size(0), 5)).float().to(sub_pre)
    rois[:, 0] = points[:, 0]
    rois[:, 1] = torch.clamp(points[:, 2] - radius, min=0)
    rois[:, 2] = torch.clamp(points[:, 3] - radius, min=0)
    rois[:, 3] = torch.clamp(points[:, 2] + radius, max=w)
    rois[:, 4] = torch.clamp(points[:, 3] + radius, max=h)

    pre_data = {'num': count, 'points': points, 'rois': rois}
    return pre_data
