import re, os
import argparse
parser = argparse.ArgumentParser(
    description='VIC test and demo',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--MODEL', type=str, default='GD3A',
    help='Used model for test')
parser.add_argument(
    '--DATASET', type=str, default='MovingDroneCrowd',
    help='Used dataset for test')
parser.add_argument(
    '--output_dir', type=str, default='test_results',
    help='Directory where to write test results')
parser.add_argument(
    '--test_name', type=str, default='tracker_evaluation',
    help='Customed name for the test')
parser.add_argument(
    '--test_interval', type=int, default=4,
    help='Frame interval for test')
parser.add_argument(
    '--test_split', type=str, default='test',
    help='Dataset split for test')
parser.add_argument(
    '--test_visual', type=bool, default=False,
    help='Whether to save test visualizations')
parser.add_argument(
    '--skip_flag', type=bool, default=True,
    help='if you need to caculate the MIAE and MOAE, it should be False')
parser.add_argument(
    '--SEED', type=int, default=3035,
    help='Random seed to use')
parser.add_argument(
    '--counter', type=str,
    default='STEERER',
     help='Used pretrained counter')
parser.add_argument(
    '--pre_trained_counter_path', type=str,
    default='',
     help='pretrained weight path of the global counter')
parser.add_argument(
    '--model_path', type=str, default='',
    help='pretrained weight path of the model')
parser.add_argument(
    '--GPU_ID', type=str, default='6')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID
if opt.MODEL != "GD3A":
    raise NotImplementedError(f"Only GD3A is supported, but got {opt.MODEL}")
def extract_ep(path):
    """Extract ep_xxx from model filename."""
    fname = os.path.basename(path)
    m = re.search(r'ep_\d+', fname)
    return m.group(0) if m else 'best_model'
counter_ep = extract_ep(opt.pre_trained_counter_path)
assoc_ep = extract_ep(opt.model_path)
name_parts = [
    opt.test_name,
    opt.test_split,
    str(opt.test_interval),
    opt.counter,
    counter_ep,
    assoc_ep
]

# filter out empty values
name_parts = [p for p in name_parts if p]

final_name = "_".join(name_parts)

# ---- final output dir ----
opt.output_dir = os.path.join(opt.output_dir, opt.DATASET, final_name)

if not os.path.exists(opt.output_dir):
    os.makedirs(opt.output_dir, exist_ok=True)

import time
from tqdm import tqdm
from config import cfg
from copy import deepcopy
import torch
import cusdatasets
from mmcv import Config
from misc.utils import *
import torch.nn.functional as F
from importlib import import_module
from model.VIC import Video_Counter
from model.density_estimator.MyCounter import CustomedCounter
from model.density_estimator.STEERER.build_counter import Baseline_Counter as STEERER
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
class PedestrianTracker:
    def __init__(self, proximity_radius=12, min_votes=1, stride=8.0):
        self.next_id = 0
        self.proximity_radius = proximity_radius # 描述符距离行人中心多远算属于该行人
        self.min_votes = min_votes # 两个行人之间最少有多少个特征匹配才建立联系
        self.stride = stride # 特征图下采样倍数
        self.results = []
        
    def _record(self, frame_idx, tid, pos):
        # 记录时转换回原图坐标
        self.results.append([
            frame_idx, tid, 
            pos[0] - 3, 
            pos[1] - 3,
            6,
            6
        ])
        
    def _get_peaks(self, density_map):
        """提取峰值坐标"""
        if isinstance(density_map, torch.Tensor):
            density_map = density_map.detach().cpu().numpy()
        if density_map.ndim == 3: density_map = np.squeeze(density_map)
        
        # threshold_abs 决定了忽略极小的密度值
        threshold_abs = density_map.max() * 0.1
        peaks = peak_local_max(density_map, min_distance=14, threshold_abs=threshold_abs)
        if len(peaks) > 0:
            return peaks[:, ::-1].astype(np.float32) # (x, y)
        return np.empty((0, 2))

    def initialize(self, global_map, frame_idx=1):
        """根据 I0 全局密度图初始化"""
        peaks = self._get_peaks(global_map)
        self.active_tracks = {}
        for p in peaks:
            self.active_tracks[self.next_id] = p
            self._record(frame_idx, self.next_id, p)
            self.next_id += 1
        print(f"[Init] Frame {frame_idx}: {len(peaks)} people.")
        
    def update(self, G2, matched_results, next_frame_idx):
        """
        处理两帧之间的跟踪
        G2: I2 的全局密度图
        matched_results: 包含 kpts0, kpts1, matches0 (indicesa) 等
        """
        # 1. 检测：提取两帧所有的行人位置
        # 如果是连续处理，I1 的点其实可以在上一轮缓存，这里为了清晰重复提取
        pts_dst = self._get_peaks(G2)

        # 2. 转换描述符坐标到原图尺度
        def to_numpy(x):
            return x.cpu().numpy() if torch.is_tensor(x) else x

        kptsa = to_numpy(matched_results['kpts0'][0]) * self.stride
        kptsb = to_numpy(matched_results['kpts1'][0]) * self.stride
        indicesa = to_numpy(matched_results['matches0'][0])
        indicesb = to_numpy(matched_results['matches1'][0])

        # 3. 初始化 ID 分配
        current_src_ids = list(self.active_tracks.keys())
        pts_src = np.array(list(self.active_tracks.values()))

        if len(pts_dst) == 0:
            self.active_tracks = {}
            return []

        # 4. 描述符归属关联 (Spatial Assignment)
        # 找到 I1 每个描述符属于哪个行人中心
        dist_a2p = cdist(kptsa, pts_src)
        kptsa_owner = np.argmin(dist_a2p, axis=1)
        kptsa_owner[np.min(dist_a2p, axis=1) > self.proximity_radius] = -1

        # 找到 I2 每个描述符属于哪个行人中心
        dist_b2p = cdist(kptsb, pts_dst)
        kptsb_owner = np.argmin(dist_b2p, axis=1)
        kptsb_owner[np.min(dist_b2p, axis=1) > self.proximity_radius] = -1

        # 5. 构建行人级的投票矩阵 (Score Matrix)
        score_matrix = np.zeros((len(pts_src), len(pts_dst)))
        for i, match_idx_in_b in enumerate(indicesa):
            if match_idx_in_b < 0: continue
            
            p_idx_a = kptsa_owner[i]
            p_idx_b = kptsb_owner[match_idx_in_b]
            
            if p_idx_a != -1 and p_idx_b != -1:
                score_matrix[p_idx_a, p_idx_b] += 1

        # 遍历所有 I2 的描述符匹配关系
        for i, match_idx_in_a in enumerate(indicesb):
            # match_idx_in_a 是 I1 中描述符的索引。在某些实现中，不匹配通常设为 -1
            if match_idx_in_a < 0:
                continue
            
            person_idx_b = kptsb_owner[i]
            person_idx_a = kptsa_owner[match_idx_in_a]

            # 只有当匹配的两个描述符分别属于 I1 和 I2 的两个行人中心时，才投票
            if person_idx_a != -1 and person_idx_b != -1:
                score_matrix[person_idx_a, person_idx_b] += 1
                
        # 6. 辅助位置约束 (解决由于旋转、形变导致特征匹配较少的情况)
        dist_matrix = cdist(pts_src, pts_dst)
        # 即使没有特征匹配，如果位置极近，也给一个小权重
        spatial_bias = 1.0 / (1.0 + dist_matrix / 10) 
        
        # 最终代价矩阵（负数因为我们要最大化匹配数）
        cost_matrix = -(score_matrix * 10.0 + spatial_bias)

        # 7. 匈牙利算法求解最优匹配
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # 8. 结果整理与 ID 传递
        new_dst_ids = np.full(len(pts_dst), -1, dtype=int)

        for r, c in zip(row_indices, col_indices):
            # 匹配准则：有特征匹配，或者位置极其接近
            if score_matrix[r, c] >= self.min_votes:
                new_dst_ids[c] = current_src_ids[r]

        # 9. 处理新出现的行人 (Inflow)
        for i in range(len(pts_dst)):
            if new_dst_ids[i] == -1:
                new_dst_ids[i] = self.next_id
                self.next_id += 1

        # 更新缓存用于下一帧
        new_active_tracks = {}
        for i in range(len(pts_dst)):
            tid = new_dst_ids[i]
            new_active_tracks[tid] = pts_dst[i]
            self._record(next_frame_idx, tid, pts_dst[i])
        self.active_tracks = new_active_tracks
        

def module2model(module_state_dict):
    state_dict = {}
    for k, v in module_state_dict.items():
        while k.startswith("module."):
            k = k[7:]
        state_dict[k] = v
    return state_dict

def test(cfg_data):
    TEST_MAX_SHORT = cfg_data.TEST_MAX_SHORT
    TEST_MAX_LONG = cfg_data.TEST_MAX_LONG
    cfg.MODEL = opt.MODEL
    model = Video_Counter(cfg, cfg_data).cuda()
    if opt.counter == 'STEERER':
        counter_config = Config.fromfile("model/density_estimator/STEERER/configs/MDC.py")
        global_counter = STEERER(counter_config.network, 
                                            counter_config.dataset.den_factor, 
                                            counter_config.train.route_size)
    elif opt.counter == 'customed':
        global_counter = CustomedCounter()
    global_counter = global_counter.cuda()
    with open(os.path.join(cfg_data.DATA_PATH, 'scene_labels.txt'), 'r') as f:
        lines = f.readlines()
    scene_label = {}
    for line in lines:
        line = line.rstrip().split(',')
        scene_label.update({line[0]: [i.strip() for i in line[1:]] })

    test_loader, restore_transform = cusdatasets.loading_testset(opt.DATASET, opt.test_interval, opt.skip_flag, mode=opt.test_split)
    global_counter_dict = {}     
    if opt.pre_trained_counter_path:
        global_counter_pretrained_dict = torch.load(opt.pre_trained_counter_path, map_location='cpu')
        for k, v in global_counter_pretrained_dict.items():
            if opt.counter == 'customed':
                if 'Extractor' in k or 'global_decoder' in k:
                    if k.startswith("module."):
                        global_counter_dict[k[7:]] = v 
                    else:
                        global_counter_dict[k] = v 
            else:
                if k.startswith("module."):
                    global_counter_dict[k[7:]] = v 
                else:
                    global_counter_dict[k] = v 

    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(module2model(state_dict), strict=True)
    global_counter.load_state_dict(global_counter_dict, strict=True)

    model.eval()
    global_counter.eval()
    sing_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}
    scenes_pred_dict = {'all':[], 'density0':[],'density1':[],'density2':[], 'density3':[]}
    scenes_gt_dict =  {'all':[], 'density0':[],'density1':[],'density2':[], 'density3':[]}

    if opt.skip_flag:
        intervals = 1
    else:
        intervals = []

    for scene_id, (scene_name, sub_valset) in enumerate(test_loader, 0):
        test_interval = sub_valset.dataset.interval
        if not opt.skip_flag: 
            intervals.append(test_interval)
        gen_tqdm = tqdm(sub_valset)
        video_time = len(sub_valset) + test_interval
        pred_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
        gt_dict = {'id': scene_id, 'time': video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}

        visual_maps = []
        imgs = []
        visual_maps_count = []
        tracker = PedestrianTracker()
        for vi, data in enumerate(gen_tqdm, 0):
            if vi % test_interval == 0 or vi == len(sub_valset) - 1:
                frame_signal = 'match'
            else:
                frame_signal = 'skip'
            
            if frame_signal == 'match' or not opt.skip_flag:
                img, label = data
                for i in range(len(label)):
                    for key, data in label[i].items():
                        if torch.is_tensor(data):
                            label[i][key]=data.cuda()

                img = img.cuda()
                ori_width = label[0]['ori_width']
                ori_height = label[0]['ori_height']
                with torch.no_grad():
                    b, c, h, w = img.shape
                    if h % 32 != 0:
                        pad_h = 32 - h % 32
                    else:
                        pad_h = 0
                    if w % 32 != 0:
                        pad_w = 32 - w % 32
                    else:
                        pad_w = 0
                    pad_dims = (0, pad_w, 0, pad_h)
                    img = F.pad(img, pad_dims, "constant")
                    h, w = img.size(2), img.size(3)
                    place_holder_img = torch.zeros((1, h, w)).cuda()
                
                    start_time = time.time()
                    pre_map, gt_den, pre_share_map, gt_share_den, pre_in_out_map, gt_in_out_den, _, matched_results, _ = model(img, label, global_counter)
                    end_time = time.time()
                    #    -----------Counting performance------------------
                    gt_count, pred_cnt = gt_den[0].sum().item(),  pre_map[0].sum().item()

                    s_mae = abs(gt_count - pred_cnt)
                    s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                    sing_cnt_errors['mae'].update(s_mae)
                    sing_cnt_errors['mse'].update(s_mse)

                    if vi == 0:
                        tracker.initialize(pre_map[0], frame_idx=1)
                        pred_dict['first_frame'] = pred_cnt
                        gt_dict['first_frame'] = gt_count

                    pred_dict['inflow'].append(pre_in_out_map[1].sum().item())
                    pred_dict['outflow'].append(pre_in_out_map[0].sum().item())
                    gt_dict['inflow'].append(gt_in_out_den[1].sum().item())
                    gt_dict['outflow'].append(gt_in_out_den[0].sum().item())
                    tracker.update(pre_map[1], matched_results, next_frame_idx = vi + test_interval + 1)
                    
                    #===================test visual==============================
                    if opt.test_visual:
                        if vi % test_interval == 0:
                            img0 = img[0]
                            gt_den0 = gt_den[0]
                            pre_map0 = pre_map[0]

                            if vi == 0:
                                gt_share_den_before = deepcopy(place_holder_img)
                                pre_share_den_before = deepcopy(place_holder_img)
                                gt_in_den = deepcopy(place_holder_img)
                                pre_in_den = deepcopy(place_holder_img)
                            else:
                                gt_share_den_before = previous_gt_share_den[1]
                                pre_share_den_before = previous_pre_share_map[1]
                                gt_in_den = previous_gt_in_out_den[1]
                                pre_in_den = previous_pre_in_out_map[1]

                            gt_share_den_next = gt_share_den[0]
                            pre_share_den_next = pre_share_map[0]
                            gt_out_den = gt_in_out_den[0]
                            pre_out_den = pre_in_out_map[0]

                            visual_map = torch.stack([gt_den0, pre_map0, gt_share_den_before, pre_share_den_before,
                                                    gt_in_den, pre_in_den, gt_share_den_next, pre_share_den_next, gt_out_den, pre_out_den], dim=0)
                            visual_maps_count.append([gt_den0.sum().item(), pre_map0.sum().item(), gt_share_den_before.sum().item(),
                                                     pre_share_den_before.sum().item(), gt_in_den.sum().item(), pre_in_den.sum().item(),
                                                     gt_share_den_next.sum().item(), pre_share_den_next.sum().item(),
                                                     gt_out_den.sum().item(), pre_out_den.sum().item()])
                            visual_maps.append(visual_map)
                            imgs.append(img0)

                            previous_gt_share_den = gt_share_den
                            previous_pre_share_map = pre_share_map
                            previous_gt_in_out_den = gt_in_out_den
                            previous_pre_in_out_map = pre_in_out_map

                            if (vi + test_interval) > (len(sub_valset) - 1):
                                visual_map = torch.stack([gt_den[1], pre_map[1], gt_share_den[1], pre_share_map[1],
                                                    gt_in_out_den[1], pre_in_out_map[1], deepcopy(place_holder_img), deepcopy(place_holder_img), deepcopy(place_holder_img), deepcopy(place_holder_img)], dim=0)
                                visual_maps_count.append([gt_den[1].sum().item(), pre_map[1].sum().item(), gt_share_den[1].sum().item(),
                                                         pre_share_map[1].sum().item(), gt_in_out_den[1].sum().item(), pre_in_out_map[1].sum().item(),
                                                         0, 0, 0, 0])
                                visual_maps.append(visual_map)
                                imgs.append(img[1])

        if opt.test_visual:
            visual_maps = torch.stack(visual_maps, dim=0)
            save_test_visual(visual_maps, imgs, np.array(visual_maps_count), scene_name, restore_transform, opt.output_dir, 0, 0)
        scenes_pred_dict['all'].append(pred_dict)
        scenes_gt_dict['all'].append(gt_dict)
        if scene_name in scene_label:
            scene_l = scene_label[scene_name]
            if scene_l[0] == '0': scenes_pred_dict['density0'].append(pred_dict);  scenes_gt_dict['density0'].append(gt_dict)
            if scene_l[0] == '1': scenes_pred_dict['density1'].append(pred_dict);  scenes_gt_dict['density1'].append(gt_dict)
            if scene_l[0] == '2': scenes_pred_dict['density2'].append(pred_dict);  scenes_gt_dict['density2'].append(gt_dict)
            if scene_l[0] == '3': scenes_pred_dict['density3'].append(pred_dict);  scenes_gt_dict['density3'].append(gt_dict)
        save_path = os.path.join(opt.output_dir, scene_name + '.txt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            curr_long = max(ori_height, ori_width)
            curr_short = min(ori_height, ori_width)
            scale_long = TEST_MAX_LONG / curr_long
            scale_short = TEST_MAX_SHORT / curr_short
            if scale_long < 1 or scale_short < 1:
                scale = min(scale_long, scale_short)
            else:
                scale = 1
            for r in tracker.results:
                print(f"Frame {int(r[0])} | ID {r[1]} | Box ({r[2] / scale:.1f}, {r[3] / scale:.1f}, {r[4] / scale:.1f}, {r[5] / scale:.1f})")
                line = f"{int(r[0])},{int(r[1])},{r[2] / scale:.1f},{r[3] / scale:.1f},{r[4] / scale:.1f},{r[5] / scale:.1f},1,-1,-1,-1\n"
                f.write(line)
                           
    with open(os.path.join(opt.output_dir, 'results.txt'), 'w') as f:
        for key in scenes_pred_dict.keys():
            s_pred_dict = scenes_pred_dict[key]
            s_gt_dict = scenes_gt_dict[key]
            MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(s_pred_dict, s_gt_dict, intervals)
            if key == 'all':
                save_cnt_result = cnt_result

            print('='*20, key, '='*20)
            print('MAE: %.2f, MSE: %.2f  WRAE: %.2f WIAE: %.2f WOAE: %.2f' % (MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))

            f.write(f"{'='*20}{key}{'='*20}\n")
            f.write(f"MAE: {MAE.item():.2f}, MSE: {MSE.item():.2f}  WRAE: {WRAE.item():.2f} "
                    f"WIAE: {MIAE.item():.2f} WOAE: {MOAE.item():.2f}\n")

        pre_vs_gt_msg = f"Pre vs GT: {save_cnt_result}"
        print(pre_vs_gt_msg)
        f.write(pre_vs_gt_msg + "\n")

        mae = sing_cnt_errors['mae'].avg
        mse = np.sqrt(sing_cnt_errors['mse'].avg)
        final_msg = f"mae: {mae:.2f}, mse: {mse:.2f}"

        print(final_msg)
        f.write(final_msg + "\n")

if __name__=='__main__':
    seed = opt.SEED
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # ------------prepare data loader------------
    data_mode = opt.DATASET
    datasetting = import_module(f'cusdatasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    # ------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    test(cfg_data)

