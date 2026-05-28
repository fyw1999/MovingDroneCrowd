import re, os
import argparse
parser = argparse.ArgumentParser(
    description='VIC test and demo',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--MODEL', type=str, default='GD3A',
    help='Used model for test')
parser.add_argument(
    '--DATASET', type=str, default='SENSE',
    help='Used dataset for test')
parser.add_argument(
    '--output_dir', type=str, default='test_results',
    help='Directory where to write test results')
parser.add_argument(
    '--test_name', type=str, default='',
    help='Customed name for the test')
parser.add_argument(
    '--test_interval', type=int, default=15,
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
    default='customed',
     help='Used pretrained counter')
parser.add_argument(
    '--pre_trained_counter_path', type=str,
    default='',
     help='pretrained weight path of the global counter')
parser.add_argument(
    '--model_path', type=str, default='',
    help='pretrained weight path of the association model')
parser.add_argument(
    '--GPU_ID', type=str, default='6')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.GPU_ID
def extract_ep(path):
    """Extract ep_xxx from model filename."""
    fname = os.path.basename(path)
    m = re.search(r'ep_\d+', fname)
    return m.group(0) if m else 'best_model'
assoc_ep = extract_ep(opt.model_path)
if opt.MODEL == "GD3A":
    counter_ep = extract_ep(opt.pre_trained_counter_path)
    name_parts = [
        opt.test_name,
        opt.test_split,
        str(opt.test_interval),
        counter_ep,
        assoc_ep
    ]
elif opt.MODEL == "SDNet":
    name_parts = [
    opt.test_name,
    opt.test_split,
    str(opt.test_interval),
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
def module2model(module_state_dict):
    state_dict = {}
    for k, v in module_state_dict.items():
        while k.startswith("module."):
            k = k[7:]
        state_dict[k] = v
    return state_dict

def test(cfg_data):
    cfg.MODEL = opt.MODEL
    model = Video_Counter(cfg, cfg_data).cuda()
    with open(os.path.join(cfg_data.DATA_PATH, 'scene_label.txt'), 'r') as f:
        lines = f.readlines()
    scene_label = {}
    for line in lines:
        line = line.rstrip().split(' ')
        scene_label.update({line[0]: [i.strip() for i in line[1:]] })

    test_loader, restore_transform = cusdatasets.loading_testset(opt.DATASET, opt.test_interval, opt.skip_flag, mode=opt.test_split)
    
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(module2model(state_dict), strict=True)
    model.eval()
    
    if opt.MODEL == "GD3A":
        if opt.counter == 'STEERER':
            counter_config = Config.fromfile("model/density_estimator/STEERER/configs/MDC.py")
            global_counter = STEERER(counter_config.network, 
                                                counter_config.dataset.den_factor, 
                                                counter_config.train.route_size)
        elif opt.counter == 'customed':
            global_counter = CustomedCounter()
        global_counter = global_counter.cuda()
    
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

        global_counter.load_state_dict(global_counter_dict, strict=True)
        global_counter.eval()
        
    sing_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}
    scenes_pred_dict = {'all':[], 'density0':[],'density1':[],'density2':[], 'density3':[], 'density4':[]}
    scenes_gt_dict =  {'all':[], 'density0':[],'density1':[],'density2':[], 'density3':[], 'density4':[]}

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
                    pre_map, gt_den, pre_share_map, gt_share_den, pre_in_out_map, gt_in_out_den, *extra = model(img, label, global_counter)
                    if opt.MODEL == "SDNet":
                        pre_in_out_map[pre_in_out_map < 0] = 0
                    end_time = time.time()
                    #    -----------Counting performance------------------
                    gt_count, pred_cnt = gt_den[0].sum().item(),  pre_map[0].sum().item()

                    s_mae = abs(gt_count - pred_cnt)
                    s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
                    sing_cnt_errors['mae'].update(s_mae)
                    sing_cnt_errors['mse'].update(s_mse)

                    if vi == 0:
                        pred_dict['first_frame'] = pred_cnt
                        gt_dict['first_frame'] = gt_count

                    pred_dict['inflow'].append(pre_in_out_map[1].sum().item())
                    pred_dict['outflow'].append(pre_in_out_map[0].sum().item())
                    gt_dict['inflow'].append(gt_in_out_den[1].sum().item())
                    gt_dict['outflow'].append(gt_in_out_den[0].sum().item())
                    
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
            save_test_visual(visual_maps, imgs, scene_name, restore_transform, opt.output_dir, 0, 0, np.array(visual_maps_count))
        scenes_pred_dict['all'].append(pred_dict)
        scenes_gt_dict['all'].append(gt_dict)

        if scene_name in scene_label:
            scene_l = scene_label[scene_name]
            if scene_l[3] == '0': scenes_pred_dict['density0'].append(pred_dict);  scenes_gt_dict['density0'].append(gt_dict)
            if scene_l[3] == '1': scenes_pred_dict['density1'].append(pred_dict);  scenes_gt_dict['density1'].append(gt_dict)
            if scene_l[3] == '2': scenes_pred_dict['density2'].append(pred_dict);  scenes_gt_dict['density2'].append(gt_dict)
            if scene_l[3] == '3': scenes_pred_dict['density3'].append(pred_dict);  scenes_gt_dict['density3'].append(gt_dict)
            if scene_l[3] == '4': scenes_pred_dict['density4'].append(pred_dict);  scenes_gt_dict['density4'].append(gt_dict)  
             
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
