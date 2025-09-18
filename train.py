import os
import torch
import datasets
import numpy as np
from tqdm import tqdm
from misc import tools
from config import cfg
from torch import optim
from misc.utils import *
from copy import deepcopy
import torch.nn.functional as F
from model.points_from_den import *
from torch.nn import SyncBatchNorm
from model.VIC import Video_Counter
from misc.tools import is_main_process
import timm.optim.optim_factory as optim_factory
class Trainer():
    def __init__(self, cfg_data, pwd):
        self.exp_name = cfg.EXP_NAME
        self.exp_path = cfg.EXP_PATH
        self.pwd = pwd
        self.model = self.model_without_ddp = Video_Counter(cfg, cfg_data)
        self.model.cuda()
        self.val_frame_intervals = cfg_data.VAL_FRAME_INTERVALS
        if cfg.distributed:
            sync_model = SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(
                sync_model, device_ids=[cfg.gpu], find_unused_parameters=False)
            self.model_without_ddp = self.model.module
        self.train_loader, self.sampler_train, self.val_loader, self.restore_transform = \
                datasets.loading_data(cfg.DATASET, self.val_frame_intervals, cfg.distributed, is_main_process())
        param_groups = optim_factory.add_weight_decay(self.model_without_ddp, cfg.WEIGHT_DECAY)
        self.optimizer = optim.Adam(param_groups, lr=cfg.LR_Base)
        self.i_tb = 0
        self.epoch = 1
        self.num_iters = cfg.MAX_EPOCH * int(len(self.train_loader))
        self.timer={'iter time': Timer(), 'train time': Timer(), 'val time': Timer()}
        self.train_record = {'best_model_name': '', 'mae': 1e20, 'mse': 1e20, 'seq_MAE':1e20, 'WRAE':1e20, 'MIAE': 1e20, 'MOAE': 1e20, 'share_mae':1e20, 'share_mse':1e20}
        if cfg.RESUME:
            latest_state = torch.load(cfg.RESUME_PATH)
            self.model.load_state_dict(latest_state['net'], strict=True)
            self.optimizer.load_state_dict(latest_state['optimizer'])
            self.epoch = latest_state['epoch']
            self.i_tb = latest_state['i_tb']
            self.train_record = latest_state['train_record']
            self.exp_path = latest_state['exp_path']
            self.exp_name = latest_state['exp_name']
            print("Finish loading resume mode")
        if cfg.PRE_TRAIN_COUNTER:
            counting_pre_train = torch.load(cfg.PRE_TRAIN_COUNTER)
            model_dict = self.model.state_dict()
            new_dict = {}
            for k, v in counting_pre_train.items():
                if 'Extractor' in k or 'global_decoder' in k:
                    if 'module' in k:
                        if cfg.distributed:
                            new_dict[k] = v
                        else:
                            new_dict[k[7:]] = v 
                    else:
                        if cfg.distributed:
                            new_dict['module.' + k] = v
                        else:
                            new_dict[k] = v 
            model_dict.update(new_dict)
            self.model.load_state_dict(model_dict, strict=True)
        if is_main_process():
            self.writer, self.log_txt = logger(self.exp_path, self.exp_name, self.pwd, ['exp','eval','figure','img', 'vis','output', 'exp_backup', 'pre_train_model', 'visual_results'], resume=cfg.RESUME)

    def forward(self):
        for epoch in range(self.epoch, cfg.MAX_EPOCH + 1):
            self.epoch = epoch
            if cfg.distributed:
                self.sampler_train.set_epoch(epoch)
            self.timer['train time'].tic()
            self.train()
            self.timer['train time'].toc(average=False)
            print( 'train time: {:.2f}s'.format(self.timer['train time'].diff) )
            print( '='*20 )
            if epoch % cfg.VAL_INTERVAL == 0 and epoch >= cfg.START_VAL:
                if is_main_process():
                    self.timer['val time'].tic()
                    self.validate()
                    self.timer['val time'].toc(average=False)
                    print('val time: {:.2f}s'.format(self.timer['val time'].diff))
                if cfg.distributed:
                    torch.distributed.barrier()

    def train(self): # training for all datasets
        self.model.train()
        lr = adjust_learning_rate(self.optimizer, cfg.LR_Base, self.num_iters, self.i_tb)
        batch_loss = {}
        for i, data in enumerate(self.train_loader, 0):
            self.timer['iter time'].tic()
            self.i_tb += 1
            img, target = data
            for i in range(len(target)):
                for key, data in target[i].items():
                    if torch.is_tensor(data):
                        target[i][key]=data.cuda()
            img = img.cuda()

            pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, loss_dict = self.model(img, target)
            pre_global_cnt = pre_global_den.sum()
            gt_global_cnt = gt_global_den.sum()

            all_loss = 0
            for v in loss_dict.values():
                all_loss += v
            self.optimizer.zero_grad()
            all_loss.backward()
            self.optimizer.step()

            loss_dict_reduced = reduce_dict(loss_dict)

            for k, v in loss_dict_reduced.items():
                if not k in batch_loss:
                    batch_loss[k] = AverageMeter()
                batch_loss[k].update(v.item())

            if (self.i_tb) % cfg.PRINT_FREQ == 0:
                if is_main_process():
                    self.writer.add_scalar('lr', lr, self.i_tb)
                    for k, v in loss_dict_reduced.items():
                        self.writer.add_scalar(k, v.item(), self.i_tb)
                    self.timer['iter time'].toc(average=False)

                    loss_str = ''.join([f"[loss_{key} {value.avg:.4f}]" for key, value in batch_loss.items()])
                    print(f"[ep {self.epoch}][it {self.i_tb}]{loss_str}[{self.timer['iter time'].diff:.2f}s]")
                    
                    print('[cnt: gt: %.1f pred: %.1f max_pre: %.1f max_gt: %.1f]  ' %
                            (gt_global_cnt.item(), pre_global_cnt.item(), pre_global_den.max().item()*cfg_data.DEN_FACTOR, gt_global_den.max().item()*cfg_data.DEN_FACTOR))
                    
                    
            if (self.i_tb) % 100 == 0:
                save_visual_results([img, gt_global_den, pre_global_den, gt_share_den, pre_share_den, 
                                    gt_in_out_den, pre_in_out_den], self.restore_transform, 
                                    os.path.join(self.exp_path, self.exp_name, "training_visual"), 
                                    self.i_tb,
                                    int(os.environ['RANK']) if 'RANK' in os.environ else 0)

    def validate(self):
        self.model.eval()
        global_cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter()}
        scenes_pred_dict = []
        scenes_gt_dict = []
        for scene_id, (scene_name, sub_valset) in enumerate(self.val_loader, 0):
            gen_tqdm = tqdm(sub_valset)
            video_time = len(sub_valset) + self.val_frame_intervals
            pred_dict = {'id': scene_id, 'time':video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            gt_dict  = {'id': scene_id, 'time':video_time, 'first_frame': 0, 'inflow': [], 'outflow': []}
            visual_maps = []
            imgs = []
            for vi, data in enumerate(gen_tqdm, 0):
                if vi % self.val_frame_intervals == 0 or vi == len(sub_valset)-1:
                    frame_signal = 'match'
                else: 
                    frame_signal = 'skip'
                
                if frame_signal == 'match':
                    img, label = data
                    for i in range(len(label)):
                        for key, data in label[i].items():
                            if torch.is_tensor(data):
                                label[i][key]=data.cuda()

                    img = img.cuda()
                    with torch.no_grad():
                        b, c, h, w = img.shape 
                        if h % 32 != 0: pad_h = 32 - h % 32
                        else: pad_h = 0
                        if w % 32 != 0: pad_w = 32 - w % 32
                        else: pad_w = 0
                        pad_dims = (0, pad_w, 0, pad_h)
                        img = F.pad(img, pad_dims, "constant")
                        h, w = img.size(2), img.size(3)
                        place_holder_img = torch.zeros((1, h, w)).cuda()

                        if cfg.distributed:
                            pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, loss_dict = self.model.module(img, label)
                        else:
                            pre_global_den, gt_global_den, pre_share_den, gt_share_den, pre_in_out_den, gt_in_out_den, loss_dict = self.model(img, label)
                        pre_in_out_den[pre_in_out_den < 0] = 0
                        #    -----------Counting performance------------------
                        gt_global_cnt, pre_global_cnt = gt_global_den[0].sum().item(),  pre_global_den[0].sum().item()

                        s_mae = abs(gt_global_cnt - pre_global_cnt)
                        s_mse = ((gt_global_cnt - pre_global_cnt) * (gt_global_cnt - pre_global_cnt))
                        global_cnt_errors['mae'].update(s_mae)
                        global_cnt_errors['mse'].update(s_mse)

                        if vi == 0:
                            pred_dict['first_frame'] = pre_global_cnt
                            gt_dict['first_frame'] = gt_global_cnt

                        pred_dict['inflow'].append(pre_in_out_den[1].sum().item())
                        pred_dict['outflow'].append(pre_in_out_den[0].sum().item())
                        gt_dict['inflow'].append(gt_in_out_den[1].sum().item())
                        gt_dict['outflow'].append(gt_in_out_den[0].sum().item())

                        if vi % self.val_frame_intervals == 0:
                            img0 = img[0]
                            gt_global_den0 = gt_global_den[0]
                            pre_global_den0 = pre_global_den[0]

                            if vi == 0:
                                gt_share_den_before = deepcopy(place_holder_img)
                                pre_share_den_before = deepcopy(place_holder_img)
                                gt_in_den = deepcopy(place_holder_img)
                                pre_in_den = deepcopy(place_holder_img)
                            else:
                                gt_share_den_before = previous_gt_share_den[1]
                                pre_share_den_before = previous_pre_share_den[1]
                                gt_in_den = previous_gt_in_out_den[1]
                                pre_in_den = previous_pre_in_out_den[1]

                            gt_share_den_next = gt_share_den[0]
                            pre_share_den_next = pre_share_den[0]
                            gt_out_den = gt_in_out_den[0]
                            pre_out_den = pre_in_out_den[0]

                            visual_map = torch.stack([gt_global_den0, pre_global_den0, gt_share_den_before, pre_share_den_before,
                                                    gt_in_den, pre_in_den, gt_share_den_next, pre_share_den_next, gt_out_den, pre_out_den], dim=0)
                            visual_maps.append(visual_map)
                            imgs.append(img0)

                            previous_gt_share_den = gt_share_den
                            previous_pre_share_den = pre_share_den
                            previous_gt_in_out_den = gt_in_out_den
                            previous_pre_in_out_den = pre_in_out_den

                            if (vi + self.val_frame_intervals) > (len(sub_valset) - 1):
                                visual_map = torch.stack([gt_global_den[1], pre_global_den[1], gt_share_den[1], pre_share_den[1],
                                                    gt_in_out_den[1], pre_in_out_den[1], deepcopy(place_holder_img), deepcopy(place_holder_img), deepcopy(place_holder_img), deepcopy(place_holder_img)], dim=0)
                                visual_maps.append(visual_map)
                                imgs.append(img[1])
                        
            visual_maps = torch.stack(visual_maps, dim=0)
            save_test_visual(visual_maps, imgs, scene_name, self.restore_transform, 
                             os.path.join(self.exp_path, self.exp_name, "val_visual", scene_name), 
                             self.epoch, int(os.environ['RANK']) if cfg.distributed else 0)
            scenes_pred_dict.append(pred_dict)
            scenes_gt_dict.append(gt_dict)

        MAE, MSE, WRAE, MIAE, MOAE, cnt_result = compute_metrics_all_scenes(scenes_pred_dict, scenes_gt_dict, 1)

        print('MAE: %.2f, MSE: %.2f  WRAE: %.2f WIAE: %.2f WOAE: %.2f' % (MAE.data, MSE.data, WRAE.data, MIAE.data, MOAE.data))
        print('Pre vs GT:', cnt_result)
        mae = global_cnt_errors['mae'].avg
        mse = np.sqrt(global_cnt_errors['mse'].avg)

        self.train_record = update_model(self, {'mae':mae, 'mse':mse, 'seq_MAE':MAE, 'WRAE':WRAE, 'MIAE': MIAE, 'MOAE': MOAE })

        print_NWPU_summary_det(self, {'mae':mae, 'mse':mse, 'seq_MAE':MAE, 'WRAE':WRAE, 'MIAE': MIAE, 'MOAE': MOAE})
        torch.cuda.empty_cache()
        
if __name__=='__main__':
    import os
    import numpy as np
    import torch
    import datasets
    from config import cfg
    from importlib import import_module
    # ------------prepare enviroment------------
    tools.init_distributed_mode(cfg)
    tools.set_randomseed(cfg.SEED + tools.get_rank())

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # ------------prepare data loader------------
    data_mode = cfg.DATASET
    datasetting = import_module(f'datasets.setting.{data_mode}')
    cfg_data = datasetting.cfg_data

    # ------------Start Training------------
    pwd = os.path.split(os.path.realpath(__file__))[0]
    cc_trainer = Trainer(cfg_data, pwd)
    cc_trainer.forward()

