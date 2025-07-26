import os
import cv2
import pdb
import math
import time
import torch
import random
import shutil
import numpy as np
from torch import nn
from PIL import Image
import torch.distributed as dist
import  torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.transforms as standard_transforms

# def adjust_learning_rate(optimizer, epoch, base_lr1=0, power=0.9):
#     lr1 =  base_lr1 * power ** ((epoch - 1))
#     optimizer.param_groups[0]['lr'] = lr1
#     optimizer.param_groups[1]['lr'] = lr2
#     return lr1 , lr2
def adjust_learning_rate(optimizer, base_lr, max_iters, cur_iters, power=0.9):
    lr = base_lr * ((1 - float(cur_iters) / max_iters) ** (power))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def weights_normal_init(*models):
    for model in models:
        dev=0.01
        if isinstance(model, list):
            for m in model:
                weights_normal_init(m, dev)
        else:
            for m in model.modules():            
                if isinstance(m, nn.Conv2d):        
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)


def logger(exp_path, exp_name, work_dir, exception, resume=False):

    from tensorboardX import SummaryWriter
    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    writer = SummaryWriter(exp_path+ '/' + exp_name)
    log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    
    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_file, 'a') as f:
        f.write(''.join(cfg_lines) + '\n\n\n\n')

    if not resume:
        copy_cur_env(work_dir, exp_path+ '/' + exp_name + '/code', exception)

    return writer, log_file


def logger_txt(log_file, epoch, scores):
    snapshot_name = 'ep_%d' % epoch
    for key, data in scores.items():
        snapshot_name+= ('_'+ key+'_%3f'%data)
    with open(log_file, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')
        f.write(snapshot_name + '\n')
        f.write('[')
        for key, data in scores.items():
            f.write(' '+ key+' %.2f'% data)
        f.write('\n')
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')


def save_results_more(iter, exp_path, restore, img, pred_map, gt_map, binar_map,threshold_matrix,Instance_weights, boxes=None):  # , flow):

    pil_to_tensor = standard_transforms.ToTensor()

    UNIT_H , UNIT_W = img.size(2), img.size(3)
    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map, binar_map, threshold_matrix,Instance_weights)):
        if idx > 1:  # show only one group
            break
        pil_input = restore(tensor[0])
        pred_color_map = cv2.applyColorMap((255 * tensor[1] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map = cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        binar_color_map = cv2.applyColorMap((255 * tensor[3] / (tensor[4].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_matched_color_map = cv2.applyColorMap((255 * tensor[4]/ (tensor[4].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        weights_color_map = cv2.applyColorMap((255 * tensor[5] / (tensor[5].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

        point_color = (0, 255, 0)  # BGR
        thickness = 1
        lineType = 4
        pil_input = np.array(pil_input)

        # for i, box in enumerate(boxes, 0):
        #     wh_LeftTop = (box[0], box[1])
        #     wh_RightBottom = (box[2], box[3])
        #     cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
        #     cv2.rectangle(pil_input, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)

        pil_input = Image.fromarray(pil_input)
        pil_label = Image.fromarray(cv2.cvtColor(gt_color_map, cv2.COLOR_BGR2RGB))
        pil_output = Image.fromarray(cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))
        pil_binar = Image.fromarray(cv2.cvtColor(binar_color_map, cv2.COLOR_BGR2RGB))
        pil_gt_matched = Image.fromarray(cv2.cvtColor(gt_matched_color_map, cv2.COLOR_BGR2RGB))
        pil_weights = Image.fromarray(cv2.cvtColor(weights_color_map, cv2.COLOR_BGR2RGB))

        imgs = [pil_input, pil_label, pil_output, pil_binar, pil_gt_matched,pil_weights]

        w_num , h_num=3, 2

        target_shape = (w_num * (UNIT_W + 10), h_num * (UNIT_H + 10))
        target = Image.new('RGB', target_shape)
        count = 0
        for img in imgs:
            x, y = int(count%w_num) * (UNIT_W + 10), int(count // w_num) * (UNIT_H + 10)  # 左上角坐标，从左到右递增
            target.paste(img, (x, y, x + UNIT_W, y + UNIT_H))
            count+=1

        target.save(os.path.join(exp_path,'{}_den.jpg'.format(iter)))
        # cv2.imwrite('./exp/{}_vis_.png'.format(iter), img)


def vis_results_more(exp_name, epoch, writer, restore, img, pred_map, gt_map, binar_map,threshold_matrix, pred_boxes, gt_boxes):

    pil_to_tensor = standard_transforms.ToTensor()

    x = []
    y = []

    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map, binar_map, threshold_matrix)):
        if idx > 1:  # show only one group
            break

        pil_input = restore(tensor[0])
        pred_color_map = cv2.applyColorMap((255 * tensor[1] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map = cv2.applyColorMap((255 * tensor[2] / (tensor[2].max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        binar_color_map = cv2.applyColorMap((255 * tensor[3] ).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        threshold_color_map = cv2.applyColorMap((255 * tensor[4] / (tensor[2].max()  + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

        point_color = (0, 255, 0)  # BGR
        thickness = 1
        lineType = 4
        pil_input = np.array(pil_input)

        for i, box in enumerate(pred_boxes, 0):
            wh_LeftTop = (box[0], box[1])
            wh_RightBottom = (box[2], box[3])
            # print(wh_LeftTop, wh_RightBottom)
            cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
            cv2.rectangle(pil_input, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
        point_color = (255, 0, 0)  # BGR

        for i, box in enumerate(gt_boxes, 0):
            wh_LeftTop = (box[0], box[1])
            wh_RightBottom = (box[2], box[3])
            cv2.rectangle(pil_input, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)

        pil_input = Image.fromarray(pil_input)
        pil_label = Image.fromarray(cv2.cvtColor(gt_color_map, cv2.COLOR_BGR2RGB))
        pil_output = Image.fromarray(cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))
        pil_binar = Image.fromarray(cv2.cvtColor(binar_color_map, cv2.COLOR_BGR2RGB))

        pil_threshold = Image.fromarray(cv2.cvtColor(threshold_color_map, cv2.COLOR_BGR2RGB))


        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_to_tensor(pil_label.convert('RGB')),
                  pil_to_tensor(pil_output.convert('RGB')), pil_to_tensor(pil_binar.convert('RGB')),
                  pil_to_tensor(pil_threshold.convert('RGB'))])

    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy() * 255).astype(np.uint8)

    writer.add_image(exp_name + '_epoch_' + str(epoch + 1), x)

def vis_results(exp_name, epoch, writer, restore, img, pred_map, gt_map,binar_map,boxes):#, flow):

    pil_to_tensor = standard_transforms.ToTensor()

    x = []
    y = []
    
    for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map,binar_map)):
        if idx>1:# show only one group
            break

        pil_input = restore(tensor[0])
        pred_color_map = cv2.applyColorMap((255*tensor[1]/(tensor[2].max()+1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        gt_color_map = cv2.applyColorMap((255*tensor[2]/(tensor[2].max()+1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
        binar_color_map = cv2.applyColorMap((255*tensor[3]/(tensor[2].max()+1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)

        point_color = (0, 255, 0)  # BGR
        thickness = 1
        lineType = 4
        pil_input = np.array(pil_input)
        # print(pil_input, binar_color_map)
        for i, box in enumerate(boxes, 0):
            wh_LeftTop = (box[0], box[1])
            wh_RightBottom = (box[0] + box[2], box[1] + box[3])
            # print(wh_LeftTop, wh_RightBottom)
            cv2.rectangle(binar_color_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
            cv2.rectangle(pil_input,       wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)

        pil_input =Image.fromarray(pil_input)
        pil_label = Image.fromarray(cv2.cvtColor(gt_color_map,cv2.COLOR_BGR2RGB))
        pil_output = Image.fromarray(cv2.cvtColor(pred_color_map,cv2.COLOR_BGR2RGB))
        pil_binar = Image.fromarray(cv2.cvtColor(binar_color_map, cv2.COLOR_BGR2RGB))
        x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_to_tensor(pil_label.convert('RGB')),
                  pil_to_tensor(pil_output.convert('RGB')),pil_to_tensor(pil_binar.convert('RGB'))])
        # pdb.set_trace()  sum(sum(flow[0].cpu().data.numpy().transpose((1,2,0))[:,:,0]))
        # flow = flow[0].cpu().data.numpy().transpose((1,2,0))
        # flow0 = cv2.applyColorMap((255*flow[:,:,0]/(flow[:,:,0].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
        # flow1 = cv2.applyColorMap((255*flow[:,:,1]/(flow[:,:,1].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
        # flow2 = cv2.applyColorMap((255*flow[:,:,2]/(flow[:,:,2].max()+1e-10)).astype(np.uint8).squeeze(),cv2.COLORMAP_JET)
        # flow0 = Image.fromarray(cv2.cvtColor(flow0,cv2.COLOR_BGR2RGB))
        # flow1 = Image.fromarray(cv2.cvtColor(flow1,cv2.COLOR_BGR2RGB))
        # flow2 = Image.fromarray(cv2.cvtColor(flow2,cv2.COLOR_BGR2RGB))
        # y.extend([pil_to_tensor(flow0.convert('RGB')), pil_to_tensor(flow1.convert('RGB')), pil_to_tensor(flow2.convert('RGB'))])


    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=4, padding=5)
    x = (x.numpy()*255).astype(np.uint8)

    # y = torch.stack(y,0)
    # y = vutils.make_grid(y,nrow=3,padding=5)
    # y = (y.numpy()*255).astype(np.uint8)

    # x = np.concatenate((x,y),axis=1)
    writer.add_image(exp_name + '_epoch_' + str(epoch+1), x)


def print_NWPU_summary(trainer, scores):
    f1m_l, ap_l, ar_l, mae, mse, nae, loss = scores
    train_record = trainer.train_record
    with open(trainer.log_txt, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n')
        f.write(str(trainer.epoch) + '\n\n')

        f.write('  [F1 %.4f Pre %.4f Rec %.4f ] [mae %.4f mse %.4f nae %.4f] [val loss %.4f]\n\n' % (f1m_l, ap_l, ar_l,mae, mse, nae,loss))

        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

    print( '='*50 )
    print( trainer.exp_name )
    print( '    '+ '-'*20 )
    print( '  [F1 %.4f Pre %.4f Rec %.4f] [mae %.2f mse %.2f], [val loss %.4f]'\
            % (f1m_l, ap_l, ar_l, mae, mse, loss) )
    print( '    '+ '-'*20 )
    print( '[best] [model: %s] , [F1 %.4f Pre %.4f Rec %.4f] [mae %.2f], [mse %.2f], [nae %.4f]' % (train_record['best_model_name'], \
                                                        train_record['best_F1'], \
                                                        train_record['best_Pre'], \
                                                        train_record['best_Rec'],\
                                                        train_record['best_mae'],\
                                                        train_record['best_mse'],\
                                                        train_record['best_nae']) )
    print( '='*50 )  

def print_NWPU_summary_det(trainer, scores):
    train_record = trainer.train_record
    with open(trainer.log_txt, 'a') as f:
        f.write('='*15 + '+'*15 + '='*15 + '\n')
        f.write(str(trainer.epoch) + '\n\n')
        f.write('  [')
        for key, data in scores.items():
            f.write(' ' +key+  ' %.3f'% data)
        f.write('\n\n')
        f.write('='*15 + '+'*15 + '='*15 + '\n\n')

    print( '='*50 )
    print( trainer.exp_name )
    print( '    '+ '-'*20 )
    content = '  ['
    for key, data in scores.items():
        if isinstance(data,str):
            content +=(' ' + key + ' %s' % data)
        else:
            content += (' ' + key + ' %.3f' % data)
    content += ']'
    print( content)
    print( '    '+ '-'*20 )
    best_str = '[best]'
    for key, data in train_record.items():
        best_str += (' [' + key +' %s'% data + ']')
    print( best_str)
    print( '='*50 )

def update_model(trainer, scores):
    train_record = trainer.train_record
    log_file = trainer.log_txt
    epoch = trainer.epoch
    snapshot_name = 'ep_%d_iter_%d'% (epoch, trainer.i_tb)
    for key, data in scores.items():
        snapshot_name += ('_'+ key+'_%.3f'%data)
    # snapshot_name = 'ep_%d_F1_%.3f_Pre_%.3f_Rec_%.3f_mae_%.1f_mse_%.1f' % (epoch + 1, F1, Pre, Rec, mae, mse)

    for key, data in  scores.items():
        print(key,data)
        if data < train_record[key] :
            train_record['best_model_name'] = snapshot_name
            if log_file is not None:
                logger_txt(log_file, epoch, scores)
            to_saved_weight = trainer.model.state_dict()

            torch.save(to_saved_weight, os.path.join(trainer.exp_path, trainer.exp_name, snapshot_name + '.pth'))

        if data < train_record[key]:
            train_record[key] = data
    latest_state = {'train_record':train_record, 'net':trainer.model.state_dict(), 'optimizer':trainer.optimizer.state_dict(),
                    'epoch': trainer.epoch, 'i_tb':trainer.i_tb,\
                    'exp_path':trainer.exp_path, 'exp_name':trainer.exp_name}
    torch.save(latest_state, os.path.join(trainer.exp_path, trainer.exp_name, 'latest_state.pth'))

    return train_record


def copy_cur_env(work_dir, dst_dir, exception):

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for filename in os.listdir(work_dir):

        file = os.path.join(work_dir,filename)
        dst_file = os.path.join(dst_dir,filename)

        if os.path.isdir(file) and filename not in exception:
            shutil.copytree(file, dst_file)
        elif os.path.isfile(file):
            shutil.copyfile(file,dst_file)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val
        self.count += 1
        self.avg = self.sum / self.count


class AverageCategoryMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self,num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.cur_val = np.zeros(self.num_class)
        self.sum = np.zeros(self.num_class)


    def update(self, cur_val):
        self.cur_val = cur_val
        self.sum += cur_val


# class AverageCategoryMeter(object):
#     """Computes and stores the average and current value"""
#
#     def __init__(self,num_class):
#         self.num_class = num_class
#         self.reset()
#
#     def reset(self):
#         self.cur_val = np.zeros(self.num_class)
#         self.avg = np.zeros(self.num_class)
#         self.sum = np.zeros(self.num_class)
#         self.count = np.zeros(self.num_class)
#
#     def update(self, cur_val, class_id):
#         self.cur_val[class_id] = cur_val
#         self.sum[class_id] += cur_val
#         self.count[class_id] += 1
#         self.avg[class_id] = self.sum[class_id] / self.count[class_id]


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def vis_results_img(img, pred, restore):
    # pdb.set_trace()
    img = img.cpu()
    pred = pred.cpu().numpy()
    pil_input = restore(img)
    pred_color_map = cv2.applyColorMap(
        (255*pred / (pred.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
    pil_output = Image.fromarray(
        cv2.cvtColor(pred_color_map, cv2.COLOR_BGR2RGB))
    x = []
    pil_to_tensor = standard_transforms.ToTensor()
    x.extend([pil_to_tensor(pil_input.convert('RGB')),
              pil_to_tensor(pil_output.convert('RGB'))])
    x = torch.stack(x, 0)
    x = vutils.make_grid(x, nrow=3, padding=5)
    x = (x.numpy() * 255).astype(np.uint8)

    # pdb.set_trace()
    return x

def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text = [], restore_transform=None,
                            id0=None,id1=None
                            ):

    image0 = np.array(restore_transform(image0))
    image1 = np.array(restore_transform(image1))
    image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    H0, W0, C = image0.shape
    H1, W1, C = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W, C), np.uint8)
    out[:H0, :W0,:] = image0
    out[:H1, W0+margin:,:] = image1
    # out = np.stack([out]*3, -1)
    # import pdb
    # pdb.set_trace()
    out_by_point = out.copy()
    point_r_value = 15
    thickness = 3
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (0, 0, 255)
    blue = (255, 0, 0)
    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        for x, y in kpts0:
            cv2.circle(out, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 3, white, -1, lineType=cv2.LINE_AA)

            cv2.circle(out_by_point, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)

        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), point_r_value, red, thickness,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 3, white, -1, lineType=cv2.LINE_AA)

            cv2.circle(out_by_point, (x + margin + W0, y), point_r_value, blue, thickness,
                       lineType=cv2.LINE_AA)

        if id0 is not  None:
            for i, (id, centroid) in enumerate(zip(id0, kpts0)):
                cv2.putText(out, str(id), (centroid[0],centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if id1 is not None:
            for i, (id, centroid) in enumerate(zip(id1, kpts1)):
                cv2.putText(out, str(id), (centroid[0]+margin+W0, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]

    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), point_r_value, green, thickness,
                   lineType=cv2.LINE_AA)

        cv2.circle(out_by_point, (x0, y0), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out_by_point, (x1 + margin + W0, y1), point_r_value, green, thickness,
                   lineType=cv2.LINE_AA)

    # Ht = int(H*30 / 480)  # text height
    # txt_color_fg = (255, 255, 255)
    # txt_color_bg = (0, 0, 0)
    # for i, t in enumerate(text):
    #     cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
    #                 H*1.0/480, txt_color_bg, 2, cv2.LINE_AA)
    #     cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
    #                 H*1.0/480, txt_color_fg, 1, cv2.LINE_AA)
    #     cv2.putText(out_by_point, t, (10, Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
    #             H * 1.0 / 480, txt_color_fg, 1, cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(str(path), out)
        cv2.imwrite(str('point_'+path), out_by_point)
    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out,out_by_point

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

def change2map(intput_map):
    intput_map = intput_map.squeeze(0)
    vis_map = (intput_map - intput_map.min()) / (intput_map.max() - intput_map.min() + 1e-5)
    vis_map = (vis_map * 255).astype(np.uint8)
    vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
    return vis_map

def save_visual_results(data, restor_transform, save_base, iter_num, rank):
    assert (len(data)-1) % 2 == 0
    num = (len(data)-1) // 2
    h = data[0].size(2)
    w = data[0].size(3)
    
    batch_size = data[0].size(0)
    margin = 5

    W = w * len(data) + margin * num + 3 * margin * num
    H = h * batch_size + margin * (batch_size - 1)

    out = np.zeros((H, W, 3))
    
    start_h = 0
    
    for i in range(batch_size):
        start_w = 0
        img = cv2.cvtColor(np.array(restor_transform(data[0][i])), cv2.COLOR_RGB2BGR)
        out[start_h:start_h + h, start_w:start_w + w] = img
        start_w += w + 3 * margin  

        for j in range(num):
            data_map = data[1 + j*2][i].detach().cpu().numpy()
            vis_data_map = change2map(data_map.copy())
            out[start_h:start_h + h, start_w:start_w + w] = vis_data_map
            start_w += w + margin

            data_map = data[1 + j*2 + 1][i].detach().cpu().numpy()
            vis_data_map = change2map(data_map.copy())
            out[start_h:start_h + h, start_w:start_w + w] = vis_data_map
            start_w += w + 3 * margin  
        
        start_h += h + margin
    if not os.path.exists(save_base):
        os.makedirs(save_base, exist_ok=True)
    cv2.imwrite(os.path.join(save_base, "{}_{}_visual.jpg".format(rank, iter_num)), out)

def save_test_visual(visual_maps, imgs, scene_name, restor_transform, save_path, iter, rank):
    visual_data = [visual_maps[:, i, :, :] for i in range(visual_maps.shape[1])]
    visual_data = [torch.stack(imgs, dim=0)] + visual_data
    save_visual_results(visual_data, restor_transform, os.path.join(save_path, scene_name), iter, rank)
    
def change2map(intput_map):
    intput_map = intput_map.squeeze(0)
    vis_map = (intput_map - intput_map.min()) / (intput_map.max() - intput_map.min() + 1e-5)
    vis_map = (vis_map * 255).astype(np.uint8)
    vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
    return vis_map

def compute_metrics_single_scene(pre_dict, gt_dict, intervals):
    pair_cnt = len(pre_dict['inflow'])
    inflow_cnt, outflow_cnt = torch.zeros(pair_cnt, 2), torch.zeros(pair_cnt, 2)
    pre_crowdflow_cnt  = pre_dict['first_frame']
    gt_crowdflow_cnt =  gt_dict['first_frame']
    for idx, data in enumerate(zip(pre_dict['inflow'],  pre_dict['outflow'],\
                                   gt_dict['inflow'], gt_dict['outflow']),0):
        inflow_cnt[idx, 0] = data[0]
        inflow_cnt[idx, 1] = data[2]
        outflow_cnt[idx, 0] = data[1]
        outflow_cnt[idx, 1] = data[3]

        if idx %intervals == 0 or  idx== len(pre_dict['inflow'])-1:
            pre_crowdflow_cnt += data[0]
            gt_crowdflow_cnt += data[2]

    return pre_crowdflow_cnt, gt_crowdflow_cnt,  inflow_cnt, outflow_cnt

def compute_metrics_all_scenes(scenes_pred_dict, scene_gt_dict, intervals):
    scene_cnt = len(scenes_pred_dict)
    metrics = {'MAE':torch.zeros(scene_cnt,2), 'WRAE':torch.zeros(scene_cnt,2), 'MIAE':torch.zeros(0), 'MOAE':torch.zeros(0)}
    for i,(pre_dict, gt_dict) in enumerate( zip(scenes_pred_dict, scene_gt_dict),0):
        time = pre_dict['time']

        pre_crowdflow_cnt, gt_crowdflow_cnt, inflow_cnt, outflow_cnt=\
            compute_metrics_single_scene(pre_dict, gt_dict, intervals)
        mae = np.abs(pre_crowdflow_cnt - gt_crowdflow_cnt)
        metrics['MAE'][i, :] = torch.tensor([pre_crowdflow_cnt, gt_crowdflow_cnt])
        metrics['WRAE'][i,:] = torch.tensor([mae/(gt_crowdflow_cnt+1e-10), time])

        metrics['MIAE'] =  torch.cat([metrics['MIAE'], torch.abs(inflow_cnt[:,0]-inflow_cnt[:,1])])
        metrics['MOAE'] = torch.cat([metrics['MOAE'], torch.abs(outflow_cnt[:, 0] - outflow_cnt[:, 1])])

    MAE = torch.mean(torch.abs(metrics['MAE'][:, 0] - metrics['MAE'][:, 1]))
    MSE = torch.mean((metrics['MAE'][:, 0] - metrics['MAE'][:, 1]) ** 2).sqrt()
    WRAE = torch.sum(metrics['WRAE'][:,0]*(metrics['WRAE'][:,1]/(metrics['WRAE'][:,1].sum()+1e-10)))*100
    MIAE = torch.mean(metrics['MIAE'] )
    MOAE = torch.mean(metrics['MOAE'])

    return MAE,MSE, WRAE,MIAE,MOAE,metrics['MAE']

def local_maximum_points(sub_pre, gaussian_maximun, radius=8.):
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
