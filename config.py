import os
from easydict import EasyDict as edict
import time
# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 3035  # random seed,  for reproduction
__C.MODEL = "GD3A"  # model selection: GD3A, SDNet
__C.NAME = 'mdc++_vgg16'  # name of the experiment
__C.DATASET = 'MovingDroneCrowd'       # dataset selection:  HT21, SENSE
__C.encoder = "VGG16_FPN"
__C.RESUME = False # continue training
__C.RESUME_PATH = ''
__C.PRE_TRAINED_MODEL = ''
__C.GPU_ID = '3,4,6,7'

#   for GD3A
__C.global_counter = "STEERER"
__C.pre_trained_global_counter = ""
__C.sinkhorn_iterations = 100
__C.down_scale = 8
__C.ROI_RADIUS = 2.
__C.filter_thre = 0.005
__C.matched_thre = 0.1
__C.top_k = 9
__C.LR_Thre = 1e-4

# for SDNet
__C.PRE_TRAIN_COUNTER = ''
__C.cross_attn_embed_dim = 256
__C.cross_attn_num_heads = 4
__C.mlp_ratio = 4
__C.cross_attn_depth = 2

# for both
__C.FEATURE_DIM = 256
__C.LR_Base = 5e-5  # learning rate
__C.WEIGHT_DECAY = 1e-5

__C.MAX_EPOCH = 20
__C.VAL_INTERVAL = 2
__C.START_VAL = 1
__C.PRINT_FREQ = 20
# print
now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
    + '_' + __C.DATASET \
    + '_' + str(__C.LR_Base) \
    + '_' + __C.NAME

__C.EXP_PATH = os.path.join('./exp', __C.DATASET)  # the path of logs, checkpoints, and current codes

os.makedirs(__C.EXP_PATH, exist_ok=True)