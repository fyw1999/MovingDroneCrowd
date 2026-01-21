import os
from easydict import EasyDict as edict
import time
import os
# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
__C.SEED = 3035  # random seed,  for reproduction
__C.DATASET = 'MovingDroneCrowd'       # dataset selection:  HT21, SENSE
__C.NAME = ''
__C.encoder = "VGG16_FPN"
__C.RESUME = False # continue training
__C.RESUME_PATH = ''
__C.PRE_TRAIN_COUNTER = ''
__C.GPU_ID = '0,1,2,3'
os.environ["CUDA_VISIBLE_DEVICES"] = __C.GPU_ID

__C.cross_attn_embed_dim = 256
__C.cross_attn_num_heads = 4
__C.mlp_ratio = 4
__C.cross_attn_depth = 2

__C.FEATURE_DIM = 256
# learning rate settings
__C.LR_Base = 1e-5  # learning rate
__C.WEIGHT_DECAY = 1e-6
# when training epoch is more than it, the learning rate will be begin to decay

__C.MAX_EPOCH = 120
__C.VAL_INTERVAL = 10
__C.START_VAL = 20
__C.PRINT_FREQ = 20
# print
now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
    + '_' + __C.DATASET \
    + '_' + str(__C.LR_Base) \
    + '_' + __C.NAME

__C.VAL_VIS_PATH = './exp/'+__C.DATASET+'_val'
__C.EXP_PATH = os.path.join('./exp', __C.DATASET)  # the path of logs, checkpoints, and current codes
if not os.path.exists(__C.EXP_PATH ):
    os.makedirs(__C.EXP_PATH )