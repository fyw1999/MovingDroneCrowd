from easydict import EasyDict as edict


# init
__C_UAVVIC = edict()

cfg_data = __C_UAVVIC

__C_UAVVIC.TRAIN_SIZE = (720,1280) # (848,1536) #
__C_UAVVIC.TRAINING_MAX_LONG = 2560
__C_UAVVIC.TRAINING_MAX_SHORT = 1440
__C_UAVVIC.TEST_MAX_LONG = 1920
__C_UAVVIC.TEST_MAX_SHORT = 1080
__C_UAVVIC.DATA_PATH = '/nvme0/yaowu/dataset/counting/uavvic/'
__C_UAVVIC.TRAIN_LST = 'train.txt'
__C_UAVVIC.VAL_LST =  'val.txt'
__C_UAVVIC.TEST_LST =  'test.txt'

__C_UAVVIC.MEAN_STD = (
    [117/255., 110/255., 105/255.], [67.10/255., 65.45/255., 66.23/255.]
)

__C_UAVVIC.DEN_FACTOR = 200.

__C_UAVVIC.RESUME_MODEL = ''#model path
__C_UAVVIC.TRAIN_BATCH_SIZE = 2 #  img pairs
__C_UAVVIC.TRAIN_FRAME_INTERVALS=(2,6)  # 2s-5s
__C_UAVVIC.VAL_FRAME_INTERVALS = 5
__C_UAVVIC.VAL_BATCH_SIZE = 1 # must be 1


