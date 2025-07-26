from easydict import EasyDict as edict


# init
__C_HT21 = edict()

cfg_data = __C_HT21

__C_HT21.TRAIN_SIZE =(768,1024)#(768,1024) # (848,1536) #
__C_HT21.TRAINING_MAX_LONG = 2560
__C_HT21.TRAINING_MAX_SHORT = 1440
__C_HT21.TEST_MAX_LONG = 1920
__C_HT21.TEST_MAX_SHORT = 1080
__C_HT21.DATA_PATH = '/data1/fyw/datasets/counting/HT21/'
__C_HT21.TRAIN_LST = 'train.txt'
__C_HT21.VAL_LST =  'val.txt'
__C_HT21.TEST_LST =  'test.txt'

__C_HT21.MEAN_STD = (
    [117/255., 110/255., 105/255.], [67.10/255., 65.45/255., 66.23/255.]
)

__C_HT21.DEN_FACTOR = 200.

__C_HT21.RESUME_MODEL = ''#model path
__C_HT21.TRAIN_BATCH_SIZE = 1 #  img pairs
__C_HT21.TRAIN_FRAME_INTERVALS=(97,103)
__C_HT21.VAL_FRAME_INTERVALS = 100
__C_HT21.VAL_BATCH_SIZE = 1 # must be 1


