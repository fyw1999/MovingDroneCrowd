from easydict import EasyDict as edict


# init
__C_SENSE = edict()

cfg_data = __C_SENSE

__C_SENSE.TRAIN_SIZE = (768,1024) # (848,1536) #
__C_SENSE.TRAINING_MAX_LONG = 2560
__C_SENSE.TRAINING_MAX_SHORT = 1440
__C_SENSE.TEST_MAX_LONG = 1920
__C_SENSE.TEST_MAX_SHORT = 1080
__C_SENSE.DATA_PATH = '/data1/fyw/datasets/counting/VSCrowd/'
__C_SENSE.TRAIN_LST = 'filtered_filtered_train.txt'
__C_SENSE.VAL_LST =  'filtered_filtered_val.txt'
__C_SENSE.TEST_LST =  'filtered_filtered_test.txt'

__C_SENSE.MEAN_STD = (
    [117/255., 110/255., 105/255.], [67.10/255., 65.45/255., 66.23/255.]
)

__C_SENSE.DEN_FACTOR = 200.

__C_SENSE.RESUME_MODEL = ''#model path
__C_SENSE.TRAIN_BATCH_SIZE = 1 #  img pairs
__C_SENSE.TRAIN_FRAME_INTERVALS=(13,18)  # 2s-5s
__C_SENSE.VAL_FRAME_INTERVALS = 14
__C_SENSE.VAL_BATCH_SIZE = 1 # must be 1


