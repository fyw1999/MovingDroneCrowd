from easydict import EasyDict as edict


# init
__C_MDC = edict()

cfg_data = __C_MDC

__C_MDC.TRAIN_SIZE = (768, 1024) # (848,1536) #
__C_MDC.TRAINING_MAX_LONG = 2560
__C_MDC.TRAINING_MAX_SHORT = 1440
__C_MDC.TEST_MAX_LONG = 1920
__C_MDC.TEST_MAX_SHORT = 1080
__C_MDC.DATA_PATH = '/data1/fyw/datasets/counting/MovingDroneCrowd/'
__C_MDC.TRAIN_LST = 'train.txt'
__C_MDC.VAL_LST =  'val.txt'
__C_MDC.TEST_LST =  'test.txt'

__C_MDC.MEAN_STD = (
    [117/255., 110/255., 105/255.], [67.10/255., 65.45/255., 66.23/255.]
)

__C_MDC.DEN_FACTOR = 200.

__C_MDC.RESUME_MODEL = ''#model path
__C_MDC.TRAIN_BATCH_SIZE = 1 #  img pairs
__C_MDC.TRAIN_FRAME_INTERVALS=(3, 8)  # 2s-5s
__C_MDC.VAL_FRAME_INTERVALS = 4
__C_MDC.VAL_BATCH_SIZE = 1 # must be 1


