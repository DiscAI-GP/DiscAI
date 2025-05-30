IMG_HEIGHT_R = 512 
IMG_WIDTH_R = 512
IMG_HEIGHT = 256 
IMG_WIDTH = 256

IMG_CHANNELS_MRI = 1
IMG_CHANNELS_MASK = 1
MASK_TARGET_LOWER = 201
MASK_TARGET_UPPER = 207

SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 16 

TIMESTEPS = 1000  
BETA_START = 0.0001
BETA_END = 0.02

LEARNING_RATE = 1e-4 
EPOCHS = 50        

MODEL_BASE_CHANNELS = 64 
MODEL_NUM_DOWN_BLOCKS = 4 
MODEL_TIME_EMB_DIM = 128 

print("Configuration loaded.")