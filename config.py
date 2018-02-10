# Network settings
MAX_TOKEN_LEN = 90
NUM_OF_TOKEN = 115
IMH = 256
IMW = 512

# Training settings
NUM_EPOCH = 500
LR = 0.0001
LR_DECAY = 1/1.5
NUM_EPOCH_TO_DECAY = 10
MOMENTUM = 0.9
MAX_GRAD_CLIP = 0.1
BATCH_SIZE = 12
RAND_TRANSFORM = True

# Saving
NUM_EPOCH_TO_SAVE = 5 # Save every SAVE_STEP epoch
SAVE_NAME = 'attend_gru_021018_weight' # MMDDYY
META_NAME = 'attend_gru_021018_meta'

# For traking/displaying
NUM_ITE_TO_LOG = 10 # Display information (loss, current epoch, etc.) every PRINT_STEP iteration
NUM_ITE_TO_VIS = 250
VIS_PATH = './../vis/Image2Latex/'

CUDA = True

# Path settings
DATASET_PATH = './../data/'
MODEL_FOLDER = './../data/trained/CROHME2013/'

# Subdataset settings
SUBSET_LIST = ['expressmatch/', 'HAMEX/', 'KAIST/', 'MathBrush/', 'MfrDB/']
# These are the approprirate scale factors of equations came from the corresponding sub-datasets (computed using CROHME_parser.size_statistics), they are used to render numpy images from inkml data
SCALE_FACTORS = [1.0, 100, 0.065, 0.04, 0.8] 
