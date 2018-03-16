# Network settings
MAX_TOKEN_LEN = 90
NUM_OF_TOKEN = 115
IMH = 256
IMW = 512
USE_COORD = True

# Training settings
NUM_EPOCH = 500
LR = 0.0005
LR_DECAY = 1/1.3
NUM_EPOCH_TO_DECAY = 5
MOMENTUM = 0.9
MAX_GRAD_CLIP = 0.1
BATCH_SIZE = 16
RAND_TRANSFORM = True

# Test settings
BEAM_SIZE = 1

# Saving
NUM_EPOCH_TO_SAVE = 5 # Save every SAVE_STEP epoch
SAVE_NAME = 'attend_gru_031518_weight' # MMDDYY
META_NAME = 'attend_gru_031518_meta'
TEST_NAME = 'attend_gru_031518_test'

# For traking/displaying
NUM_ITE_TO_LOG = 10 # Display information (loss, current epoch, etc.) every PRINT_STEP iteration
NUM_ITE_TO_VIS = 50
VIS_PATH = './../vis/Image2Latex/'

CUDA = True

# Path settings
DATASET_PATH = './../data/'
MODEL_FOLDER = './../data/trained/CROHME2013/'

# Subdataset settings
SUBSET_LIST = ['expressmatch/', 'HAMEX/', 'KAIST/', 'MathBrush/', 'MfrDB/']
# These are the approprirate scale factors of equations came from the corresponding sub-datasets (computed using CROHME_parser.size_statistics), they are used to render numpy images from inkml data
SCALE_FACTORS = [1.0, 100, 0.065, 0.04, 0.8] 
TEST_SCALE_FACTOR = 0.5
