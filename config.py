# Network settings
MAX_TOKEN_LEN = 90
NUM_OF_TOKEN = 114
IMH = 256
IMW = 512

# Training settings
NUM_EPOCH = 50
LR = 0.0001
LR_DECAY = 1/1.15
MOMENTUM = 0.9
BATCH_SIZE = 16
SAVE_STEP = 2

CUDA = True

# Path settings
DATASET_PATH = './../data/'
MODEL_FOLDER = './trained/'

# Subdataset settings
SUBSET_LIST = ['expressmatch/', 'HAMEX/', 'KAIST/', 'MathBrush/', 'MfrDB/']
# These are the approprirate scale factors of equations came from the corresponding sub-datasets (computed using CROHME_parser.size_statistics), they are used to render numpy images from inkml data
SCALE_FACTORS = [1.0, 100, 0.065, 0.04, 0.8] 
