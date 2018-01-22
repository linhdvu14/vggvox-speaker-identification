import numpy as np
from pyaudio import paInt16


# Signal processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512

# FILTER_RANGE = (300.,np.nextafter(3000,float('-inf')))  # frequency range to keep

# Recording options for realtime test
TEST_MODE = "batch_offline"
CHUNK = 1024
NUM_CHANNEL = 1
FORMAT = paInt16
ONLINE_RECORD_SEC = 4
ONLINE_CONDITION = "17B - Mac Mic 80% - No ambient noise reduction - Random phrases - 4s"
ONLINE_SPEAKER = "Hieu"


# Model
WEIGHTS_FILE = "data/model_weights/model_0.h5"
COST_METRIC = "cosine"  # euclidean or cosine
INPUT_SHAPE=(NUM_FFT,None,1)
MAX_SEC_ENROLL = 10
MAX_SEC_TEST = 4
BUCKET_STEP_SEC = 1


# Input/Output
ENROLL_WAV_DIR = "data/wav/libri/"
ENROLL_LIST_FILE = "lst/libri/libri_dev-other_batch_enroll_list.csv"
TEST_WAV_DIR = "data/wav/libri/"
TEST_LIST_FILE = "lst/libri/libri_dev-other_batch_test_list.csv"

OFFLINE_RESULT_FILE = "res/results_offline.csv"
OFFLINE_RESULT_WRITE_OPTION = 'w'  # 'a' or 'w'

ONLINE_WAV_FILE = "data/wav/test.wav"
ONLINE_RESULT_FILE = "res/results_online.csv"
OFFLINE_RESULT_WRITE_OPTION = 'a'  # 'a' or 'w'



