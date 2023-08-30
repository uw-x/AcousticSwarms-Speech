"""
A collection of constants that probably should not be changed
"""
import numpy as np

# Universal Constants
SPEED_OF_SOUND = 343.0  # m/s
FS = 48000

# Project constants
MAX_SHIFTS = [2, 4]
ROOM_DIM = 6
MAX_SPEAKER_RELATIVE_HEIGHT = 0.8
NEG_SAMPLE_INTIAL_CANDIDATES = 30

# Robot constants
CHANNELS_PER_MIC = 1
CODEC_FRAME_DURATION_S = 0.02


### constants for pipeline
#SPR_PHAT params
INIT_WIDTH = 8
bin0 = 2
bin1 = 200
freq_bins = np.arange(bin0, bin1)
n_fft = 2048
threshold = 0.02


## localize params
MIN_AREA = 400
MIN_WIDTH = 3 #2
MIN_TOLERANCE = 4 
MAX_BIG_PATCH = 30
MIN_WIDTH_REQUIRED = 2

LOC_MODEL_THRESHOLD = 0.6
USE_RELATIVE_SPOT_POWER = False
SPOT_POWER_THRESHOLD1 = 0.008#0.008 # 0.01
SPOT_POWER_THRESHOLD2 = 0.01#0.01 #0.012
SI_SNR_POWER_THRESHOLD = 4e-3

# Energy threshold so avoid choosing silence when generating hardware data
WINDOWED_RMS_POWER_THRESHOLD = 0.001 

MAX_NUM = 25 #30
