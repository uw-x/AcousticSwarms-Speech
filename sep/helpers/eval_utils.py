import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from .constants import SI_SNR_POWER_THRESHOLD
import librosa
import mir_eval

MIN_ERR = 1e-8

def si_sdr(estimated_signal, reference_signals, scaling=True):
    """
    This is a scale invariant SDR. See https://arxiv.org/pdf/1811.02508.pdf
    or https://github.com/sigsep/bsseval/issues/3 for the motivation and
    explanation

    Input:
        estimated_signal and reference signals are (N,) numpy arrays

    Returns: SI-SDR as scalar
    """
    Rss = np.dot(reference_signals, reference_signals)
    this_s = reference_signals

    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / Rss
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum() + MIN_ERR

    SDR = 10 * math.log10(Sss/Snn)

    return SDR
    
    

def split_wav(wav, top_db = 18):
    MIN_SEG = 1000 #800
    MAX_SEG = 4000
    power_list = librosa.feature.rms(y = wav, frame_length=1024, hop_length=256)
    max_ref = np.amax(power_list)
    split_threshold =  0.04
    if max_ref <split_threshold:
        #print("Max apmlitude of signal is too low and reset the ref max value!")
        intervals = librosa.effects.split(wav, top_db=top_db, ref = split_threshold, frame_length=1024, hop_length=256)
    else:
        intervals = librosa.effects.split(wav, top_db=top_db,  frame_length=1024, hop_length=256)

    finetune_seg = []
    for indexes in intervals:
        interval_len = indexes[1] - indexes[0]
        if interval_len < MIN_SEG:
            continue
        elif interval_len > MAX_SEG :
            num_seg = interval_len//MAX_SEG
            for i in range(num_seg):
                if i >= num_seg - 1:
                    finetune_seg.append([indexes[0] + i*MAX_SEG, indexes[1]])
                else:
                    finetune_seg.append([indexes[0] + i*MAX_SEG, indexes[0] + (i+1)*MAX_SEG ])
        else:
            finetune_seg.append([indexes[0], indexes[1]])

    return finetune_seg


def split_wise_sisdr(estimated_signal, reference_signals, seg_index):
    assert(len(seg_index) > 0)
    sisdr_list = []
    for range_index in seg_index:
        seg1 = estimated_signal[range_index[0]: range_index[1]]
        seg2 = reference_signals[range_index[0]: range_index[1]]
        similarity = si_sdr(seg1, seg2)
        sisdr_list.append(similarity)
    
    return sisdr_list
