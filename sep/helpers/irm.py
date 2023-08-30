import argparse
import json
import multiprocessing.dummy as mp
import os

from pathlib import Path

import numpy as np
import tqdm

from scipy.signal import stft, istft



def do_irm(gt, mix, alpha=1):
    """
    Computes and applies the Ideal Ratio Mask
    gt: (n_voices, 1, t)
    mix: (1, t)
    """
    n_voices = gt.shape[0]
    nfft = 2048
    hop = 1024
    eps = np.finfo(np.float).eps

    N = mix.shape[-1] # number of samples
    X = stft(mix, nperseg=nfft)[2]
    (I, F, T) = X.shape

    # Compute sources spectrograms
    P = []
    for gt_idx in range(n_voices):
        P.append(np.abs(stft(gt[gt_idx], nperseg=nfft)[2]) ** alpha)
        
    # compute model as the sum of spectrograms
    model = eps
    for gt_idx in range(n_voices):
        model += P[gt_idx]

    # perform separation
    estimates = []
    for gt_idx in range(n_voices):
        # Create a ratio Mask
        mask = np.divide(np.abs(P[gt_idx]), model)
        
        # apply mask
        Yj = np.multiply(X, mask)

        target_estimate = istft(Yj)[1][:,:N]

        estimates.append(target_estimate)

    estimates = np.array(estimates) # (S, 1, T)
    
    return estimates

