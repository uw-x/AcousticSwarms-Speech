import argparse

from pathlib import Path

import numpy as np
import tqdm

from scipy.signal import stft, istft


def do_ibm(gt, mix, alpha, theta):
    """
    Computes and applies the Ideal Binary Mask
    gt: (n_voices, 1, t)
    mix: (1, t)
    """
    n_voices = gt.shape[0]
    nfft = 2048
    eps = np.finfo(np.float).eps

    N = mix.shape[-1] # number of samples
    X = stft(mix, nperseg=nfft)[2]
    (I, F, T) = X.shape

    # perform separation
    estimates = []
    for gt_idx in range(n_voices):
        # compute STFT of target source
        Yj = stft(gt[gt_idx], nperseg=nfft)[2]

        # Create binary Mask
        mask = np.divide(np.abs(Yj)**alpha, (eps + np.abs(X) ** alpha))
        mask[np.where(mask >= theta)] = 1
        mask[np.where(mask < theta)] = 0

        Yj = np.multiply(X, mask)
        target_estimate = istft(Yj)[1][:,:N]

        estimates.append(target_estimate)

    estimates = np.array(estimates)

    return estimates