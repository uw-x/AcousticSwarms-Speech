import os
import json
import torch
import sep.helpers.utils as utils
import mir_eval
import numpy as np
from asteroid.metrics import get_metrics


def get_items(curr_dir, denoise_gt=False):
    """
    This is a modified version of the SpatialAudioDataset DataLoader
    """

    # Load metadata
    with open(os.path.join(curr_dir, 'metadata.json'), 'rb') as json_file:
        metadata = json.load(json_file)

    # Load multichannel mixture
    mics = [key for key in metadata.keys() if 'mic' in key]
    mixture = []
    for i in range(len(mics)):
        mixture.append(utils.read_audio_file_torch(os.path.join(curr_dir, f'{mics[i]}_mixed.wav')))
    mixture = torch.vstack(mixture)

    if len(mixture.shape) < 2:
        mixture.unsqueeze(0)

    # Load ground truth audio for each speaker
    voices = [key for key in metadata.keys() if 'voice' in key]
    target_voice_data = []
    if denoise_gt:
        for voice in voices:
            denoise_file = os.path.join(curr_dir, f'{mics[0]}_{voice}_denoised.wav')
            if os.path.exists(denoise_file):
                target_voice_data.append(utils.read_audio_file_torch(denoise_file))
            else:
                target_voice_data.append(utils.read_audio_file_torch(os.path.join(curr_dir, f'{mics[0]}_{voice}.wav')))
    else:
        for voice in voices:
            target_voice_data.append(utils.read_audio_file_torch(os.path.join(curr_dir, f'{mics[0]}_{voice}.wav')))
    target_voice_data = torch.vstack(target_voice_data)

    return metadata, mixture, target_voice_data

def compute_metrics(input_signal: np.ndarray, est_signal: np.ndarray, gt: np.ndarray, permute=False):
    # Compute SDR using mir_eval
    # Input mixture is the same for all ground truth signals (same reference microphone), so no
    # need to permute
    input_sdr, _, _, _ = mir_eval.separation.bss_eval_sources(gt, input_signal, compute_permutation=False) 
    output_sdr, sir, sar, perm = mir_eval.separation.bss_eval_sources(gt, est_signal, compute_permutation=permute)
    output_sdr = output_sdr[perm]

    # Compute SI-SDR
    metrics_dict = get_metrics(mix=input_signal[0],
                               clean=gt,
                               estimate=est_signal,
                               metrics_list=['si_sdr'],
                               compute_permutation=permute,
                               sample_rate=48000, # sr shouldn't matter since we're only computing SI-SDR
                               average=False)

    # Store results in list
    input_sisdr = []
    output_sisdr = []
    for i in range(gt.shape[0]):
        input_sisdr.append(metrics_dict['input_si_sdr'][i][0])
        output_sisdr.append(metrics_dict['si_sdr'][i])

    return input_sdr, output_sdr, input_sisdr, output_sisdr


        
