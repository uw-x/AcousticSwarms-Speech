"""A collection of useful helper functions"""
import os
import json
from typing import Dict, Tuple, List

import time
import librosa
import soundfile as sf
import random
import numpy as np
import noisereduce as nr

import torch
import torchaudio
import torch.nn.functional as F

from sep.helpers.constants import SPEED_OF_SOUND


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def read_json(path):
    assert os.path.exists(path), f'File not found: {path}'
    with open(path, 'rb') as f:
        return json.load(f)
    
def write_json(dict, path):    
    with open(path, 'w') as f:
        json.dump(dict, f, indent=4)


class Timer(object):
    def __init__(self, device) -> None:
        self.device = str(device)
        
        if self.device == 'cpu':
            pass
        else:
            torch.cuda.synchronize()
            self.t1 = torch.cuda.Event(enable_timing=True)
            self.t2 = torch.cuda.Event(enable_timing=True)
            
    def start_recording(self):
        if self.device == 'cpu':
            self.t1 = time.time()
        else:
            self.t1.record()
    
    def stop_recording(self):
        if self.device == 'cpu':
            self.t2 = time.time()
            dt = self.t2 - self.t1
        else:
            self.t2.record()
            torch.cuda.synchronize()
            dt = self.t1.elapsed_time(self.t2)
            
        return dt

def denoise(signal, noise_sample, sr, stationary=False, n_jobs=1):
    return nr.reduce_noise(y=signal, sr=sr, y_noise=noise_sample, stationary=stationary, n_jobs=n_jobs)

def list_top_level_directories(path) -> List[str]:
    return sorted([a for a in os.listdir(path) if os.path.isdir(os.path.join(path, a))])

def read_metadata(dir_path) -> dict:
    metadata_path = os.path.join(dir_path, 'metadata.json')
    with open(metadata_path, 'r') as json_file:
        metadata = json.load(json_file)
    assert metadata, 'Something went wrong when reading scene metadata. Are you sure this file exists in the specified directory?'
    return metadata

def read_audio_file(file_path, sr):
    """
    Reads audio file to system memory.
    """
    return librosa.core.load(file_path, mono=False, sr=sr)[0]

def read_audio_file_torch(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform

def write_audio_file(file_path, data, sr):
    """
    Writes audio file to system memory.
    @param file_path: Path of the file to write to
    @param data: Audio signal to write (n_channels x n_samples)
    @param sr: Sampling rate
    """
    sf.write(file_path, data.T, sr)

def phase_offset(a, b, sr):
    if len(a.shape) == 1:
        return (np.linalg.norm(b-a, axis=0) * sr/SPEED_OF_SOUND)
    else:
        return (np.linalg.norm(b-a, axis=1) * sr/SPEED_OF_SOUND)

def criterion(m, s, sr):
    """
    Shift audio from channel at m to appear as though it started at s.
    """
    samples = -phase_offset(m, s, sr)
    return samples

def shift_fn(x, shift_samples):
    if isinstance(x, np.ndarray):
        return np.roll(x, shift_samples)
    elif isinstance(x, torch.Tensor):
        return torch.roll(x, shift_samples)
    else:
        return x

def shift_mixture_given_samples(input_data, shifts, inverse=False):
    """
    Shifts the input given  a vector fo sample shifts
    """
    output_data = input_data * 0
    num_channels = input_data.shape[0]

    # Shift each channel of the mixture to align with mic0
    for channel_idx in range(num_channels):
        shift_samples = shifts[channel_idx]

        if np.abs(shift_samples) > input_data.shape[1]:
            shift_samples = input_data.shape[1]
            shifts[channel_idx] = shift_samples
            output_data[channel_idx] *= 0
            continue
        
        if inverse:
            shift_samples *= -1
        output_data[channel_idx] = shift_fn(input_data[channel_idx], shift_samples)
        shifts[channel_idx] = shift_samples

    return output_data, shifts



def get_shift_vector(target_position, mic_positions, sr, reference_channel=0, inverse=False):
    vec = []
    
    for channel_idx in range(mic_positions.shape[0]):
        shift_samples = criterion(mic_positions[channel_idx], target_position, sr) - criterion(mic_positions[reference_channel], target_position, sr)
        vec.append(shift_samples)
    
    vec = np.array(vec)
    return np.round(vec).astype(np.int32)

def to_categorical(index: int, num_classes: int):
    """Creates a 1-hot encoded np array"""
    data = np.zeros((num_classes))
    data[index] = 1
    return data

import importlib
import glob
from typing import Literal

def load_model_from_exp(exp_dir: str, mode: Literal['best', 'last', 'new'] = 'best'):
    with open(os.path.join(exp_dir, 'description.json'), 'rb') as f:
        experiment_params = json.load(f)
    
    if 'experiment_name' in experiment_params:
        exp_name = experiment_params['experiment_name']
        checkpoint_dir = exp_name
    else:
        exp_name = os.path.basename(exp_dir.strip('/'))
        checkpoint_dir = 'checkpoints'
    
    model_name = experiment_params['model_name']
    model_params = experiment_params['model_params']

    print('exp_dir', exp_dir)
    print(os.path.join(exp_dir, exp_name, f'{exp_name}_*.pt'))
    
    network2 = importlib.import_module(f'sep.training.{model_name}.network')
    model = network2.Network(**model_params)

    # Compat
    if mode == 'best':
        state_path = os.path.join(exp_dir, checkpoint_dir, f'state.pt')
        if not os.path.exists(state_path):
            print("[WARNING] Could not find experiment state dict, using load mode \'last\' instead")
            mode = 'last'

    # Load best model on val set
    if mode == 'best':
        state = torch.load(os.path.join(exp_dir, checkpoint_dir, f'state.pt'))
        val_losses = state['val_losses']
        best_epoch = np.argmin(val_losses)
        model.load_state_dict(torch.load(
            os.path.join(exp_dir, checkpoint_dir, f'{exp_name}_{best_epoch}.pt')),
            strict=True)
        print('Loaded best checkpoint', best_epoch)
    # Load last model
    elif mode == 'last':
        checkpoints = [x[:-len('.pt')] for x in glob.glob(os.path.join(exp_dir, checkpoint_dir, f'{exp_name}_*.pt'))]
        checkpoints_sorted = sorted(checkpoints, key = lambda c: -int(c[c.rfind('_')+1:]))
        if len(checkpoints_sorted) > 0:
            last_checkpoint = checkpoints_sorted[0] + '.pt'
            model.load_state_dict(torch.load(last_checkpoint), strict=True)
            print('Loaded last checkpoint', last_checkpoint)
        else:
            print("[WARNING] Provided experiment has no pretrained checkpoint, using default parameters instead")
    # Used new/untrained model (no need to do anything)
    elif mode == 'new':
        pass

    return model
