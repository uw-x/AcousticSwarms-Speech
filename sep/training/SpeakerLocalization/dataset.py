"""
Torch dataset object for synthetically rendered
spatial data
"""
import json
import random

from typing import Tuple
from pathlib import Path

import torch
import numpy as np
import os

from sep.training.base_dataset import BaseDataset
import sep.helpers.utils as utils
from sep.helpers.pink_noise import *
from sep.helpers.constants import MAX_SHIFTS


class Dataset(BaseDataset):
    def __init__(self, dataset_type, input_dir, n_mics=7, sr=48000,
                 negatives=0.3, max_white_noise_variance = 1e-3,
                 max_pink_noise_variance = 5e-3, compression_prob = 0.7,
                 fixed_window_condition=-1, challenge_ratio=0.8,
                 use_dereverb=False, use_denoised=False, scale_neg_prob=False):
        super().__init__(input_dir=input_dir,
                         dataset_type=dataset_type,
                         sr=sr,
                         compression_prob=compression_prob,
                         max_white_noise_variance = max_white_noise_variance,
                         max_pink_noise_variance = max_pink_noise_variance)
        # Physical params
        self.n_mics = n_mics
        self.sr = sr
        self.window_condition = fixed_window_condition

        # Fraction of negative samples (no speakers nearby)
        self.negatives = negatives
        self.challenge_ratio = challenge_ratio
        self.scale_neg_prob = scale_neg_prob

        self.dereverb = use_dereverb
        self.use_denoised = use_denoised

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mixed_data - M x T
            target_voice_data - M x T
            window_idx_one_hot - 1-D
        """
        curr_dir = self.dirs[idx%len(self.dirs)]

        # Get metadata
        with open(Path(curr_dir) / 'metadata.json') as json_file:
            metadata = json.load(json_file)

        # Check if this is a real sample or a synthetic sample
        real_sample = ('real' in metadata and metadata['real'] == True)
        
        # Compute speaker shifts for synthetically generated data
        if not real_sample:
            for key in metadata:
                if key.startswith('voice'):
                    metadata[key]['shifts'] = [0] + metadata[key]['shifts']
        
        # Choose width condition as either relaxed (4 samples) or strict (2 samples)
        if self.window_condition < 0:
            window_condition = np.random.randint(2)
        else:
            window_condition = self.window_condition
        width_embedding = torch.tensor(utils.to_categorical(window_condition, 2))

        # Random split of negatives and positives
        with open(Path(curr_dir) / 'challeng_sample.json') as json_file:
            srp_phat_predictions = json.load(json_file)
            srp_phat_false_positives = srp_phat_predictions['negative_sample']

        if self.scale_neg_prob:
            speakers = [key for key in metadata.keys() if 'voice' in key]
            num_speakers = len(speakers)
            
            false_positives_per_speaker = len(srp_phat_false_positives) / num_speakers
            neg_prob = (0.5 - 0.3) / (14 - 6) * false_positives_per_speaker + 0.15
            
            if neg_prob > 0.5:
                neg_prob = 0.5
            elif neg_prob < 0.2:
                neg_prob = 0.2
        else:
            neg_prob = self.negatives
        
        if np.random.uniform() < neg_prob:   
            use_challenging_sample = False
            if np.random.uniform() < self.challenge_ratio:
                if len(srp_phat_false_positives) > 0:
                    target_shift, _ = self.get_negative_region_SRP(metadata, window_condition, srp_phat_false_positives)
                    use_challenging_sample = True
            
            if not use_challenging_sample:
                target_shift, _ = self.get_negative_region(metadata, window_condition)
            
            pos = False
        else:
            target_shift = self.get_positive_region(metadata, window_condition)
            pos = True
        
        # Load mixture audio and ground truth audio given target shift and window condition
        all_sources, target_voice_data = self.get_mixture_and_gt(metadata, curr_dir, target_shift, window_condition, pos)
        
        # Sanity check, negative regions should be all zeros, positive regions should have something audible 
        if pos:
            assert (target_voice_data > 0).any()
        else:
            assert (target_voice_data == 0).all()

        return all_sources.float(), target_voice_data.float(), width_embedding.float()

    def get_positive_region(self, metadata, window_condition):
        """Chooses a target position containing a speaker source"""
        real_sample = ('real' in metadata and metadata['real'] == True)
        
        # Choose a random voice
        voice_keys = [x for x in metadata if "voice" in x]
        random_voice_key = random.choice(voice_keys)
        
        # Get speaker sample shifts
        shifts = np.array(metadata[random_voice_key]['shifts'])
        shifts -= shifts[0]
        shifts = -shifts
            
        # During training, perturb shift vector (only applies to synthetic data)
        if not real_sample and self.training:
            shifts += np.random.randint(low = -MAX_SHIFTS[window_condition], high = +MAX_SHIFTS[window_condition] + 1, size=shifts.shape[-1])
            shifts[0] = 0 # Reference microphone should not be shifted

        return shifts

    def get_mixture_and_gt(self, metadata, curr_dir, target_shift, window_condition, pos=True):
        """
        Given a target position and window size, this function figures out
        the voices inside the region and returns them as GT waveforms
        """
        real_sample = ('real' in metadata and metadata['real'] == True)

        # Get speaker and microphone positions
        voices = [key for key in metadata.keys() if 'voice' in key]
        mics = [key for key in metadata.keys() if 'mic' in key]
        
        # Load mixture
        mixture = []
        for mic in mics:
            channel = utils.read_audio_file_torch(os.path.join(curr_dir, mic) + '_mixed.wav')
            mixture.append(channel)
        mixture = torch.vstack(mixture)

        # Shift mixture given the target sample shift
        target_shift = np.round(target_shift).astype(np.int32)
        shifted_vector, shifts = utils.shift_mixture_given_samples(mixture, target_shift)
        
        assert shifts[0] == 0, f'Sample shift on reference microphone should be equal to 0.\
                                 Found {shifts[0]}.'
        
        # Create ground truth audio
        # Ground truth is the audio from the closest speaker to the target shift that is 
        # whose sample shifts are close enough to the target shift
        target_voice_data = torch.zeros((1, shifted_vector.shape[-1]))
        included_voices = []
        for i, (voice) in enumerate(voices):
            # Get speaker shift
            voice_shift = np.array(metadata[voice]['shifts'])
            voice_shift -= voice_shift[0]
            voice_shift = -voice_shift
            
            # If the voice shift is close enough to the target shift, included it in gt
            if np.linalg.norm(voice_shift - target_shift, ord=np.inf) <= MAX_SHIFTS[window_condition]:
                included_voices.append((np.linalg.norm(voice_shift - target_shift, ord=np.inf), i))
        
        # Sort voices by proximity to target shift
        included_voices = sorted(included_voices, key=lambda x: x[0])
        
        # Load closest speaker audio as ground truth
        if len(included_voices) > 0:
            i = included_voices[0][1]
            voice = voices[i]
            
            if self.use_denoised:
                denoised_file = os.path.join(curr_dir, mics[0] + '_' + str(voice) + '_denoised.wav')
                audio_file = os.path.join(curr_dir, mics[0] + '_' + str(voice) + '.wav')
                
                if os.path.exists(denoised_file):
                    gt = utils.read_audio_file_torch(denoised_file)
                else:
                    gt = utils.read_audio_file_torch(audio_file)
            
            else:
                suffix = ''
                if self.dereverb:
                    suffix = '_dereverb'

                audio_file = os.path.join(curr_dir, mics[0] + '_' + str(voice) + suffix + '.wav')
                gt = utils.read_audio_file_torch(audio_file)

            target_voice_data = gt
        
        # Apply audio perturbations during training
        if self.training:
            shifted_vector = self.perturb_audio(shifted_vector)

        # Apply codec to synthetic samples
        # Note that if the codec is applied during training, it will either always be 
        # applied or never be applied during validation due to compression prob initialization
        if (not real_sample) and (np.random.random() < self.compression_prob):
            shifted_vector, target_voice_data = self.apply_codec(shifted_vector, target_voice_data)

        return shifted_vector, target_voice_data
