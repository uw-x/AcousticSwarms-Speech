"""
Torch dataset object for synthetically rendered
spatial data
"""
import json
import random

from typing import Tuple

import torch
import numpy as np
import os

from sep.training.base_dataset import BaseDataset
import sep.helpers.utils as utils
from sep.helpers.constants import MAX_SHIFTS


class Dataset(BaseDataset):
    def __init__(self, input_dir, dataset_type, n_mics=7, n_speakers = 5,
                 sr=48000, compression_prob=0.7, max_white_noise_variance = 1e-3,
                 max_pink_noise_variance = 5e-3, speaker_drop_prob=0.1, speaker_add_prob=0.1):
        super().__init__(input_dir=input_dir,
                         dataset_type=dataset_type,
                         sr=sr,
                         compression_prob=compression_prob,
                         max_white_noise_variance = max_white_noise_variance,
                         max_pink_noise_variance = max_pink_noise_variance)

        # Physical params
        self.n_mics = n_mics
        self.sr = sr
        self.n_speakers = n_speakers

        # Data augmentation
        self.speaker_drop_prob = speaker_drop_prob
        self.speaker_add_prob = speaker_add_prob

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Choose directory
        curr_dir = self.dirs[idx%len(self.dirs)]

        # Load mixture
        all_sources, target_voice_data, num_speakers = self.get_mixture_and_gt(curr_dir)

        return (all_sources.float(),
                target_voice_data.float(),
                num_speakers)

    def get_mixture_and_gt(self, curr_dir):
        """
        Given a target position and window size, this function figures out
        the voices inside the region and returns them as GT waveforms
        """
        # Get metadata
        with open(os.path.join(curr_dir, 'metadata.json'), 'rb') as json_file:
            metadata = json.load(json_file)

        real_sample = ('real' in metadata and metadata['real'] == True)

        # Get speakers
        voices = [key for key in metadata.keys() if 'voice' in key]

        # During training, randomly add and remove speakers to simulate 
        # errors in the localization stage
        if self.training:
            random.shuffle(voices)
            
            # With some probability, remove one speaker
            if random.random() < self.speaker_drop_prob:
                voices.pop()
                random.shuffle(voices)
            
            # With some probability, add a speaker
            if len(voices) < self.n_speakers:
                if random.random() < self.speaker_add_prob:
                    voices.append('fake_voice')
                    fake_shifts, fake_pos = self.get_negative_region(metadata, 1)
                    fake_shifts = -fake_shifts # For compatibility
                    metadata['fake_voice'] = dict(position=fake_pos, shifts=fake_shifts)
                    random.shuffle(voices)

        voice_positions = np.array([metadata[key]['position'] for key in voices])

        # Get microphone positions
        mics = [key for key in metadata.keys() if 'mic' in key]
        mic_positions = np.array([metadata[key]['position'] for key in mics])
        
        # Load mixture
        mixture = []
        for mic in mics:
            channel = utils.read_audio_file_torch(os.path.join(curr_dir, mic) + '_mixed.wav')
            mixture.append(channel)
        mixture = torch.vstack(mixture)

        # Initialize vector of shifted mixtures and vector of target speeches
        shifted_vector = torch.zeros((self.n_mics * self.n_speakers, mixture.shape[1]))
        target_voice_data = torch.zeros((self.n_speakers, mixture.shape[-1]))
       
        assert len(voices) <= self.n_speakers, f"Dataset has too many speakers\n\
                Expected <= {self.n_speakers}. Found {len(voices)}"
        
        for i in range(len(voices)):
            voice = voices[i]
            # Load ground truth audio
            if voice == 'fake_voice':
                gt = None#torch.zeros((mixture.shape[1]))
            else:
                denoised_file = os.path.join(curr_dir, mics[0] + '_' + str(voice) + '_denoised.wav')
                audio_file = os.path.join(curr_dir, mics[0] + '_' + str(voice) + '.wav')
                
                if os.path.exists(denoised_file):
                    gt = utils.read_audio_file_torch(denoised_file)
                else:
                    gt = utils.read_audio_file_torch(audio_file)
            
            # For real samples, use sample shifts because they're more accurate
            if real_sample:
                voice_shift = np.array(metadata[voice]['shifts'])
                voice_shift -= voice_shift[0]
                voice_shift = -voice_shift
            else:
                voice_pos = voice_positions[i]
                voice_shift = utils.get_shift_vector(voice_pos, mic_positions, self.sr, reference_channel=0)
            
            # Perturb sample shifts during training (only for synthetic data)
            if self.training:
                if not real_sample:
                    perturbation = np.random.randint(low = -MAX_SHIFTS[0], high = +MAX_SHIFTS[0] + 1, size=voice_shift.shape[-1])
                    perturbation[0] = 0 # Reference microphone must not be shifted
                    voice_shift += perturbation

            # Shift mixture to given samples
            shifted, _ = utils.shift_mixture_given_samples(mixture, voice_shift.astype(np.int32))
            
            # Assign shifted mixture to the num_mic-channel block dedicated to this speaker
            shifted_vector[i * self.n_mics: (i + 1) * self.n_mics] = shifted
            
            if gt is not None:
                target_voice_data[i] = gt
        
        # Apply audio perturbations during training
        if self.training:
            shifted_vector = self.perturb_audio(shifted_vector)

        # Apply codec to synthetic samples
        # Note that if the codec is applied during training, it will always be 
        # applied during validation due to compression_prob initialization
        if (not real_sample) and (np.random.random() < self.compression_prob):
            shifted_vector, target_voice_data = self.apply_codec(shifted_vector, target_voice_data)
       
        return shifted_vector, target_voice_data, len(voices)
