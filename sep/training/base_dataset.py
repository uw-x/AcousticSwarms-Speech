import torch
from pathlib import Path
import numpy as np

from sep.helpers.pink_noise import powerlaw_psd_gaussian
import sep.helpers.utils as utils
from sep.helpers.codec import OpusCodec
from sep.helpers.constants import CHANNELS_PER_MIC,\
                                  CODEC_FRAME_DURATION_S,\
                                  MAX_SHIFTS,\
                                  ROOM_DIM,\
                                  MAX_SPEAKER_RELATIVE_HEIGHT,\
                                  NEG_SAMPLE_INTIAL_CANDIDATES


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 input_dir,
                 dataset_type,
                 sr,
                 compression_prob,
                 max_white_noise_variance = 1e-3,
                 max_pink_noise_variance = 5e-3):
        super().__init__()
        # List all sample directories
        self.dirs = sorted(list(Path(input_dir).glob('[0-9]*')))
        
        self.dataset_type = dataset_type
        if self.dataset_type == 'train':
            self.training = True
            
            # Always apply perturbation on training set    
            self.perturb_prob = 1
            self.compression_prob = compression_prob
        else:
            self.training = False
            
            # Never perturb
            self.perturb_prob = 0

            # Always use compression if there's a chance we do it on training set
            self.compression_prob = (abs(compression_prob) > 1e-6)
        
        # Initialize codec if needed
        if self.compression_prob > 0:
            self.codec = OpusCodec(CHANNELS_PER_MIC, sr, CODEC_FRAME_DURATION_S)

        # Initialize noise parameters
        self.max_white_noise_variance = max_white_noise_variance
        self.max_pink_noise_variance = max_pink_noise_variance
    
    def get_points_outside_patch_union(self, candidate_points, voice_sample_offsets, mic_positions, window_condition):
        """
        Filters points in the room by their proximity to any speaker. Points that are too close, i.e. within
        some number of samples given by the window condition, to any speaker in the room get discarded.
        """       
        remaining_shifts = []
        valid_points = []
        distances = []
        
        # For each candidate point
        for i, candidate in enumerate(candidate_points):
            # Get shift vector from position
            candidate_sample_offset = utils.get_shift_vector(candidate, mic_positions, self.sr, reference_channel=0)
        
            # Get absolute difference between this shift vector and that of all other voices
            diff = np.absolute(voice_sample_offsets - candidate_sample_offset)

            # Compute L-inf distance relative to each source position
            Linf_distances = np.max(diff, axis=1)

            # If closest source is not covered by a patch centered at this point, with width given
            # by the window condition, then this is a negative point (no sources nearby)
            closest_source_distance = np.min(Linf_distances)
            if closest_source_distance > MAX_SHIFTS[window_condition] + 1:
                # Store candidate point and corresponding shift as valid
                remaining_shifts.append(candidate_sample_offset)
                valid_points.append(candidate)

                # We store L1-norm of candidates for weighting later
                distances.append(np.min(np.linalg.norm(diff, ord=1, axis=1)))
        
        return remaining_shifts, distances, valid_points
        
    def get_negative_region_SRP(self, metadata, window_condition, negative_list):
        """
        Chooses a point where corresponding sample offset vector where minimum sample offset from
        voice closest voice pos is > MAX_OFFSET
        """
        real_sample = ('real' in metadata and metadata['real'] == True)

        voices = [x for x in metadata if 'voice' in x]
        mic_positions = np.array([metadata[x]['position'] for x in metadata if 'mic' in x])

        # Compute sample shifts for each voice
        voice_sample_offsets = np.zeros((len(voices), mic_positions.shape[0]))
        
        # For real samples, use shifts as is in the metadata
        if real_sample:
            for i, voice in enumerate(voices):
                voice_shift = np.array(metadata[voice]['shifts'])
                voice_shift -= voice_shift[0]
                voice_shift = -voice_shift
                voice_sample_offsets[i] = voice_shift
        # For synthetic samples, compute shifts from position
        else:
            voice_positions = np.array([metadata[key]['position'] for key in voices])
            for i, voice in enumerate(voice_positions):
                shifts = utils.get_shift_vector(voice, mic_positions, self.sr, reference_channel=0)
                voice_sample_offsets[i] = shifts
        patch_width = MAX_SHIFTS[window_condition]
        # Sample random points and keep whatever is far enough from any voice
        # Define room borders to sample from
        
        valid_negative = False

        while not valid_negative:
            challeng_index = np.random.choice([i for i in range(len(negative_list))])
            challeng_sample = negative_list[challeng_index]
            random_shift = [0]
            for p in challeng_sample:
                random_shift.append(p)
            random_shift = np.array(random_shift)
            random_shift = -1*random_shift
            random_shits = np.random.choice([-2, -1, 0, 1, 2], 6)
            random_shift[1:] += random_shits

            valid_negative = True
            for i in range(voice_sample_offsets.shape[0]):
                diff = np.amax(np.abs(voice_sample_offsets[i] - random_shift))
                if diff <= patch_width + 1:
                    valid_negative = False
                    break
           
        return random_shift, None


    def get_negative_region(self, metadata, window_condition):
        """
        Chooses a point where corresponding sample offset vector where minimum sample offset from
        voice closest voice pos is > MAX_OFFSET
        """
        real_sample = ('real' in metadata and metadata['real'] == True)

        voices = [x for x in metadata if 'voice' in x]
        mic_positions = np.array([metadata[x]['position'] for x in metadata if 'mic' in x])

        # Compute sample shifts for each voice
        voice_sample_offsets = np.zeros((len(voices), mic_positions.shape[0]))
        
        # For real samples, use shifts as is in the metadata
        if real_sample:
            for i, voice in enumerate(voices):
                voice_shift = np.array(metadata[voice]['shifts'])
                voice_shift -= voice_shift[0]
                voice_shift = -voice_shift
                voice_sample_offsets[i] = voice_shift
        # For synthetic samples, compute shifts from position
        else:
            voice_positions = np.array([metadata[key]['position'] for key in voices])
            for i, voice in enumerate(voice_positions):
                shifts = utils.get_shift_vector(voice, mic_positions, self.sr, reference_channel=0)
                voice_sample_offsets[i] = shifts

        # Sample random points and keep whatever is far enough from any voice
        # Define room borders to sample from
        lx = np.min(mic_positions[:, 0]) - ROOM_DIM
        ux = np.max(mic_positions[:, 0]) + ROOM_DIM
        ly = np.min(mic_positions[:, 1]) - ROOM_DIM
        uy = np.max(mic_positions[:, 1]) + ROOM_DIM
        
        candidate_shifts = []
        while len(candidate_shifts) == 0:
            # Choose 30 random points
            npts = NEG_SAMPLE_INTIAL_CANDIDATES
            candidate_points = [np.random.uniform(lx, ux, size=npts), np.random.uniform(ly, uy, size=npts)]
            
            # If dataset is 3d, choose random z-coordinate
            if len(mic_positions[0]) == 3:
                candidate_points.append(np.random.uniform(0, MAX_SPEAKER_RELATIVE_HEIGHT, size=npts))
            
            candidate_points = np.array(candidate_points).T
            
            # Filter candidate points
            candidate_shifts, distances, valid_points = self.get_points_outside_patch_union(candidate_points, voice_sample_offsets, mic_positions, window_condition)

        # Sample one of the points from the remaning valid points, weighted by 1/L1-distance to closest speaker
        p = np.zeros(len(candidate_shifts))
        for i, pt in enumerate(candidate_shifts):
            p[i] = np.min(1/distances[i])
        p /= np.sum(p)

        # Get index of sampled point
        idx = np.random.choice(len(candidate_shifts), p = p)
        
        # Get corresponding point and sample shift
        random_shift = candidate_shifts[idx]
        chosen_point = valid_points[idx]

        return random_shift, chosen_point
    
    def perturb_audio(self, input_audio):
        """
        Perturbs input audio by adding white and pink noise with random variance
        Small note: Although some variables have "variance" in the name, this
        may not correspond to the actual variance. Think of it more as some arbitrary
        scale factor instead.
        """
        # Choose pink noise level
        pink_noise_level = self.max_pink_noise_variance * np.random.rand()
        
        # Generate pink noise
        pink_noise = powerlaw_psd_gaussian(1, input_audio.shape, random_state=np.random.randint(2**31))
        pink_noise = pink_noise_level*torch.tensor(pink_noise)
        
        # Choose white noise level
        white_noise_level = self.max_white_noise_variance * np.random.rand()

        # Generate white noise
        white_noise = white_noise_level*torch.tensor(np.random.normal(0, 1, size=input_audio.shape))
        
        # Sum them up
        output = input_audio + white_noise + pink_noise
        
        return output

    def apply_codec(self, shifted_vector, target_voice_data):
        """
        Applies Opus compression & decompression to each channel in audio mixture and ground truth
        """
        # Apply codec to each input channel separately
        for i in range(shifted_vector.shape[0]):
            shifted_vector[i] = self.codec.apply(shifted_vector[i])
        
        # Apply codec to each gt channel separately
        for i in range(target_voice_data.shape[0]):
            target_voice_data[i] = self.codec.apply(target_voice_data[i])

        return shifted_vector, target_voice_data
