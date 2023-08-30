from typing import Any
import torch
import torch.nn as nn
import numpy as np
import time
from sep.Mic_Array import Mic_Array
import sep.helpers.utils as utils
from sep.training.base_network import BaseNetwork
from sep.training.SpeakerLocalization.network import unnormalize_input, normalize_input


def roll_by_gather(mat,dim, shifts: torch.LongTensor):
    """
    Applies sample shifts quickly
    """
    n_rows, n_cols = mat.shape
    
    if dim==0:
        arange1 = torch.arange(n_rows, device=mat.device).view((n_rows, 1)).repeat((1, n_cols))
        arange2 = (arange1 - shifts) % n_rows
        return torch.gather(mat, 0, arange2)
    elif dim==1:
        arange1 = torch.arange(n_cols, device=mat.device).view(( 1,n_cols)).repeat((n_rows,1))
        arange2 = (arange1 - shifts) % n_cols
        return torch.gather(mat, 1, arange2)

class DataParallelSpotModel(BaseNetwork):
    def __init__(self, model: nn.Module, use_fp16: bool=False, batch_size: int=128):
        super().__init__()
        self.model = nn.DataParallel(model)
        self.batch_size = batch_size

        self.dtype = torch.bfloat16 if use_fp16 else torch.float32
        if use_fp16:
            self.model.to(torch.bfloat16)
    
    def shift_and_sep(self,
                     input_channels: torch.Tensor,
                     patch_list: list,
                     Strict: int = 0,
                     save_input: bool = False) -> np.ndarray:
        BATCH_SIZE = self.batch_size
        num_patches = len(patch_list)

        num_mic = input_channels.shape[0]
        device = self.device
        
        # Inference on device
        mdl = self.model
        mdl.eval()
        
        shifted_mixtures = [] # Debugging
        with torch.no_grad():
            # Copy once to device
            mix = input_channels.to(device, dtype=self.dtype)
            
            # Instantiate tensors to hold data/results on device
            data = torch.zeros((BATCH_SIZE, num_mic, mix.shape[-1]), device=device, dtype=self.dtype)
            results = torch.zeros((num_patches, mix.shape[-1]), device=device, dtype=self.dtype)
            
            # Tensor for strict window conditions
            window_condition_strict = torch.zeros((BATCH_SIZE, 2), device=device, dtype=self.dtype)
            window_condition_strict[:, 0] = 1

            # Tensor for relaxed window conditions
            window_condition_relaxed = torch.zeros((BATCH_SIZE, 2), device=device, dtype=self.dtype)
            window_condition_relaxed[:, 1] = 1

            # Choose which window condition should be used
            if Strict == 1:
                window_condition = window_condition_strict
            else:
                window_condition = window_condition_relaxed

            for i in range(0, num_patches, BATCH_SIZE):
                target_patch = patch_list[i:i+BATCH_SIZE]
                num_elements_in_batch = len(target_patch)
                
                # Apply sample shifts to each element in the batch
                for j in range(len(target_patch)):
                    sample_shifts = -torch.Tensor([0, *target_patch[j].sample_offset]).unsqueeze(1)
                    sample_shifts = torch.round(sample_shifts).long().to(device)
                    data[j] = roll_by_gather(mix, 1, sample_shifts)
                
                # Debugging
                if save_input:
                    shifted_mixtures.extend(data)

                # Normalize                
                data_norm, means, stds = normalize_input(data[:num_elements_in_batch])

                # Forward pass
                result = mdl(data_norm, window_condition[:num_elements_in_batch])

                # Denormalize
                results[i:i+BATCH_SIZE] = unnormalize_input(result, means, stds)[:, 0]

            # Copy results to CPU
            result = results.cpu().float().numpy()

        if save_input:
            return result, shifted_mixtures.cpu()
        else:
            return result

class JointModel(BaseNetwork):
    def __init__(self,
                 spot_exp_dir: str,
                 sep_exp_dir: str,
                 use_spot_dataparallel: bool = True,
                 use_fp16: bool = False,
                 spot_batch_size: int = 128,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.spot_model = utils.load_model_from_exp(spot_exp_dir, mode='best')
        self.times = [0, 0, 0, 0, 0]
        if use_spot_dataparallel:
            self.spot_model = DataParallelSpotModel(self.spot_model, use_fp16, spot_batch_size)

        self.sep_model = utils.load_model_from_exp(sep_exp_dir, mode='best')
        
        self.previous_config = None

    def setup(self, mic_positions, speaker_range, cached = False, cached_folder = None):
        """
        Initializes microphone array. If same microphone positions and speaker range is used,
        then we use don't reinitialize the microphone array to save time.
        """
        current_config = '~'.join([f"{x:.05f}" for x in mic_positions.flatten()])\
                                   + '|' + '~'.join([f"{x:.05f}" for x in speaker_range])
        if current_config == self.previous_config:
            print("reuse the previous recycle!")
            pass
        else:
            self.Mic_processor = Mic_Array(mic_positions, Spk_Range=speaker_range, cached = cached, cached_folder = cached_folder)
            self.previous_config = current_config

    def reset_tracking(self):
        self.Mic_processor.reset_tracking()

    def forward(self, mix_data: torch.Tensor) -> Any:
        self.times = [0, 0, 0, 0, 0]
        patches, audio_loc, SRP_drop, stage1_drop, spot_times = self.localize_by_separation(mix_data)
        t0 = time.time()
        audio = self.separate_by_localization(mix_data, patches)
        t1 = time.time()
        self.times[4] = t1 - t0
        return patches, audio_loc, audio, SRP_drop, stage1_drop, spot_times

    def localize_by_separation(self, mix_data: torch.Tensor, run_demo_folder = None, tracking = False):
        assert self.previous_config is not None,\
            "Microphone positions and spk range were not provided\, did you forget to call .setup()?"
        
        # Apply SRP_PHAT to prune the locations
        t0 = time.time()
        patch_list, simple_pos = self.Mic_processor.Apply_SRP_PHAT(mix_data)
        t1 = time.time()
        self.times[0] = t1 - t0
        SRP_drop = 0

        if(len(patch_list)) <= 0:
            print("No spk picked in SRP-PHAT")
            return [], [], 0, 0, 0
        
        # Perform spotforming on the candidates, with a relaxed window
        t0 = time.time()
        patch_list = self.Mic_processor.Spotform_Big_Patch(mix_data, patch_list, self.spot_model)
        t1 = time.time()
        self.times[1] = t1 - t0

        if(len(patch_list)) <= 0:
            print("No spk picked in Spotform_Big_Patch")
            return [], [], 0, 0, 0
        stage1_drop = 0


        # Perform spotforming on the candidates, with a strict window
        t0 = time.time()
        output_pair = self.Mic_processor.Spotform_Small_Patch_Parallel(mix_data, patch_list, self.spot_model, run_demo_folder = run_demo_folder)
        t1 = time.time()
        self.times[2] = t1 - t0
        if(len(output_pair)) <= 0:
            print("No spk picked in Spotform_Small_Patch")
            return [], [], 0, 0, 0

        # Cluster the resulting
        t0 = time.time()
        if tracking:
            audio_final, patch_final, spot_times, _ = self.Mic_processor.Clustering_tracking_new(output_pair)
        else:
            audio_final, patch_final, spot_times, _ = self.Mic_processor.Clustering_new(output_pair)
        t1 = time.time()
        self.times[3] = t1 - t0
        if(len(patch_final)) <= 0:
            print("No spk picked in Clustering")
            return [],[], 0, 0, 0
        audio_final = np.array(audio_final)
        return patch_final, audio_final, SRP_drop, stage1_drop, spot_times
    
    def separate_by_localization(self, mix_data: torch.Tensor, target_patches: list):
        if len(target_patches) == 0:
            return None
        return self.sep_model.infer(mix_data, [p[0] for p in target_patches])

    def separate_by_localization_by_sample(self, mix_data: torch.Tensor, sample_lists: list):
        if len(sample_lists) == 0:
            return None
        return self.sep_model.infer_sample(mix_data, sample_lists)

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self.spot_model.to(device)
            self.sep_model.to(device)
        return super().to(device, *args, **kwargs)
