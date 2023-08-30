import argparse
import json
import keyword
import numpy as np
import os
import glob
import importlib
import librosa
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal
import torch
import torch.nn as nn

import sep.helpers.utils as utils
from sep.helpers.eval_utils import si_sdr
from sep.helpers.constants import  FS, INIT_WIDTH, freq_bins, n_fft, SPEED_OF_SOUND

from sep.Mic_Array import Mic_Array, Mic_Array_Monitor
import time
seed = 0
np.random.seed(seed)
import itertools

def preprocess_metadata(metadata):
    ## load the microphones information
    mics = ['mic00', 'mic01', 'mic02', 'mic03','mic04', 'mic05', 'mic06']  
    mic_positions = np.array([metadata[key]['position'] for key in mics ])   
    ### load the speaker ground truth information
    sources = [key for key in metadata.keys() if key.startswith('voice')]
    voice_positions = []
    num_spk = len(sources)
    for spk in sources:
        pos= metadata[spk]["position"]
        voice_positions.append([pos[0], pos[1], pos[2]])
    voice_positions = np.array(voice_positions)
    ### caluculate the ground-truth for sample offset by spk and mic
    sample_offsets_gt = np.zeros((mic_positions.shape[0] - 1, num_spk ))
    for j in range(voice_positions.shape[0] ):
        for i in range(1, mic_positions.shape[0]):
            source_pos = voice_positions[j]
            dis_offset = np.linalg.norm(source_pos - mic_positions[i, :]) - np.linalg.norm(source_pos - mic_positions[0, :])
            sample_offset = int(np.round(dis_offset/SPEED_OF_SOUND*FS))
            sample_offsets_gt[i-1, j] = sample_offset
    if "real" in metadata.keys():
        if metadata["real"]:
            print("real world data")
            sample_offsets_gt = np.zeros((mic_positions.shape[0] - 1, num_spk ))
            for j in range(voice_positions.shape[0] ):
                sample_offset = np.array(metadata[sources[j]]["shifts"])
                sample_offset = sample_offset - sample_offset[0]
                sample_offsets_gt[:, j] = sample_offset[1:]
    #### calculate the speaker range
    ## for synthetic data, it use the preset range in the dataset generation script
    ## for the real-world data, it use a minimim 5x5 m range which cover all the speakers,
    ## becaus when we collect data from different room it does not have a uniform range for speaker settings

    Range_spk = metadata['ROI']

    return mics, mic_positions, sources, voice_positions, sample_offsets_gt, Range_spk    

def check_label(sample_list, sample_offsets_gt):
    negatives = []
    positives = []
    num_spk = sample_offsets_gt.shape[1]
    for sample in sample_list:
        find_inside = False
        for i in range(num_spk):
            max_diff = np.amax(np.abs(sample_offsets_gt[:, i] - sample))
            if max_diff < 4.9:#4.5:
                find_inside = True
                break
        if not find_inside:
            negatives.append(sample.tolist())
        else:
            positives.append(sample.tolist())

    return negatives, positives

def main(args):
    ### -------------------- start localization and separation pipeline ------------------------ ###
    curr_dir = args.input_dir


    if args.debug_num < 0:
        begin_index = 0
    else:
        begin_index = args.debug_num

    debug = False

    T_begin = 0
    L = 3
    out_dir_root = "./outputs"
    Prone_method = "SRP"

    for idx in range(begin_index, begin_index + args.sample_number):
        ## create folder to save
        output_prefix_dir = os.path.join(curr_dir, '{:05d}'.format(idx))
        print("-"*15)
        print(output_prefix_dir)
        out_dir = out_dir_root + "/" + '{:05d}'.format(idx)
        if debug:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            curr_writing_dir_debug = os.path.join(out_dir, "debug")
            if not os.path.exists(curr_writing_dir_debug):
                os.makedirs(curr_writing_dir_debug)

        '''
        Read the configuiration data from json
        '''
        metadata_path = os.path.join(output_prefix_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            print("warning 1!!!! not file found")
            continue
        metadata = utils.read_metadata(output_prefix_dir)
        grid_size = 0.065
        mics, mic_positions, sources, voice_positions, sample_offsets_gt, Range_spk = preprocess_metadata(metadata)

        Mic_processor = Mic_Array(mic_positions, Spk_Range = Range_spk, Prone_method = Prone_method, SRP_fast=args.SRP_fast, cached = args.cached_init, cached_folder =output_prefix_dir, grid_size = grid_size)
        
        ### enable the monitor to save the data and print the debug number
        if debug:
            monitor = Mic_Array_Monitor(mic_positions, voice_positions, sample_offsets_gt, out_dir)
            Mic_processor.plugin_monitor(monitor)
        
        mix_data = []
        for m in mics:  
            channel = utils.read_audio_file_torch(os.path.join(output_prefix_dir, m) + '_mixed.wav')
            raw = channel[:, int(T_begin*48000):int((T_begin+L)*48000)]#.numpy()
            mix_data.append(raw)
        
        mix_data = torch.cat(mix_data, dim = 0)

        '''
            Conduct the localization with SRP-PHAT and Spotforming 
        '''
        save_data = {}
        # Mic_processor.Output_dir = out_dir
        ## apply SRP_PHAT
        patch_list, simple_pos = Mic_processor.Apply_SRP_PHAT(mix_data)
        # continue
        sample_list = [p.sample_offset for p in patch_list]
        negative_samples, positive_samples = check_label(sample_list, sample_offsets_gt)

        save_data["negative_sample"] = negative_samples
        save_data["positive_sample"] = positive_samples
        # print(positive_samples)
        # print(negative_samples)
        print("Generated sample:", len(negative_samples), len(positive_samples), len(sources))
        metadata_file =  output_prefix_dir + "/challeng_sample.json"
        with open(metadata_file, "w") as f:
            json.dump(save_data, f, indent=4)
        f.close()
  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help="Path to the input file")
    parser.add_argument('--sample_number',
                        type=int,
                        default=1,
                        help="Number of testing sample")
    parser.add_argument('--debug',
                        action='store_true',
                        help="Save intermediate outputs")
    parser.add_argument('--debug_num',
                        type=int,
                        default=-1,
                        help="Number of sample to debug")

    parser.add_argument('--SRP_fast',
                        action='store_true',
                        help="SRP_fast")

    parser.add_argument('--cached_init',
                        action='store_true',
                        help="cached_init")
    main(parser.parse_args())
