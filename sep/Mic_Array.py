import torch
import numpy as np
import soundfile as sf
import time 

import sep.helpers.utils as utils

from sep.helpers.eval_utils import si_sdr, split_wav, split_wise_sisdr
from sep.helpers.constants import FS, SPEED_OF_SOUND, INIT_WIDTH, freq_bins, n_fft, MIN_TOLERANCE, MIN_WIDTH, SPOT_POWER_THRESHOLD2, USE_RELATIVE_SPOT_POWER, SI_SNR_POWER_THRESHOLD, SPOT_POWER_THRESHOLD1
from sep.Traditional_SP.SRP_Prunning import SRP_PHAT
from sep.Traditional_SP.Patch_3D import Patch
from sep.helpers.local_utils_3d import visualize_single, visualize,visualize_color, visualize_result, binary_search_baseline, search_area, max_avg_power, visualize_small_patch

import matplotlib.pyplot as plt


### check the sisnr window by window
def check_sisnr_win(sisnr_list, SISNR_THRESHOLD = -2, SISNR_THRESHOLD2 = -7): # -2. -6
    
    SAME_FLAG = False
    SAME_FLAG2 = True
    for value in sisnr_list:
        if value > SISNR_THRESHOLD:
            SAME_FLAG = True
        if value < SISNR_THRESHOLD2:
            SAME_FLAG2 = False

    return (SAME_FLAG and SAME_FLAG2)

### instead of pick up the highest power patch as the localization results, 
### we use the weighted sum to compute the localization result with threshold
def weight_mean_pos(patch_list, powers, id_lists):
    total_pos = np.zeros((3, ))
    
    total_power = 0
    max_power = powers[id_lists[0]]
    offsets = patch_list[0].sample_offset
    
    total_offsets = np.zeros(offsets.shape)
    for _id in id_lists:
        if powers[_id] < max_power*0.75: # 0.9
            continue
        pos = patch_list[_id].center_pos()
        total_pos += powers[_id]*pos
        total_offsets += powers[_id]* patch_list[_id].sample_offset
        total_power += powers[_id]
    return total_pos/total_power, total_offsets/total_power


def find_merge_center(merged_offests, init_area, mic_positions, Big_patch_center):
    begin_width = 3
    num_pair = mic_positions.shape[0] - 1
    width_list0 = [begin_width for i in range(num_pair)]  

    patch_center = Patch(merged_offests, width_list0, None)

    area = patch_center.hyperbola_general_area(init_area[0, :], init_area[1, :],init_area[2, :], mic_positions, SPEED_OF_SOUND, FS)
    area = (area == 1)

    if np.sum(area) == 0:
        ### fail to find the points satisfied the sample offset and init width
        ### loose the width constrains to find the inside points
        begin_width = 3
        find_center = False
        for factor in range(4):
            patch_center.width_list = [begin_width + factor for n in range(num_pair)]
            area = patch_center.hyperbola_general_area(init_area[0, :], init_area[1, :],  init_area[2, :], mic_positions, SPEED_OF_SOUND, FS)
            area = (area == 1)
            if np.sum(area) > 0:
                points0 = init_area[:, area]
                patch_center.area_points = points0
                find_center = True
            break
        if not find_center:
            # print("warining!!!! use old center")
            patch_center.peak_pos = Big_patch_center
    else:
        points0 = init_area[:, area]
        patch_center.area_points = points0

    return patch_center

### monitor to debug
class Mic_Array_Monitor(object):
    def __init__(self, mic_positions, voice_positions, sample_offsets_gt, Output_dir):
        self.Output_dir = Output_dir
        self.voice_positions = voice_positions
        self.mic_positions = mic_positions
        self.sample_offsets_gt = sample_offsets_gt



### class for our localization by separation algorithm
class Mic_Array(object):
    def __init__(self,  mic_positions, demo = False, Spk_Range=None, grid_size = 0.05, Prone_method = "SRP", MIN_TRIGGER_POWER = 0.5, SRP_fast = False, cached = False,cached_folder = None):
        self.Prone_method = Prone_method
        self.MIN_TRIGGER_POWER = MIN_TRIGGER_POWER
        self.visual_save = False

        '''
        Initalize the speaker Range
        '''
       
        self.Range_spk = Spk_Range
        print("Init the Range_spk: ", Spk_Range)


        '''
        Read the microphones configuiration data from json
        ''' 
        self.mic_positions = mic_positions
        self.num_mic = mic_positions.shape[0]
        self.upper_bound_pairwise =  np.zeros((mic_positions.shape[0] - 1, )) 
        for i in range(1, mic_positions.shape[0]):
            self.upper_bound_pairwise[i-1] = (np.linalg.norm(mic_positions[i] - mic_positions[0]) + 0.08)/SPEED_OF_SOUND*FS

        '''
            init SRP_PHAT to do coarse localize
        '''
        threshold_settings = [0.15, 0.015, 0.05] #[relative ratio of threshold, lower bound of threshold, upper bound of threshold]


        if SRP_fast:
            if torch.cuda.is_available():
                device = torch.device("cuda") 
            else:
                device = torch.device("cpu") 
        else:
            device = None


        self.SRP_node = SRP_PHAT(
                mic_pos = mic_positions, 
                freq_bins = freq_bins,
                Range_spk = Spk_Range,
                grid_size = grid_size,
                FS = FS,
                n_fft = n_fft,
                threshold=threshold_settings,
                WIDTH=INIT_WIDTH,
                device = device,
                cached = cached,
                cached_name = cached_folder)

        self.original_times = 0
        self.spotforming_times = 0

    def plugin_monitor(self, monitor):
        self.monitor = monitor
        self.visual_save = True

    def Apply_SRP_PHAT(self, mix_data):
        # print("Begin Apply_SRP_PHAT")
        self.SRP_node.reset()
        self.spotforming_times = 0
        self.original_times = 0
        mix_data_np = mix_data.numpy()

        t1 = time.time()
        if mix_data_np.shape[1] >= 72000:
            WIN_SIZE = 36000            
        else:
            WIN_SIZE = 24000

        if self.Prone_method == "SRP":
            self.SRP_node.SRP_Map_WINDOW_new(mix_data_np, window = WIN_SIZE) 
        elif  self.Prone_method == "MUSIC":
            self.SRP_node.MUSIC_Map_WINDOW(mix_data_np, window = WIN_SIZE)
        elif  self.Prone_method == "TOPS":
            self.SRP_node.TOPS_Map_WINDOW(mix_data_np, window = WIN_SIZE)

        t2 = time.time()
        ### pruning the TDoA space use SRP-PHAT, 
        ### the patch list is the list of the remaining hypercubes of TDoA space
        patch_list = self.SRP_node.local_source_adaptive()
        t3 = time.time()
        
        simple_pos = np.zeros((3, 3))


        ### this part is for debugging
        ### it will visualize SRP-PHAT image and the remaining hypercubes in the space
        if self.visual_save:
            print("SRP-pHAT TIME: ", t2 - t1)
            print("local peak time: ", t3 - t2)
            voice_positions = self.monitor.voice_positions
            save_folder = self.monitor.Output_dir + '/debug/'
            self.SRP_node.visualize_each_layer(voice_positions, save_folder)
            # plt.figure()
            visualize(patch_list, self.mic_positions, voice_positions, self.Range_spk)
            plt.savefig(self.monitor.Output_dir + "/SRP_PHAT.png")
            

        return patch_list, simple_pos

    def Spotform_Big_Patch(self, mix_data, patch_list, spot_model):
        candidate_finished = []
        self.big_spotforming_times = len(patch_list) 
        if self.visual_save:
            print("----- Step2: remove big patch using Spotforming ------")
            gt_label = []
            for patch in patch_list:
                valid = patch.check_gt(self.monitor.sample_offsets_gt)
                if valid:
                    gt_label.append(True)
                else:
                    gt_label.append(False)
        ### apply the localization model on the width=4 hypercubes and do coarsely pickup
        candidate_finished, powers_with_dis, Relative_Threshold = binary_search_baseline(mix_data, spot_model, patch_list, self.mic_positions)   
        
        ### this part is for debugging  
        ### it will output the sample offsets and output power of each hypercubes candidates
        if self.visual_save:
            for i in range(len(patch_list)):
                print(i, gt_label[i],SPOT_POWER_THRESHOLD1, powers_with_dis[i])
                print(patch_list[i].sample_offset, patch_list[i].width_list[0])    
            print("{} source in the room is found, to be determine further with non-max supression".format(len(candidate_finished)))

        self.Relative_Threshold = Relative_Threshold

        
        return candidate_finished
    

    def Spotform_Small_Patch_Parallel(self, mix_data, candidate_finished, spot_model, sample_gt = None, run_demo_folder = None):
        width_list0 = [2 for i in range(self.num_mic - 1)]   
        output_pair = []
        debug_pos = []
        debug_power = []


        total_patch = []
        patches_indexes = [0]
        init_area_total = []
        Big_patch_center_total = []
        self.spotforming_times = 0
        ### select the power threshold relatievly or absolutely
        if USE_RELATIVE_SPOT_POWER:
            SPOT_POWER_THRESHOLD_new = min([SPOT_POWER_THRESHOLD2, self.Relative_Threshold])
        else:
            SPOT_POWER_THRESHOLD_new = SPOT_POWER_THRESHOLD2    
        

        ### step 3.1 divide patch and apply spotforming on each small patches ####
        for i in range(len(candidate_finished)):
            patch_processed = search_area([candidate_finished[i]], self.mic_positions, self.upper_bound_pairwise)
            init_area = candidate_finished[i].area_points
            init_area_total.append(init_area)
            
            patch_center0 = Patch(candidate_finished[i].sample_offset, width_list0, None, candidate_finished[i].peak_pos)  
            Big_patch_center = patch_center0.center_pos()
            Big_patch_center_total.append(Big_patch_center)

            if Big_patch_center is not None:  
                patch_processed.append(patch_center0)
            else:
                print("it is impossible to be here")
            
            
            self.spotforming_times += len(patch_processed) 
            total_patch.extend(patch_processed)
            patches_indexes.append(self.spotforming_times)
        sep_data_total = spot_model.shift_and_sep(mix_data, total_patch, Strict = 1)

        ######################
        
        for i in range(len(patches_indexes) - 1):
            big_offset = candidate_finished[i].sample_offset
            big_label = -1
            if sample_gt is not None:
                for k in range(sample_gt.shape[1]):
                    delta_offset = big_offset - sample_gt[:, k]         
                    if np.amax(np.abs(delta_offset)) < 3.5:
                        big_label = k
                        break

            sep_data = sep_data_total[patches_indexes[i] : patches_indexes[i + 1]]
            patch_processed = total_patch[patches_indexes[i] : patches_indexes[i + 1]]
            init_area = init_area_total[i]
            Big_patch_center = Big_patch_center_total[i]
            
            powers = []
            powers2 = []     

            if self.visual_save:
                print("------------------------")
                print(i)

            ### calculate the power for each output
            for j in range(len(patch_processed)):
                sep_data[j,:] = sep_data[j,:] - np.mean(sep_data[j,:])
                power = np.sum(sep_data[j,:]**2)
                power2, _ = max_avg_power(sep_data[j,:])
                powers.append(power)
                powers2.append(power2)
                
                if run_demo_folder is not None:
                    sample_offsets = patch_processed[j].sample_offset
                    center = patch_processed[j].center_pos()
                    # print(j, center, power, power2,sample_offsets)
                    debug_pos.append(center)
                    debug_power.append(power)
                ### this is for debugging ###################################
                if self.visual_save:
                    sample_offsets = patch_processed[j].sample_offset
                    width_list =  patch_processed[j].width_list
                    center = patch_processed[j].center_pos()
                    size0 = patch_processed[j].area_size()
                    min_difference = 99999
                    min_delta = None
                    min_voice_id = -1
                    for k in range(self.monitor.sample_offsets_gt.shape[1]):
                        delta_offset = sample_offsets - self.monitor.sample_offsets_gt[:, k]         
                        difference = np.sum(np.abs(delta_offset))
                        if difference < min_difference:
                            min_difference = difference
                            min_delta = delta_offset.astype(int)
                            min_voice_id = k
                    # similarity = si_sdr(sep_data[j,:], audio_gt[min_voice_id, :])
                    debug_pos.append(center)
                    debug_power.append(power)
                    print(j, min_voice_id, size0, power, power2, min_delta, width_list)

                    # print(sample_offsets, self.monitor.sample_offsets_gt[:, min_voice_id]  )

                ###############################################################
            
            if candidate_finished[i].center_pos().shape[0] == 3:
                d = np.linalg.norm(candidate_finished[i].center_pos() - self.mic_positions[0])
            else:
                d = 4         
            max_power = np.amax(powers2)
            if max_power < SPOT_POWER_THRESHOLD_new / (1 + d):
                if self.visual_save:
                    print("all discarded !!!")
                continue
            
            ### step 3.2 apply the clustering on the small patches within a big Patch ####
            sort_idx = np.argsort(-1*np.array(powers)) #power2
            SI_SDR_THRESHOLD = -4
            
            clusters = {}
            MIN_TRIGGER_POWER2 = self.MIN_TRIGGER_POWER/(3*48000)*sep_data[j,:].shape[0]
            for _id in sort_idx:
                unique = True
                d = np.linalg.norm(patch_processed[_id].center_pos() - self.mic_positions[0])
                threshold = SPOT_POWER_THRESHOLD_new / (1 + d)
                if powers2[_id] < threshold or powers[_id] < MIN_TRIGGER_POWER2:
                    continue 
                
                for cluster_id in clusters:
                    final_candidate_id = clusters[cluster_id][0]
                    similarity = si_sdr(sep_data[_id, :], sep_data[final_candidate_id])
                    if similarity > SI_SDR_THRESHOLD:
                        clusters[final_candidate_id].append(_id)
                        unique = False
                        break
                if unique:
                    clusters[_id] = [_id]             
            if self.visual_save:
                print("cluster number is ", len(clusters.keys()))
            if len(clusters.keys()) <= 0 :
                continue


            ### calulte the property of each cluster and merge the information with the nearby patches
            for cluster_id in clusters:
                position, offests =  weight_mean_pos(patch_processed, powers, clusters[cluster_id])

                patch_center = find_merge_center(offests, init_area, self.mic_positions, Big_patch_center)
                if patch_center.center_pos() is None:
                    print("Warning some bug happen one source may be drop")
                ### patch center is the things which merge the close small patches by weight_mean_pos function
                save_offsets = {
                    "audio_offset": patch_processed[cluster_id].sample_offset, 
                    "localization_offset": offests 
                }# separation offset and the localization offset
                
                save_audio = sep_data[cluster_id, :]
                save_power = powers[cluster_id]
                pair = (patch_center, save_audio, save_power, str(i) + '_' + str(cluster_id), save_offsets, big_label)

                output_pair.append(pair)

                if self.visual_save:
                    print(str(i) + '_' +  str(cluster_id) , patch_center.center_pos(), len(output_pair))
                    utils.write_audio_file( self.monitor.Output_dir + "/debug/cluster" + str(i) + '_' +  str(cluster_id) +  ".wav", save_audio, sr=48000)

        if self.visual_save and len(debug_pos) > 0:
            visualize_small_patch(self.mic_positions, self.monitor.voice_positions, self.Range_spk, debug_pos, debug_power)
            plt.savefig(self.monitor.Output_dir + "/Spotforming_power.png") 
        if run_demo_folder is not None:
            visualize_small_patch(self.mic_positions, run_demo_folder["pos_gt"], self.Range_spk, debug_pos, debug_power)
            plt.savefig(run_demo_folder["curr_writing_dir"] + "/Spotforming_power.png") 
        return output_pair



    def Clustering_new(self, output_pair,  simple_pos = None, sample_gt = None): 
        #### ### step 3.3 apply the non-max supression fto remove false positive ####
        SI_SDR_THRESHOLD = -1
        candidates = sorted(output_pair, key=lambda x: -x[2])
        clusters = {}
        wrong_spotforming = []
        for _id in range(len(candidates)):
            belong_cluster = -1
            unique = True ## indicates whether the current candidates is generated from existing speaker
            ### chunking the audio signal and calculate power and sisnr separately
            sisnr_seg = []

            big_label = candidates[_id][-1]
            center1 = candidates[_id][0].center_pos()
            audio1 = candidates[_id][1]
            text1 = candidates[_id][3]
            if self.visual_save:
                print("*"*10, big_label)
            power1 = candidates[_id][2]            
            offset1 =  candidates[_id][4]["audio_offset"]

            seg_win = split_wav(audio1)
            if(len(seg_win) == 0):
                print("discard because no invalid split!!!")
                continue

            for cluster_id in clusters:
                final_candidate_id = clusters[cluster_id][0] ### the id for the cluster center
                
                audio2 = candidates[final_candidate_id][1]
                text2 = candidates[final_candidate_id][3]
                center2 = candidates[final_candidate_id][0].center_pos()
                
                similarity = si_sdr(audio1, audio2)
                sisdr_list = split_wise_sisdr(audio1, audio2, seg_win)
                sisnr_seg.append(sisdr_list)
                
                
                dis = np.linalg.norm(center1[:2] - center2[:2])
                check_valid = check_sisnr_win(sisdr_list)

                if self.visual_save:
                    print(text1, _id, text2, cluster_id, ', Similarity', similarity, 'Dis, ', dis)
                

                if (similarity > SI_SDR_THRESHOLD) or check_valid or dis < 0.45: 
                    clusters[final_candidate_id].append(_id)
                    unique = False
                    belong_cluster = cluster_id
                    belong_similarity = similarity
                    break
                    
            if len(sisnr_seg) != 0:
                sisnr_seg = np.array(sisnr_seg)
                # print(sisnr_seg)
                sisnr_seg = np.amax(sisnr_seg, axis = 0)
                if self.visual_save:
                    print(seg_win)
                    print(big_label, sisnr_seg)
                check_valid = check_sisnr_win(sisnr_seg, SISNR_THRESHOLD = -1, SISNR_THRESHOLD2 = -5) #-5
                if check_valid:
                    unique = False


            ## if find the current audio can be a new speaker 
            if unique:
                if self.visual_save:
                    num = len(clusters.keys())
                    print("++++++ new cluster",num,  _id)
                clusters[_id] = [_id]
            elif big_label >= 0 and sample_gt is not None and belong_cluster >= 0:
                final_candidate_id = clusters[belong_cluster][0]
                cluster_label = candidates[final_candidate_id][-1]
                power2 = candidates[final_candidate_id][2]
                offset1 = candidates[final_candidate_id][-2]["audio_offset"]#candidates[final_candidate_id][0].sample_offset
                delta_offset = offset1 - sample_gt[:, big_label]   
                delta_offset = delta_offset.astype(int)  
                if cluster_label == -1:    
                    wrong_spotforming.append((big_label, cluster_label, delta_offset, power1/power2))
        # maybe to deploy something here to fine tune the model with the elements in the cluster

        print("final speaker number is ", len(clusters.keys()))
        for big_label,cluster_label, _off, power_comp in wrong_spotforming:
            print("Wrong high-power spotforming = ", big_label, cluster_label, _off, power_comp)
        
        patch_final = []
        audio_final = []

        for cluster_id in clusters:
            final_candidate_id = clusters[cluster_id][0]
            patch_final.append(candidates[final_candidate_id])
            save_wav = candidates[final_candidate_id][1]
            audio_final.append(save_wav)

        if self.visual_save:
            for spk_id, wav in enumerate(audio_final):
                utils.write_audio_file(self.monitor.Output_dir + "/out_" +  str(spk_id) +  ".wav", wav, sr=48000)
            visualize_result(self.mic_positions, self.monitor.voice_positions, patch_final, simple_pos, Range_spk=self.Range_spk) 
            plt.savefig(self.monitor.Output_dir + "/final_loc.png")

        
        return audio_final, patch_final, self.big_spotforming_times + self.spotforming_times, wrong_spotforming

