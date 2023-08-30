import argparse
import os

import torch
import numpy as np
import cv2
from sep.training.JointModel.network import JointModel
from sep.eval.get_items import compute_metrics, get_items
import sep.helpers.utils as utils
from sep.helpers.eval_utils import si_sdr
from sep.helpers.constants import FS, SPEED_OF_SOUND
from asteroid.metrics import get_metrics
import json
import itertools


## this function is to match the our predictions and results
def find_best_permutation(wav_gt, wav_pred , pos_gt, pos_pred, acceptable_range=1, accept_sisdr=-15):
    num_gt = pos_gt.shape[0]
    num_pred = pos_pred.shape[0]
    n = max(num_gt, num_pred)

    neg_sisdr_matrix = np.ones((n, n))*10000
    dis_matrix = np.ones((n, n))*10000

    for i in range(num_gt):
        for j in range(num_pred):
            dis_matrix[i, j] = np.linalg.norm(pos_gt[i][:2] - pos_pred[j][:2])
            neg_sisdr_matrix[i, j] = -si_sdr(wav_pred[j], wav_gt[i])

    # print("neg_sisdr_matrix: ", neg_sisdr_matrix)
    # print("dis_matrix: ", dis_matrix)
    best_perm = None
    best_inliers = -1
    best_err = 10000

    for perm in itertools.permutations(range(n)):
        curr_inliers = 0
        loss_err = []
        paired = []
        for idx1, idx2  in enumerate(perm):
            neg_sisnr_err = neg_sisdr_matrix[idx1, idx2]
            dis_err = dis_matrix[idx1, idx2]
            ### best matching depends on both sample shift error and SISDR error
            loss =  neg_sisnr_err + dis_err
            if dis_err < acceptable_range and neg_sisnr_err < -accept_sisdr:
                curr_inliers += 1
                loss_err.append(loss)
                paired.append((idx2, idx1))  ### idx2 for output id, idx1 for ground-truth id
        if len(loss_err) > 0:
            curr_err = np.mean(loss_err) 
        else:
            curr_err = np.inf 
        if (curr_inliers > best_inliers) or (curr_inliers == best_inliers and curr_err < best_err):
            best_inliers = curr_inliers
            best_perm = paired
            best_err = curr_err
    # print("best_perm: ", best_perm)
    return best_perm

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
    
    #### calculate the speaker range
    Range_spk = metadata['ROI']
    Range_spk[-1] += 0.02

    return mics, mic_positions, sources, voice_positions, sample_offsets_gt, Range_spk



def main(args):
    # Choose device
    device = torch.device('cuda') if args.use_cuda else torch.device('cpu')
    args.device = device

    # Load model
    model = JointModel(args.spot_experiment_dir,
                       args.sep_experiment_dir,
                       use_spot_dataparallel=True,
                       use_fp16=args.use_fp16,
                       spot_batch_size=args.spot_batch_size,)
    model.to(device)
    model.eval()


    all_tp = []
    all_fn = []
    all_fp = []

    with torch.no_grad():
        # Go over each sample in the directories
        for sample_no in utils.list_top_level_directories(args.dataset):
            dir = os.path.join(args.dataset, sample_no)
            print(" ------------------- Running on sample ", dir)
            save_data = {}  # dict to save results json

            # Load metadata, multichannel mixture and ground truth signals from sample
            # Load microphone information, speaker information    
            metadata, mix, gt = get_items(dir, denoise_gt=True)
            mics, mic_positions, sources, gt_speaker_positions, sample_offsets_gt, speaker_range = preprocess_metadata(metadata)
            gt_num_speakers = gt_speaker_positions.shape[0]
            
            # Prepare model for inference
            model.setup(mic_positions=mic_positions, speaker_range=speaker_range, cached= args.cached_init,  cached_folder=dir)

            # Localize and separate speakers from mixture
            patches, audio_loc, audio, _, _, _ = model(mix)

            # Get estimated speaker positions
            est_positions = np.array([p[0].center_pos() for p in patches])
            est_offsets = [p[4]["localization_offset"] for p in patches]
            est_num_speakers = len(est_positions)

            
            gt = gt.cpu().numpy()
            # Evaluate
            Acceptable_range = 1
            perm = find_best_permutation(gt, audio, gt_speaker_positions, est_positions,acceptable_range = Acceptable_range)
            # print("perm: ", perm)

            save_data["mic_pos"] = mic_positions.tolist()
            save_data["speaker_pos"] = gt_speaker_positions.tolist()
            save_data["gt"] = []
            save_data["pred"] = []
            save_data["false_positive"] = []
            save_data["est_offsets"] = np.array(est_offsets).tolist()
            save_data["perm"] = perm
             
            n_gt = gt.shape[0]
            n_out = audio.shape[0]
            n_match = len(perm)
            
            tp = n_match
            fn = n_gt - n_match
            fp = n_out - n_match

            # Save true positives, false positives and false negatives
            all_tp.append(tp)
            all_fn.append(fn)
            all_fp.append(fp)
            
            # Reorder audio output according to permutation
            if len(perm) > 0:
                perm = np.array(perm)

                audio = audio[perm[:, 0]]
                audio_loc = audio_loc[perm[:, 0]]
                gt = gt[perm[:, 1]]
                reference_signal = mix[0:1].cpu().numpy().repeat(audio.shape[0], axis=0)


                # Compute SI-SDR from localization by separation model only 
                metrics_dict = get_metrics(mix=reference_signal[0],
                                        clean=gt,
                                        estimate=audio_loc,
                                        metrics_list=['si_sdr'],
                                        compute_permutation=False,
                                        sample_rate=48000, # sr shouldn't matter since we're only computing SI-SDR
                                        average=False)
                # Store results in list
                input_sisdr_old = []
                output_sisdr_old = []
                for i in range(gt.shape[0]):
                    input_sisdr_old.append(metrics_dict['input_si_sdr'][i][0])
                    output_sisdr_old.append(metrics_dict['si_sdr'][i])
                
                # Compute metrics for separation by localization model (inter-speaker attention)
                input_sdr, output_sdr, input_sisdr, output_sisdr = compute_metrics(reference_signal, audio, gt, permute=False)
                match_ids = perm.tolist() 
            else:
                input_sdr, output_sdr, input_sisdr, output_sisdr = [], [], [], []
                match_ids = perm

            ### save the info of ground-truth of sample
            for s in range(gt_num_speakers):
                spk_gt = {}
                spk_gt["sample"] =  sample_offsets_gt[:, s].tolist()
                spk_gt["pos"] =  gt_speaker_positions[s, :].tolist()
                save_data["gt"].append(spk_gt)

            ### save the info of predicted speakers
            check_idx = [i for i in range(len(patches))] ### all outputs from our model
            pred_pos = np.zeros_like(gt_speaker_positions)
            
            i = 0
            ### discard the matching ones and the remaining check_ids is the false positive
            for out_id, s in match_ids:
                spk_pred = {}
                check_idx.remove(out_id)
                shift_pred = est_offsets[out_id]
                sample_gt = sample_offsets_gt[:, s]

                pos_pred = est_positions[out_id]
                pos_gt = gt_speaker_positions[s]

                pred_pos[s, :] = pos_pred
                

                spk_pred["voice_id"] = s
                spk_pred["shifts"] = shift_pred.tolist()
                spk_pred["pos"] = pos_pred.tolist()
                

                ### localization error
                sample_err = np.mean(abs(shift_pred - sample_gt)) ## sample shift distance
                pos_err = np.linalg.norm(pos_pred[:2] - pos_gt[:2])
                spk_pred["sample_err"] = sample_err
                spk_pred["dis_err"] = pos_err


                ### separation error
                spk_pred["si_snr_in_mir"] = input_sdr[i]
                spk_pred["si_snri_mir"] = output_sdr[i] - input_sdr[i]
                # Save SI-SDR for separation by localization model (inter-speaker attention)
                spk_pred["si_snr_in"] = input_sisdr[i]
                spk_pred["si_snri"] = output_sisdr[i] - input_sisdr[i]

                # Save SI-SDR from localization by separation model only 
                spk_pred["si_snr_in_old"] = input_sisdr_old[i]
                spk_pred["si_snri_old"] = output_sisdr_old[i] - input_sisdr_old[i]

                save_data["pred"].append(spk_pred)
                i += 1
        

            ###  save the remaining speakers as false positive
            for remain_id in check_idx:
                pos_pred = est_positions[remain_id]
                spk_FP = {}
                spk_FP["pos"] = pos_pred.tolist()
                spk_FP["sample"] = patches[remain_id][4]["audio_offset"].tolist()
                save_data["false_positive"].append(spk_FP)

            if args.results_folder is not None:
                os.makedirs(args.results_folder, exist_ok=True)
                metadata_file =  args.results_folder + "/result_" + sample_no + ".json"
                with open(metadata_file, "w") as f:
                    json.dump(save_data, f, indent=4)
                    
            print("False positive = {}; False negative = {}; True positive = {}".format(fp, fn, tp ))
            print("input_sisdr = ", input_sisdr)
            print("output_sisdr = ", output_sisdr)
            
        print("Overall tp: {}, fp: {}, fn: {}".format(sum(all_tp), sum(all_fp), sum(all_fn)))
        print("Overall Precision: {} Recall: {}".format(
            sum(all_tp) / (sum(all_tp) + sum(all_fp)),
            sum(all_tp) / (sum(all_tp) + sum(all_fn))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str,
                        help="Path to the dataset directory")
    parser.add_argument('--spot_experiment_dir',
                        type=str,
                        help='Experiment directory for spotforming submodel')
    parser.add_argument('--sep_experiment_dir',
                        type=str,
                        help='Experiment directory for separation submodel')
    parser.add_argument('--sr',
                        type=float,
                        help="Sampling rate",
                        default=48000)
    parser.add_argument('--n_mics',
                        type=int,
                        help="Number of microphones",
                        default=7)
    parser.add_argument('--use_cuda',
                        dest='use_cuda',
                        action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--spot_batch_size',
                        dest='spot_batch_size',
                        type=int,
                        default=128,
                        help="Batch size to use for spotforming")
    parser.add_argument('--cached_init',
                        action='store_true',
                        help="cached_init")
    parser.add_argument('--results_folder',
                        type=str,
                        default=None,
                        help="Path to save results.json")

    args = parser.parse_args()
    args.use_fp16 = False # Never used
    main(args)
