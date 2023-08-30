import argparse
import json
import numpy as np
import os 

import torch 
from sep.helpers.eval_utils import si_sdr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import sep.helpers.utils as utils
import scipy.stats as st
import glob


def draw_cdf(arr, fig_id = 0):
    plt.figure(fig_id)
    kwargs = {'cumulative': True}
    sns.distplot(arr, hist_kws=kwargs, kde_kws=kwargs)
    
def main(args):
    curr_dir = args.input_dir
    dis_err_list = []
    sample_err_list = []


    False_negative = 0
    False_positive = 0
    True_positive = 0
    False_negative_SRP  = 0


    False_positive_num = np.zeros((5, ))
    False_negative_num = np.zeros((5, ))
    True_positive_num = np.zeros((5, ))
    

    sisnri_numspk = [[], [], [], [], [], []]
    loc_err_numspk = [[], [], [], [], [], []]

    SI_SDR_list_improve = []
    SI_SDR_list_in = []
    mireval_list_improve = []
    mireval_list_in = []
    json_files = glob.glob(curr_dir + "/result*.json")

    for result_path in json_files:
        with open(result_path, 'r') as json_file:
            result_data = json.load(json_file)
        assert result_data, 'Something went wrong when reading scene metadata. Are you sure this file exists in the specified directory?'
        ### load the data for gt and pred
        gt_data = result_data["gt"] ### ground-truth sources
        pred_data = result_data["pred"] ### output sources
        remain_data = result_data["false_positive"] ### false positive sources
           
        real_num = 0

        for i, pred in enumerate(pred_data):
            real_num += 1
            SI_SDR_list_improve.append(pred["si_snri"])
            SI_SDR_list_in.append(pred["si_snr_in"])
            mireval_list_improve.append(pred["si_snri_mir"])
            mireval_list_in.append(pred["si_snr_in_mir"])
                        
            dis_err_list.append(pred["dis_err"])
            sample_err_list.append(pred["dis_err"])

            #### divide the results to different mixture number
            sisnri_numspk[len(gt_data) - 2].append(pred["si_snri"])
            loc_err_numspk[len(gt_data) - 2].append(pred["dis_err"])
       
        True_positive += real_num 
        False_negative += len(gt_data) - real_num
        miss_spk = len(gt_data) - real_num
        False_positive += len(remain_data)

        True_positive_num[len(gt_data) - 2] += real_num
        False_negative_num[len(gt_data) - 2] += len(gt_data) - real_num
        False_positive_num[len(gt_data) - 2] += len(remain_data)
 

        print("files: " , result_path)
        print("False positive = {}; False negative = {}; True positive = {}".format(len(remain_data), len(gt_data) - len(pred_data), len(pred_data) )   )
        if miss_spk > 0:
            print("miss: ", miss_spk)



    print("False positive = {}; False negative = {}; True positive = {}".format(False_positive, False_negative, True_positive  )   )
    precision = True_positive/(True_positive + False_positive)
    recall = True_positive/(True_positive + False_negative)
    recall2 = True_positive/(True_positive + False_negative_SRP)
   

    print(True_positive, False_positive, False_negative, False_negative_SRP)
    print("precision = {:.4f} and recall = {:.4f}, recall_SRP = {:.4f}".format(precision, recall, recall2))

    print("speaker number settings")
    for i in range(0, 6): ### 
        if len(sisnri_numspk[i]) <= 0:
            continue
        precision = True_positive_num[i]/(True_positive_num[i] + False_positive_num[i])
        recall = True_positive_num[i]/(True_positive_num[i] + False_negative_num[i])
        loc_err_num = np.mean(loc_err_numspk[i])
        loc_err_median = np.percentile(loc_err_numspk[i], 50)
        loc_err_90 = np.percentile(loc_err_numspk[i], 90)
        sisnr_num = np.mean(sisnri_numspk[i])
        print("speaker_num {:d} precision = {:.4f} and recall = {:.4f}, loc_err={:.3f}, sisnri={:.3f}".format(i+2, precision, recall,loc_err_num, sisnr_num) )
        print("median=", loc_err_median, "90%=", loc_err_90)

    print("avg dis err: ", np.mean(dis_err_list))
    print("median dis err: ", np.percentile(dis_err_list, 50))
    print("0.90 dis err: ", np.percentile(dis_err_list, 90))
    print("avg si-snr i : ", np.mean(SI_SDR_list_improve))
    print("avg mir_eval si-snr i: ", np.mean(mireval_list_improve))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help="Path to the input file")
    parser.add_argument('--sample_number',
                        type=int,
                        default=1,
                        help="Number of testing sample")
    main(parser.parse_args())
