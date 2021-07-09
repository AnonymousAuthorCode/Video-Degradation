# python run_answer_quality_sum.py --obj_name car --base_name jackson-town-square \
# --test_date 2017-12-17 --ground_truth freq002_res608_yolo --ground_truth_freq 0.02 --predictions freq002_res608_yolo --sample_frac 0.02 --constraints freq002_res608_yolo  (--cons_obj_name person) --use_val False --val_frac 1 --our_alg True
import argparse

import numpy as np
import pandas as pd
import scipy.stats
from samplers_sum import *

import sys
sys.path.append('../')

from data.generate_fnames import get_csv_fname

def get_data(base_name, date, obj_name, data_path, ground_truth, predictions):
    # get true data (validation dataset)
    csv_fname = get_csv_fname(data_path, base_name, date, ground_truth)
    df = pd.read_csv(csv_fname)
    df = df[df['object_name'] == obj_name]
    true_idx = df.groupby('frame').size()
    vals = np.zeros(true_idx.index.max() + 1)
    for idx in true_idx.index:
        vals[idx] = true_idx.at[idx]

    # get predicted data (after running NN model)
    csv_fname = get_csv_fname(data_path, base_name, date, predictions)
    df = pd.read_csv(csv_fname)
    df = df[df['object_name'] == obj_name]
    pred_idx = df.groupby('frame').size()
    preds = np.zeros(true_idx.index.max() + 1)
    for idx in pred_idx.index:
        preds[idx] = pred_idx.at[idx]

    return preds, vals

def restrict_frame(base_name, date, data_path, constraints, cons_obj_name, Y_len):
    csv_fname = get_csv_fname(data_path, base_name, date, constraints)
    df = pd.read_csv(csv_fname)
    if(cons_obj_name == None):
        rest_frame = np.zeros(Y_len)
    else:
        rest_frame = np.zeros(Y_len)
        df = df[df['object_name'] == cons_obj_name]
        df = df.groupby('frame').size()
        for idx in df.index:
            rest_frame[idx] = 1

    return rest_frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../Data/')
    parser.add_argument('--obj_name', required=True)
    parser.add_argument('--base_name', required=True)
    parser.add_argument('--test_date', required=True)
    parser.add_argument('--ground_truth', required=True)
    parser.add_argument('--ground_truth_freq', type=float, required=True)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--sample_frac', type=float, required=True)
    parser.add_argument('--constraints', required=True)
    parser.add_argument('--cons_obj_name', default=None)
    parser.add_argument('--use_val', required=True)
    parser.add_argument('--val_frac', type=float, required=True)
    parser.add_argument('--our_alg', required=True)
    args = parser.parse_args()

    data_path = args.data_path
    obj_name = args.obj_name
    base_name = args.base_name
    test_date = args.test_date
    ground_truth = args.ground_truth
    predictions = args.predictions
    sample_frac = args.sample_frac
    cons_obj_name = args.cons_obj_name
    ground_truth_freq = args.ground_truth_freq
    constraints = args.constraints
    use_val = args.use_val
    val_frac = args.val_frac
    our_alg = args.our_alg

    assert sample_frac <= 1

    Y_pred, Y_val = get_data(base_name, test_date, obj_name, data_path, ground_truth, predictions)
    rest_frame = restrict_frame(base_name, test_date, data_path, constraints, cons_obj_name, len(Y_val))

    print("This code is to compute the answer quality of SUM under different interventions")
    print('True length {}, true sum {}'.format(len(Y_val)*ground_truth_freq, np.sum(Y_val)))
    print('predict sum {}'.format(np.sum(Y_pred)))
    conf = 0.05
    nb_trials = 100 
    sample_size = int(sample_frac * len(Y_val)*ground_truth_freq)
    val_size = int(val_frac * len(Y_val)*ground_truth_freq)

    if(our_alg == 'True'):
        sampler = TrueSampler(sample_size, conf, Y_pred, Y_val, ground_truth_freq, rest_frame, use_val, val_size)
    elif(our_alg == 'EBGS'):
        sampler = TrueSamplerEBGS(sample_size, conf, Y_pred, Y_val, ground_truth_freq, rest_frame, use_val, val_size)
    elif(our_alg == 'Hoeffding'):
        sampler = TrueSamplerHoeffding(sample_size, conf, Y_pred, Y_val, ground_truth_freq, rest_frame, use_val, val_size)
    elif(our_alg == 'CLT'):
        sampler = TrueSamplerCLT(sample_size, conf, Y_pred, Y_val, ground_truth_freq, rest_frame, use_val, val_size)
    elif(our_alg == 'HoeffdingSerfling'):
        sampler = TrueSamplerHoeffdingSerfling(sample_size, conf, Y_pred, Y_val, ground_truth_freq, rest_frame, use_val, val_size)
    else:
        print("Choose a correct sampler!")
    total_true_err = 0
    total_estimate_err = 0
    
    for i in range(nb_trials):
        estimate, estimate_err = sampler.sample()
        true_err = np.abs(estimate - np.sum(Y_val))/np.sum(Y_val)
        if(estimate_err >= 0 and estimate_err < true_err):
            print("WARNING! Not an upper bound")
        total_true_err += true_err
        total_estimate_err += estimate_err
    
    print("Averaged TRUE fraction error, averaged ESTIMATED fraction error:")
    print(total_true_err / nb_trials, total_estimate_err / nb_trials)

if __name__ == '__main__':
    main()
