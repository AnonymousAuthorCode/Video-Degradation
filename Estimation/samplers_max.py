import math
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression


class Sampler(object):
    def __init__(self, sample_size, conf, Y_pred, Y_val, ground_truth_freq, rest_frame, quantile, use_val, val_size):
        self.sample_size = sample_size
        self.conf = conf
        self.Y_pred = Y_pred
        self.Y_val = Y_val
        self.ground_truth_freq = ground_truth_freq
        self.use_val = use_val
        self.rest_frame = rest_frame
        self.val_size = val_size
        self.quantile = quantile

    def permute(self, Y_len, ground_truth_freq, rest_frame):
        index = []
        i = 0
        while int(i/ground_truth_freq) < Y_len:
            if(rest_frame[int(i/ground_truth_freq)] == 0):
                index.append(int(i/ground_truth_freq))
            i += 1
        index = np.array(index)
        permute_index = np.random.permutation(index)
    
        return permute_index


class TrueSampler(Sampler):
    def get_sample(self, Y, permute_index, nb_samples):
        return Y[permute_index[nb_samples-1]]

    # Revised Approximate holistic aggregation
    def sample(self):
        Y_pred = self.Y_pred
        Y_val = self.Y_val

        permute_index = self.permute(len(Y_val), self.ground_truth_freq, self.rest_frame)
        permute_index_val = self.permute(len(Y_val), self.ground_truth_freq, np.zeros(len(Y_val)))

        samples = []
        t = 0
        while t < self.sample_size:
            t += 1
            samples.append(self.get_sample(Y_pred, permute_index, t))
        samples = np.array(samples)
        samples_sort = np.sort(samples)

        if(self.conf == 0.05):
            z_score = 1.96
        m = self.sample_size
        n = int(len(Y_val)*self.ground_truth_freq) + 1

        estimate_Xe = samples_sort[int(self.quantile*m)]

        pre_index = int(self.quantile*m)
        post_index = int(self.quantile*m)
        while(samples_sort[pre_index] == estimate_Xe and pre_index != 0):
            pre_index -= 1
        if(samples_sort[pre_index] != estimate_Xe):
            pre_index += 1
        while(samples_sort[post_index] == estimate_Xe and post_index != m-1):
            post_index += 1
        if(samples_sort[post_index] != estimate_Xe):
            post_index -= 1
        f_k = (post_index - pre_index + 1) / m
        f_max = f_k
        f_min = f_k

        err_Xe = ((z_score*np.sqrt(self.quantile*(1-self.quantile))*np.sqrt((n-m)/(m*(n-1)))+f_k)/f_min + 1) * f_max / self.quantile

        if(self.use_val == 'False'):
            estimate = estimate_Xe
            err = err_Xe
        else:
            samples = []
            t = 0
            while t < self.val_size:
                t += 1
                samples.append(self.get_sample(Y_val, permute_index_val, t))
            samples = np.array(samples)
            samples_sort = np.sort(samples)

            if(self.conf == 0.05):
                z_score = 1.96
            m = self.val_size
            n = int(len(Y_val)*self.ground_truth_freq) + 1

            estimate_X = samples_sort[int(self.quantile*m)]

            pre_index = int(self.quantile*m)
            post_index = int(self.quantile*m)
            while(samples_sort[pre_index] == estimate_X and pre_index != 0):
                pre_index -= 1
            if(samples_sort[pre_index] != estimate_X):
                pre_index += 1
            while(samples_sort[post_index] == estimate_X and post_index != m-1):
                post_index += 1
            if(samples_sort[post_index] != estimate_X):
                post_index -= 1
            f_k = (post_index - pre_index + 1) / m
            f_max = f_k
            f_min = f_k

            err_X = ((z_score*np.sqrt(self.quantile*(1-self.quantile))*np.sqrt((n-m)/(m*(n-1)))+f_k)/f_min + 1) * f_max / self.quantile

            estimate = estimate_Xe
            Xe_index = 0
            while(samples_sort[Xe_index] <= estimate_Xe and Xe_index != m-1):
                Xe_index += 1
            err = np.abs(Xe_index/m - post_index/m) / self.quantile + err_X

        return estimate, err

class TrueSamplerStein(Sampler):
    def get_sample(self, Y, permute_index, nb_samples):
        return Y[permute_index[nb_samples-1]]

    # Revised Approximate holistic aggregation
    def sample(self):
        Y_pred = self.Y_pred
        Y_val = self.Y_val

        permute_index = self.permute(len(Y_val), self.ground_truth_freq, self.rest_frame)

        samples = []
        t = 0
        while t < self.sample_size:
            t += 1
            samples.append(self.get_sample(Y_pred, permute_index, t))
        samples = np.array(samples)
        samples_sort = np.sort(samples)

        m = self.sample_size
        n = int(len(Y_val)*self.ground_truth_freq) + 1

        estimate = samples_sort[int(self.quantile*m)]
        err = (np.log(1/self.conf)+1) / m
        err /= self.quantile

        return estimate, err