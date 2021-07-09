import math
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression


class Sampler(object):
    def __init__(self, sample_size, conf, Y_pred, Y_val, ground_truth_freq, rest_frame, use_val, val_size):
        self.sample_size = sample_size
        self.conf = conf
        self.Y_pred = Y_pred
        self.Y_val = Y_val
        self.ground_truth_freq = ground_truth_freq
        self.use_val = use_val
        self.rest_frame = rest_frame
        self.val_size = val_size

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

    # Revised EBS + Hoeffding-serfling
    def sample(self):
        Y_pred = self.Y_pred
        Y_val = self.Y_val

        permute_index = self.permute(len(Y_val), self.ground_truth_freq, self.rest_frame)
        permute_index_val = self.permute(len(Y_val), self.ground_truth_freq, np.zeros(len(Y_val)))

        LB = 0
        UB = 10000000
        t = 1
        total_size = len(Y_val)*self.ground_truth_freq
        rou = min(1-(self.sample_size-1)/total_size, (1-self.sample_size/total_size)*(1+1/self.sample_size))

        Xt_sum = self.get_sample(Y_pred, permute_index, t)
        R = Xt_sum
        while t < self.sample_size:
            t += 1
            sample = self.get_sample(Y_pred, permute_index, t)
            Xt_sum += sample
            R = max(R, sample)

        Xt = Xt_sum / t
        ct = R * np.sqrt(rou*np.log(2/self.conf) / (2*self.sample_size))
        # ct2 = sigmat * np.sqrt(2*rou*np.log(5/self.conf)/t) + (7/3+3/np.sqrt(2))*R*np.log(5/self.conf)/t
        LB = max(0,np.abs(Xt) - ct)
        UB = np.abs(Xt) + ct

        err_Xe = (UB - LB) / (UB + LB)
        estimate_Xe = np.sign(Xt) * 0.5 * ((1 + err_Xe) * LB + (1 - err_Xe) * UB)

        if(self.use_val == 'False'):
            estimate = estimate_Xe
            err = err_Xe
        else:
            LB = 0
            UB = 10000000
            t = 1
            total_size = len(Y_val)*self.ground_truth_freq
            rou = min(1-(self.val_size-1)/total_size, (1-self.val_size/total_size)*(1+1/self.val_size))

            Xt_sum = self.get_sample(Y_val, permute_index_val, t)
            R = Xt_sum
            while t < self.val_size:
                t += 1
                sample = self.get_sample(Y_val, permute_index_val, t)
                Xt_sum += sample
                R = max(R, sample)

            Xt = Xt_sum / t
            ct = R * np.sqrt(rou*np.log(2/self.conf) / (2*self.val_size))
            LB = max(0,np.abs(Xt) - ct)
            UB = np.abs(Xt) + ct

            err_X = (UB - LB) / (UB + LB)
            estimate_X = np.sign(Xt) * 0.5 * ((1 + err_X) * LB + (1 - err_X) * UB)

            estimate = estimate_Xe
            err = (1 + err_X) * np.abs(estimate_Xe - estimate_X) / np.abs(estimate_X) + err_X

        return estimate, err

# Restrict the sample size and use EBGS to estimate the relative error
class TrueSamplerEBGS(Sampler):
    def get_sample(self, Y, permute_index, nb_samples):
        return Y[permute_index[nb_samples-1]]

    # EBGS
    def sample(self):
        Y_pred = self.Y_pred
        Y_val = self.Y_val

        permute_index = self.permute(len(Y_val), self.ground_truth_freq, self.rest_frame)

        LB = 0
        UB = 10000000
        t = 1
        k = 1
        beta = 1.5
        p = 1.1
        c = self.conf * (p - 1) / p
        R = np.max(Y_pred[permute_index[:self.sample_size]])

        Xt_sum = self.get_sample(Y_pred, permute_index, t)
        Xt_sqsum = Xt_sum * Xt_sum
        while t < self.sample_size:
            t += 1
            if t > np.floor(beta ** k):
                k += 1
                alpha = np.floor(beta ** k) / np.floor(beta ** (k - 1))
                dk = c / (k**p) # correct formula according to the paper
                x = -alpha * np.log(dk/3)

            sample = self.get_sample(Y_pred, permute_index, t)
            Xt_sum += sample
            Xt_sqsum += sample * sample
            Xt = Xt_sum / t
            sigmat = np.sqrt(1/t * (Xt_sqsum - Xt_sum ** 2 / t))
            # # Finite sample correction
            # sigmat *= np.sqrt((len(Y_true) - t) / (len(Y_true) - 1))

            ct = sigmat * np.sqrt(2 * x / t) + 3 * R * x / t
            LB = max(LB, np.abs(Xt) - ct)
            UB = min(UB, np.abs(Xt) + ct)

        err = (UB - LB) / (UB + LB)
        estimate = np.sign(Xt) * 0.5 * ((1 + err) * LB + (1 - err) * UB)
        return estimate, err

class TrueSamplerHoeffding(Sampler):
    def get_sample(self, Y, permute_index, nb_samples):
        return Y[permute_index[nb_samples-1]]

    # Hoeffding
    def sample(self):
        Y_pred = self.Y_pred
        Y_val = self.Y_val

        permute_index = self.permute(len(Y_val), self.ground_truth_freq, self.rest_frame)

        t = 1
        R = np.max(Y_pred[permute_index[:self.sample_size]])

        Xt_sum = self.get_sample(Y_pred, permute_index, t)
        while t < self.sample_size:
            t += 1
            sample = self.get_sample(Y_pred, permute_index, t)
            Xt_sum += sample
            Xt = Xt_sum / t
        
        estimate = Xt
        abs_err = R * np.sqrt(np.log(2/self.conf) / (2*self.sample_size))
        err = abs_err / (estimate-abs_err)
        return estimate, err

class TrueSamplerCLT(Sampler):
    def get_sample(self, Y, permute_index, nb_samples):
        return Y[permute_index[nb_samples-1]]

    # central limit theorem
    def sample(self):
        Y_pred = self.Y_pred
        Y_val = self.Y_val

        permute_index = self.permute(len(Y_val), self.ground_truth_freq, self.rest_frame)

        t = 1
        if(self.conf == 0.05):
            z_score = 1.96

        Xt_sum = self.get_sample(Y_pred, permute_index, t)
        Xt_sqsum = Xt_sum * Xt_sum
        while t < self.sample_size:
            t += 1
            sample = self.get_sample(Y_pred, permute_index, t)
            Xt_sum += sample
            Xt_sqsum += sample * sample
            Xt = Xt_sum / t
        T_n = 1/(t-1) * (Xt_sqsum - Xt_sum ** 2 / t)
        
        estimate = Xt
        abs_err = np.sqrt(z_score**2*T_n / self.sample_size)
        err = abs_err / (estimate-abs_err)
        return estimate, err

class TrueSamplerHoeffdingSerfling(Sampler):
    def get_sample(self, Y, permute_index, nb_samples):
        return Y[permute_index[nb_samples-1]]

    # Hoeffding Serfling inequality
    def sample(self):
        Y_pred = self.Y_pred
        Y_val = self.Y_val

        permute_index = self.permute(len(Y_val), self.ground_truth_freq, self.rest_frame)

        t = 1
        R = np.max(Y_pred[permute_index[:self.sample_size]])
        total_size = len(Y_val)*self.ground_truth_freq
        rou = min(1-(self.sample_size-1)/total_size, (1-self.sample_size/total_size)*(1+1/self.sample_size))

        Xt_sum = self.get_sample(Y_pred, permute_index, t)
        while t < self.sample_size:
            t += 1
            sample = self.get_sample(Y_pred, permute_index, t)
            Xt_sum += sample
            Xt = Xt_sum / t
        
        estimate = Xt
        abs_err = R * np.sqrt(rou*np.log(2/self.conf) / (2*self.sample_size))
        err = abs_err / (estimate-abs_err)
        return estimate, err