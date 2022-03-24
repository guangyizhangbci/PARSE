from __future__ import print_function
import os
from scipy.io import loadmat
from scipy import signal
import matplotlib
import scipy.io as sio
import numpy as np
import math
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simps

path_amigos = './PARSE/DATA/AMIGOS/Data_Preprocessed_P{}/Data_Preprocessed_P{}.mat'

fs = 128


def preprocessing_normal(data, feature):
    freqs, psd = signal.periodogram(data, fs, scaling='density')
    # Define delta lower and upper limits
    low, high = 3, 7
    idx_min = np.argmax(freqs > low) - 1
    idx_max = np.argmax(freqs > high) - 1
    idx_delta = np.zeros(dtype=bool, shape=freqs.shape)
    idx_delta[idx_min:idx_max] = True


    delta_power = simps(psd[idx_delta], freqs[idx_delta])
    total_power = simps(psd, freqs)
    delta_rel_power = delta_power / (total_power)  # to avoid bad channels zeroing total_power
    # Define theta lower and upper limits
    low, high = 8, 10
    idx_min = np.argmax(freqs > low) - 1
    idx_max = np.argmax(freqs > high) - 1
    idx_theta = np.zeros(dtype=bool, shape=freqs.shape)
    idx_theta[idx_min:idx_max] = True
    #print('idx_max', idx_max)
    theta_power = simps(psd[idx_theta], freqs[idx_theta])
    theta_rel_power = theta_power / (total_power)
    # Define alpha lower and upper limits
    low, high = 8, 13
    idx_min = np.argmax(freqs > low) - 1
    idx_max = np.argmax(freqs > high) - 1
    idx_alpha = np.zeros(dtype=bool, shape=freqs.shape)
    idx_alpha[idx_min:idx_max] = True
    alpha_power = simps(psd[idx_alpha], freqs[idx_alpha])
    alpha_rel_power = alpha_power / (total_power)
    # Define beta lower and upper limits
    low, high = 14, 29
    idx_min = np.argmax(freqs > low) - 1
    idx_max = np.argmax(freqs > high) - 1
    idx_beta = np.zeros(dtype=bool, shape=freqs.shape)
    idx_beta[idx_min:idx_max] = True
    beta_power = simps(psd[idx_beta], freqs[idx_beta])
    beta_rel_power = beta_power / (total_power)

    low, high = 30, 45
    idx_min = np.argmax(freqs > low) - 1
    idx_max = np.argmax(freqs > high) - 1
    idx_gamma = np.zeros(dtype=bool, shape=freqs.shape)
    idx_gamma[idx_min:idx_max] = True
    gamma_power = simps(psd[idx_gamma], freqs[idx_gamma])
    gamma_rel_power = gamma_power / (total_power)

    feature = [delta_rel_power, theta_rel_power, alpha_rel_power, beta_rel_power, gamma_rel_power]

    return feature


'''AMIGOS specific: theta (3-7 Hz), slow alpha (8-10 Hz), alpha (8-13 Hz), beta(14-29 Hz) and gamma (30-47 Hz)'''

def preprocessing_amigos(raw, feature):
    overall    = signal.firwin(9,[0.046875, 0.734375],   window='hamming')
    theta      = signal.firwin(9,[0.046875, 0.109375],   window='hamming')
    slow_alpha = signal.firwin(9,[0.125,    0.15625],    window='hamming')
    alpha      = signal.firwin(9,[0.125,    0.203125],   window='hamming')
    beta       = signal.firwin(9,[0.21875,  0.453125],   window='hamming')
    gamma      = signal.firwin(9,[0.46875,  0.734375],   window='hamming')

    # overall    = signal.firwin(9,[3, 47],   window='hamming', fs=128)
    # theta      = signal.firwin(9,[3, 7],   window='hamming', fs=128)
    # slow_alpha = signal.firwin(9,[8, 10],    window='hamming', fs=128)
    # alpha      = signal.firwin(9,[8, 13],   window='hamming', fs=128)
    # beta       = signal.firwin(9,[14,29],   window='hamming', fs=128)
    # gamma      = signal.firwin(9,[30,47],   window='hamming', fs=128)


    filted_Data        = signal.filtfilt(overall,   1, raw)
    filted_theta       = signal.filtfilt(theta,     1, filted_Data)
    filted_slow_alpha  = signal.filtfilt(slow_alpha,1, filted_Data)
    filted_alpha       = signal.filtfilt(alpha,     1, filted_Data)
    filted_beta        = signal.filtfilt(beta,      1, filted_Data)
    filted_gamma       = signal.filtfilt(gamma,     1, filted_Data)


    _,psd_theta        = signal.welch(filted_theta,      fs=128, nperseg=128)
    _,psd_slow_alpha   = signal.welch(filted_slow_alpha, fs=128, nperseg=128)
    _,psd_alpha        = signal.welch(filted_alpha,      fs=128, nperseg=128)
    _,psd_beta         = signal.welch(filted_beta,       fs=128, nperseg=128)
    _,psd_gamma        = signal.welch(filted_gamma,      fs=128, nperseg=128)

    # offset = 0.001
    # if math.isnan(simps(psd_theta)) or math.isnan(simps(psd_gamma)) or math.isnan(simps(psd_alpha)) or math.isnan(simps(psd_beta)):
    #     exit(0)

    feature.append(simps(psd_theta))
    feature.append(simps(psd_slow_alpha))
    feature.append(simps(psd_alpha))
    feature.append(simps(psd_beta))
    feature.append(simps(psd_gamma))

    # feature.append(psd_theta.mean())
    # feature.append(psd_slow_alpha.mean())
    # feature.append(psd_alpha.mean())
    # feature.append(psd_beta.mean())
    # feature.append(psd_gamma.mean())

    return np.log10(feature)



def EEG_segment(signal):  #14 EEG channels

    first_segment = signal[:20*fs,:]
    last_segment = signal[-20*fs:,:]

    '''start from 5s of signal, 20s per segment'''

    segments_len = math.floor((len(signal)-5*fs)/(20*fs))

    segments = np.zeros((segments_len, 20*fs, 14))

    for i in range(segments_len):
        segments[i] = signal[(i+5)*fs:(i+5+20)*fs, :]

    first_segment  = np.expand_dims(first_segment, axis=0)
    last_segment   = np.expand_dims(last_segment, axis=0)

    total_segments = np.vstack((first_segment, segments, last_segment))

    return total_segments


#
def labeling_EEG(participant):#(file_name_csv,raw):

    raw = loadmat(path_amigos.format(participant, participant))

    # labels = np.zeros((18, 2))

    '''labeling'''

    valence_list, arousal_list = [], []
    threshold = 0.0

    for video in range(20):

        valence = raw['labels_ext_annotation'][0,video][:,1].astype(float)>(threshold)
        arousal = raw['labels_ext_annotation'][0,video][:,2].astype(float)>(threshold)

        valence_list =valence_list + valence.tolist()
        arousal_list =arousal_list + arousal.tolist()


    valence_arr =np.expand_dims(np.array(valence_list).astype(int), 1)
    arousal_arr =np.expand_dims(np.array(arousal_list).astype(int), 1)


    labels_arr = np.hstack((valence_arr, arousal_arr))


    '''data_processing'''

    EEG_segments_temp = np.zeros((0, 20*fs, 14))
    for video in range(20):
        EEG = raw['joined_data'][0,video][:,0:14]
        EEG_segments = EEG_segment(EEG)
        EEG_segments_temp = np.vstack((EEG_segments_temp, EEG_segments))


    '''feature_extraction'''

    EEG_segment_total = np.transpose(EEG_segments_temp, (0, 2, 1))

    total_features_arr = np.zeros((len(EEG_segment_total), 105))

    for segment_num in range(len(EEG_segment_total)):

        features_list = np.zeros((14, 5))
        for channel_num in range(14):
            signal  = EEG_segment_total[segment_num, channel_num]
            feature = []
            features_list[channel_num] = preprocessing_amigos(signal, feature)
            # if any(math.isnan(i) for i in features_list[channel_num]):
            #     print(segment_num, signal)
            #
            #     exit(0)
            # plt.plot((features_list[channel_num]))
            # plt.show()

            asymmetric_features = np.zeros((7, 5))
            for asy_channel_num in range(7):

                asymmetric_features[asy_channel_num] = features_list[asy_channel_num] - features_list[-(asy_channel_num+1)]

        total_features_arr[segment_num] = features_list.flatten().tolist() + asymmetric_features.flatten().tolist()


    return total_features_arr, labels_arr



exclude_list = [8, 24, 28]
# [print(i) for i in exclude_list]

for participant in tqdm(range(1, 41)):
    if not any(participant == c for c in exclude_list):
        print(participant)
        features, labels = labeling_EEG(participant)
        np.save('./PARSE/DATA/AMIGOS/EEG/psd/{}.npy'.format(participant), features)
        np.save('./PARSE/DATA/AMIGOS/EEG/label/{}.npy'.format(participant), labels)





































#
