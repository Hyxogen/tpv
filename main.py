#!/usr/bin/python
import numpy as np
import mne
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

mne.set_log_level(verbose='WARNING')

raw = mne.io.read_raw_edf('files/S001/S001R03.edf', preload=True)

# FILTER

lo_cut = 0.1
hi_cut = 30

raw_filt = raw.copy().filter(lo_cut, hi_cut)
raw_filt = raw_filt.notch_filter(freqs=50.0) # remove line noise


# EXTRACT FEATURES

sfreq = raw_filt.info["sfreq"]
channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]

ica = mne.preprocessing.ICA(method="infomax")

def get_features(epochs, lofreq, hifreq, epoch_type):
    feat_mat = []
    for idx, epoch in enumerate(epochs):
        mean = np.mean(epoch, axis=0)

        filtered = mne.filter.filter_data(epoch, method="iir", l_freq=lofreq, h_freq=hifreq,
                                 sfreq=sfreq)

        activation = filtered - mean

        energy = np.sum(activation ** 2, axis=1)
        power = energy / (len(epoch) * sfreq)

        event_type = epochs.events[idx][2]

        features = np.hstack((mean, energy, power, epoch_type, event_type))
        feat_mat.append(features)
    return feat_mat

def get_all_features(data):
    event_id = {"T1": 1, "T2": 2}
    events, event_dict = mne.events_from_annotations(raw_filt)

    erss = mne.Epochs(data, events, event_id=event_id, tmin=4.1, tmax=5.1,
                      baseline=None, picks=channels)
    mrcp = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=0,
                      baseline=None, picks=channels)
    erds = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=0.0,
                      baseline=None, picks=channels)

    ica.fit(erss)
    ica.fit(erds)
    ica.fit(mrcp)

    ers_feats = get_features(erss, 8, 30, 1)
    eds_feats = get_features(erds, 8, 30, 2)
    mrcp_feats = get_features(erds, 3, 30, 3)

    return ers_feats, eds_feats, mrcp_feats

a, b, c = get_all_features(raw_filt)
#TODO
#   extract features
#   normalize data
#   perform dimension reduction
#   train model
