#!/usr/bin/python
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

mne.set_log_level(verbose='WARNING')

def load_data(file):
    raw = mne.io.read_raw_edf(file, preload=True)

    # FILTER

    lo_cut = 0.1
    hi_cut = 30

    filtered = raw.copy().filter(lo_cut, hi_cut)
    filtered = filtered.notch_filter(freqs=50.0) # remove line noise

    return filtered

raw_filt = load_data("files/S001/S001R03.edf")

# EXTRACT FEATURES

sfreq = raw_filt.info["sfreq"]
channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]

ica = mne.preprocessing.ICA(method="infomax")

def get_features(epochs, lofreq, hifreq, epoch_type):
    feat_mat = []
    y = []
    for idx, epoch in enumerate(epochs):
        mean = np.mean(epoch, axis=0)

        filtered = mne.filter.filter_data(epoch, method="iir", l_freq=lofreq, h_freq=hifreq,
                                 sfreq=sfreq)

        activation = filtered - mean

        energy = np.sum(activation ** 2, axis=1)
        power = energy / (len(epoch) * sfreq)

        event_type = epochs.events[idx][2] - 1

        features = np.hstack((mean, energy, power))
        y.append(event_type)

        feat_mat.append(features)
    return np.array(feat_mat), np.array(y)

def standardize(arr):
    scaler = StandardScaler()
    return scaler.fit_transform(arr)

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

    ers_feats, ers_y = get_features(erss, 8, 30, 1)
    eds_feats, eds_y = get_features(erds, 8, 30, 2)
    mrcp_feats, mrcp_y = get_features(erds, 3, 30, 3)

    return ers_feats, ers_y, erss.info

a, b, info = get_all_features(raw_filt)

scaler = StandardScaler()
pca = PCA()
reg = LogisticRegression(penalty='l1', solver='liblinear')

pipeline = Pipeline([("Scaler", scaler), ("CSP", pca), ("LogisticRegression", reg)])

pipeline.fit(a, b)

predict = mne.io.read_raw_edf("files/S002/S002R03.edf", preload=True)
#TODO
#   extract features
#   normalize data
#   perform dimension reduction
#   train model
