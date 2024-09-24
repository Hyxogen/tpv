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

channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]

# EXTRACT FEATURES

sfreq = raw_filt.info["sfreq"]

event_id = {"T1": 1, "T2": 2}
events, event_dict = mne.events_from_annotations(raw_filt)

erds = mne.Epochs(raw_filt, events, event_id=event_id, tmin=-2, tmax=0.0,
                  baseline=None, picks=channels)
erss = mne.Epochs(raw_filt, events, event_id=event_id, tmin=4.1, tmax=5.1,
                  baseline=None, picks=channels)
mrcp = mne.Epochs(raw_filt, events, event_id=event_id, tmin=-2, tmax=0,
                  baseline=None, picks=channels)

ica2 = mne.preprocessing.ICA(method="infomax")

ica2.fit(erds)

feat_mat = []

for idx, event in enumerate(erds):
    res = event

    filtered = mne.filter.filter_data(res, method="iir", l_freq=8, h_freq=30,
                                      sfreq=sfreq)

    mean = np.mean(event, axis=0)

    activation = filtered - mean

    energy = np.sum(activation ** 2, axis=1)
    power = energy / (len(event) * sfreq)

    event_type = erds.events[idx][2]

    features = np.hstack((mean, energy, power, 1, event_type))
    feat_mat.append(features)

print(len(feat_mat))
print(len(feat_mat[0]))
#TODO
#   extract features
#   normalize data
#   perform dimension reduction
#   train model
