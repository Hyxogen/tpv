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

#raw.plot(title="raw", scalings='auto')
#raw_filt.plot(title="filtered", scalings='auto')
#plt.show()

channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
#channels = ['eeg']

#TODO NOTCH FILTER as in the paper

n_comp = None

# EXTRACT FEATURES

event_id = {"T1": 1, "T2": 2}
events, event_dict = mne.events_from_annotations(raw_filt)

erds = mne.Epochs(raw_filt, events, event_id=event_id, tmin=-2, tmax=0.0,
                  baseline=None, picks=channels)
erss = mne.Epochs(raw_filt, events, event_id=event_id, tmin=4.1, tmax=5.1,
                  baseline=None, picks=channels)
mrcp = mne.Epochs(raw_filt, events, event_id=event_id, tmin=-2, tmax=0,
                  baseline=None, picks=channels)

#ica = FastICA(n_components=n_comp, random_state=97, max_iter=800)
ica2 = mne.preprocessing.ICA(method="infomax")

ica2.fit(erds)

feat_mat = []

for idx, event in enumerate(erds):
    #res = ica.fit(event).components_
    res = event

    filtered = mne.filter.filter_data(res, method="iir", l_freq=8, h_freq=30,
                                      sfreq=raw_filt.info['sfreq'])

    mean = np.mean(event, axis=0)

    activation = filtered - mean

    energy = np.sum(activation ** 2, axis=1)
    power = energy / (len(event) * raw_filt.info['sfreq'])

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

#raw.compute_psd(fmax=50).plot(picks="data", exclude="bads", amplitude=False)
#raw.plot(duration=5, n_channels=30, block=True)
