#!/usr/bin/python
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

mne.set_log_level(verbose='WARNING')

channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]

predict = [
"files/S018/S018R11.edf",
"files/S042/S042R07.edf",
"files/S042/S042R03.edf",
"files/S042/S042R11.edf",
#"files/S052/S052R07.edf",
#"files/S052/S052R03.edf",
#"files/S052/S052R11.edf",
#"files/S104/S104R11.edf",
#"files/S104/S104R07.edf",
#"files/S090/S090R11.edf",
#"files/S086/S086R11.edf",
#"files/S086/S086R03.edf",
#"files/S086/S086R07.edf",
#"files/S017/S017R11.edf",
#"files/S017/S017R07.edf",
#"files/S017/S017R03.edf",
#"files/S013/S013R07.edf",
#"files/S013/S013R11.edf",
#"files/S013/S013R03.edf",
#"files/S055/S055R11.edf",
#"files/S055/S055R07.edf",
#"files/S055/S055R03.edf",
#"files/S016/S016R03.edf",
#"files/S016/S016R07.edf",
#"files/S016/S016R11.edf",
#"files/S103/S103R11.edf",
]

files = [
"files/S018/S018R07.edf",
"files/S018/S018R03.edf",
"files/S104/S104R03.edf",
"files/S091/S091R11.edf",
"files/S091/S091R03.edf",
"files/S091/S091R07.edf",
"files/S082/S082R11.edf",
"files/S082/S082R03.edf",
"files/S082/S082R07.edf",
"files/S048/S048R03.edf",
"files/S048/S048R11.edf",
"files/S048/S048R07.edf",
"files/S038/S038R11.edf",
"files/S038/S038R07.edf",
"files/S038/S038R03.edf",
"files/S040/S040R03.edf",
"files/S040/S040R07.edf",
"files/S040/S040R11.edf",
"files/S093/S093R07.edf",
"files/S093/S093R11.edf",
"files/S093/S093R03.edf",
"files/S047/S047R11.edf",
"files/S047/S047R07.edf",
"files/S047/S047R03.edf",
"files/S102/S102R07.edf",
"files/S102/S102R03.edf",
"files/S102/S102R11.edf",
#"files/S083/S083R11.edf",
#"files/S083/S083R03.edf",
#"files/S083/S083R07.edf",
#"files/S034/S034R07.edf",
#"files/S034/S034R03.edf",
#"files/S034/S034R11.edf",
#"files/S041/S041R07.edf",
#"files/S041/S041R03.edf",
#"files/S041/S041R11.edf",
#"files/S035/S035R07.edf",
#"files/S035/S035R11.edf",
#"files/S035/S035R03.edf",
#"files/S060/S060R07.edf",
#"files/S060/S060R11.edf",
#"files/S060/S060R03.edf",
#"files/S009/S009R11.edf",
#"files/S009/S009R07.edf",
#"files/S009/S009R03.edf",
#"files/S045/S045R11.edf",
#"files/S045/S045R07.edf",
#"files/S045/S045R03.edf",
#"files/S044/S044R03.edf",
#"files/S044/S044R11.edf",
#"files/S044/S044R07.edf",
#"files/S029/S029R11.edf",
#"files/S029/S029R03.edf",
#"files/S029/S029R07.edf",
#"files/S056/S056R03.edf",
#"files/S056/S056R11.edf",
#"files/S056/S056R07.edf",
#"files/S076/S076R07.edf",
#"files/S076/S076R03.edf",
#"files/S076/S076R11.edf",
#"files/S105/S105R07.edf",
#"files/S105/S105R11.edf",
#"files/S105/S105R03.edf",
#"files/S106/S106R07.edf",
#"files/S106/S106R03.edf",
#"files/S106/S106R11.edf",
#"files/S050/S050R07.edf",
#"files/S050/S050R03.edf",
#"files/S050/S050R11.edf",
#"files/S099/S099R07.edf",
#"files/S099/S099R03.edf",
#"files/S099/S099R11.edf",
#"files/S031/S031R03.edf",
#"files/S031/S031R11.edf",
#"files/S031/S031R07.edf",
#"files/S061/S061R03.edf",
#"files/S061/S061R07.edf",
#"files/S061/S061R11.edf",
#"files/S059/S059R07.edf",
#"files/S059/S059R11.edf",
#"files/S059/S059R03.edf",
#"files/S072/S072R07.edf",
#"files/S072/S072R03.edf",
#"files/S072/S072R11.edf",
#"files/S023/S023R03.edf",
#"files/S023/S023R11.edf",
#"files/S023/S023R07.edf",
#"files/S043/S043R11.edf",
#"files/S043/S043R07.edf",
#"files/S043/S043R03.edf",
#"files/S073/S073R07.edf",
#"files/S073/S073R11.edf",
#"files/S073/S073R03.edf",
#"files/S046/S046R11.edf",
#"files/S046/S046R07.edf",
#"files/S046/S046R03.edf",
#"files/S075/S075R07.edf",
#"files/S075/S075R11.edf",
#"files/S075/S075R03.edf",
#"files/S011/S011R03.edf",
#"files/S011/S011R07.edf",
#"files/S011/S011R11.edf",
#"files/S066/S066R03.edf",
#"files/S066/S066R07.edf",
#"files/S066/S066R11.edf",
#"files/S006/S006R11.edf",
#"files/S006/S006R03.edf",
#"files/S006/S006R07.edf",
#"files/S021/S021R11.edf",
#"files/S021/S021R03.edf",
#"files/S021/S021R07.edf",
#"files/S010/S010R03.edf",
#"files/S010/S010R07.edf",
#"files/S010/S010R11.edf",
#"files/S008/S008R07.edf",
#"files/S008/S008R03.edf",
#"files/S008/S008R11.edf",
#"files/S089/S089R03.edf",
#"files/S089/S089R07.edf",
#"files/S089/S089R11.edf",
#"files/S058/S058R07.edf",
#"files/S058/S058R11.edf",
#"files/S058/S058R03.edf",
#"files/S090/S090R03.edf",
#"files/S090/S090R07.edf",
]

# FILTER

def filter_raw(data):
    lo_cut = 0.1
    hi_cut = 30

    filtered = data.copy().filter(lo_cut, hi_cut)
    filtered = filtered.notch_filter(freqs=50.0) # remove line noise

    return filtered

def load_raw(files):
    raws = []
    for file in files:
        raws.append(mne.io.read_raw_edf(file, include=channels))
    return raws


print("loading data")
raws = load_raw(files)

raw_filtered = []

print("filtering...")
for raw in raws:
    raw.load_data()
    raw_filtered.append(filter_raw(raw))

print("done filtering")
raws = None

# EXTRACT FEATURES


ica = mne.preprocessing.ICA(method="infomax")

def get_features(epochs, tmin, tmax, lofreq, hifreq, epoch_type, sfreq):
    feat_mat = []
    y = []
    epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
    for idx, epoch in enumerate(epochs):
        mean = np.mean(epoch, axis=0)

        filtered = mne.filter.filter_data(epoch, method="iir", l_freq=lofreq, h_freq=hifreq,
                                 sfreq=sfreq)

        activation = filtered - mean

        mean_act = np.mean(activation, axis=1)
        energy = np.sum(activation ** 2, axis=1)
        power = energy / (len(epoch) * sfreq)

        event_type = epochs.events[idx][2] - 1

        #standarization will probably go wrong...
        #try to make 2d array
        features = np.zeros((3, 8))

        features[0] = mean_act
        features[1] = energy
        features[2] = power

        #print(mean_act.shape)
        #print(energy.shape)
        #print(power.shape)
        #features = np.hstack((mean_act, energy, power))

        y.append(event_type)

        feat_mat.append(features)
    return np.array(feat_mat), np.array(y)

def get_all_features(data):
    event_id = {"T1": 1, "T2": 2}
    events, event_dict = mne.events_from_annotations(data)
    
    sfreq = data.info["sfreq"]

    epochs = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=5.1,
                        baseline=None, preload=True)
    #erss = mne.Epochs(data, events, event_id=event_id, tmin=4.1, tmax=5.1,
    #                  baseline=None)
    #erds = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=0,
    #                  baseline=(None, None))
    #mrcp = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=0,
    #                  baseline=None)

    # TODO we're probably not actually applying the ICA algo
    # TODO use apply!
    #ica.fit(erss)
    #ica.fit(erds)
    #ica.fit(mrcp)

    ers_feats, ers_y = get_features(epochs, 4.1, 5.1, 8, 30, 1, sfreq)
    erd_feats, erd_y = get_features(epochs, -2, 0, 8, 30, 2, sfreq)
    mrcp_feats, mrcp_y = get_features(epochs, -2, 0, 3, 30, 3, sfreq)

    res = np.concatenate((ers_feats, erd_feats, mrcp_feats), axis=1)

    return res, ers_y

def get(arr):
    features = []
    y = []

    for filtered in arr:
        x,  epochs = get_all_features(filtered)
        print("got some features")
        
        for i in x:
            features.append(i)

        for i in epochs:
            y.append(i)

    features = np.array(features)
    print(features.shape)
    return features, np.array(y)

x, y = get(raw_filtered)

scaler = StandardScaler()

scalers = {}
for i in range(x.shape[1]):
    scalers[i] = StandardScaler()
    x[:, i, :] = scalers[i].fit_transform(x[:, i, :])

x = x.reshape((x.shape[0], -1))

pca = PCA()
#reg = LogisticRegression(penalty='l1', solver='liblinear')
#reg = RandomForestClassifier()
reg = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=400)

pipeline = Pipeline([("PCA", pca), ("LogisticRegression", reg)])

pipeline.fit(x, y)

predict_raw = load_raw(predict)
filtered_predict = []
for raw in predict_raw:
    raw.load_data()
    filtered_predict.append(filter_raw(raw))


px, py = get(filtered_predict)


for i in range(px.shape[1]):
    px[:, i, :] = scalers[i].transform(px[:, i, :])

px = px.reshape((px.shape[0], -1))

res = pipeline.predict(px)
print(res)
print(py)

acc = accuracy_score(py, res)
print(acc)
#TODO
#   extract features
#   normalize data
#   perform dimension reduction
#   train model
