#!/usr/bin/python
import numpy as np
import mne
import sys
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score, KFold

from dataset_preprocessor import Preprocessor
from pipeline import PipelineWrapper
from epoch_processor import EpochProcessor
from feature_extractor import FeatureExtractor
from analysis_manager import AnalysisManager

mne.set_log_level(verbose='WARNING')
channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
predict = [
"files/S018/S018R11.edf",
"files/S042/S042R07.edf",
"files/S042/S042R03.edf",
#"/files/S042/S042R11.edf",
#"/files/S052/S052R07.edf",
#"/files/S052/S052R03.edf",
#"/files/S052/S052R11.edf",
#"/files/S104/S104R11.edf",
#"/files/S104/S104R07.edf",
#"/files/S090/S090R11.edf",
#"/files/S086/S086R11.edf",
#"/files/S086/S086R03.edf",
#"/files/S086/S086R07.edf",
#"/files/S017/S017R11.edf",
#"/files/S017/S017R07.edf",
#"/files/S017/S017R03.edf",
#"/files/S013/S013R07.edf",
#"/files/S013/S013R11.edf",
#"/files/S013/S013R03.edf",
#"/files/S055/S055R11.edf",
#"/files/S055/S055R07.edf",
#"/files/S055/S055R03.edf",
#"/files/S016/S016R03.edf",
#"/files/S016/S016R07.edf",
#"/files/S016/S016R11.edf",
#"/files/S103/S103R11.edf",
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
		"files/S083/S083R11.edf",
		"files/S083/S083R03.edf",
		"files/S083/S083R07.edf",
		"files/S034/S034R07.edf",
		"files/S034/S034R03.edf",
		"files/S034/S034R11.edf",
		"files/S041/S041R07.edf",
		"files/S041/S041R03.edf",
		"files/S041/S041R11.edf",
		"files/S035/S035R07.edf",
		"files/S035/S035R11.edf",
		"files/S035/S035R03.edf",
		"files/S060/S060R07.edf",
		"files/S060/S060R11.edf",
		"files/S060/S060R03.edf",
		"files/S009/S009R11.edf",
		"files/S009/S009R07.edf",
		"files/S009/S009R03.edf",
		"files/S045/S045R11.edf",
		"files/S045/S045R07.edf",
		"files/S045/S045R03.edf",
		"files/S044/S044R03.edf",
		"files/S044/S044R11.edf",
		"files/S044/S044R07.edf",
		"files/S029/S029R11.edf",
		"files/S029/S029R03.edf",
		"files/S029/S029R07.edf",
		"files/S056/S056R03.edf",
		"files/S056/S056R11.edf",
		"files/S056/S056R07.edf",
		"files/S076/S076R07.edf",
		"files/S076/S076R03.edf",
		#"/files/S076/S076R11.edf",
		#"/files/S105/S105R07.edf",
		#"/files/S105/S105R11.edf",
		#"/files/S105/S105R03.edf",
		#"/files/S106/S106R07.edf",
		#"/files/S106/S106R03.edf",
		#"/files/S106/S106R11.edf",
		#"/files/S050/S050R07.edf",
		#"/files/S050/S050R03.edf",
		#"/files/S050/S050R11.edf",
		#"/files/S099/S099R07.edf",
		#"/files/S099/S099R03.edf",
		#"/files/S099/S099R11.edf",
		#"/files/S031/S031R03.edf",
		#"/files/S031/S031R11.edf",
		#"/files/S031/S031R07.edf",
		#"/files/S061/S061R03.edf",
		#"/files/S061/S061R07.edf",
		#"/files/S061/S061R11.edf",
		#"/files/S059/S059R07.edf",
		#"/files/S059/S059R11.edf",
		#"/files/S059/S059R03.edf",
		#"/files/S072/S072R07.edf",
		#"/files/S072/S072R03.edf",
		#"/files/S072/S072R11.edf",
		#"/files/S023/S023R03.edf",
		#"/files/S023/S023R11.edf",
		#"/files/S023/S023R07.edf",
		#"/files/S043/S043R11.edf",
		#"/files/S043/S043R07.edf",
		#"/files/S043/S043R03.edf",
		#"/files/S073/S073R07.edf",
		#"/files/S073/S073R11.edf",
		#"/files/S073/S073R03.edf",
		#"/files/S046/S046R11.edf",
		#"/files/S046/S046R07.edf",
		#"/files/S046/S046R03.edf",
		#"/files/S075/S075R07.edf",
		#"/files/S075/S075R11.edf",
		#"/files/S075/S075R03.edf",
		#"/files/S011/S011R03.edf",
		#"/files/S011/S011R07.edf",
		#"/files/S011/S011R11.edf",
		#"/files/S066/S066R03.edf",
		#"/files/S066/S066R07.edf",
		#"/files/S066/S066R11.edf",
		#"/files/S006/S006R11.edf",
		#"/files/S006/S006R03.edf",
		#"/files/S006/S006R07.edf",
		#"/files/S021/S021R11.edf",
		#"/files/S021/S021R03.edf",
		#"/files/S021/S021R07.edf",
		#"/files/S010/S010R03.edf",
		#"/files/S010/S010R07.edf",
		#"/files/S010/S010R11.edf",
		#"/files/S008/S008R07.edf",
		#"/files/S008/S008R03.edf",
		#"/files/S008/S008R11.edf",
		#"/files/S089/S089R03.edf",
		#"/files/S089/S089R07.edf",
		#"/files/S089/S089R11.edf",
		#"/files/S058/S058R07.edf",
		#"/files/S058/S058R11.edf",
		#"/files/S058/S058R03.edf",
		#"/files/S090/S090R03.edf",
		#"/files/S090/S090R07.edf",
]
# ica = mne.preprocessing.ICA(method="infomax")
#--------------------------------------------------------------------------------------------------------------------------
#beginning of preprocessor class
dataset_preprocessor = Preprocessor()
dataset_preprocessor.load_raw_data(data_path=files)
filtered_data = dataset_preprocessor.filter_raw_data()

# feature_extractor = FeatureExtractor(filtered_data)
feature_extractor = FeatureExtractor()
epoch_processor = EpochProcessor(feature_extractor) #dependency injection, in python not necessary but good habit
analysis_manager = AnalysisManager(epoch_processor)

# x_train, y_train = get(filtered_data) #analysismanager get_features_and_labels
x_train, y_train = analysis_manager.get_features_and_labels(filtered_data)

pipeline_custom = PipelineWrapper(n_comps=42)
pipeline_custom.fit(x_train, y_train)

predict_raw = dataset_preprocessor.load_raw_data(data_path=predict)
predict_filtered = dataset_preprocessor.filter_raw_data()
# px_my, py_my = get(predict_filtered)
px_my, py_my = analysis_manager.get_features_and_labels(predict_filtered)


# k_fold_cross_val = KFold(n_splits=15, shuffle=True, random_state=42)
shuffle_split_validation = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

# scoring = ['accuracy', 'precision', 'f1_micro'] this only works for: scores = cross_validate(pipeline_custom, x_train, y_train, scoring=scoring, cv=k_fold_cross_val)
scores = cross_val_score(pipeline_custom, x_train, y_train, scoring='accuracy', cv=shuffle_split_validation)
# sorted(scores.keys())

res_my = pipeline_custom.predict(px_my)
acc_my = accuracy_score(py_my, res_my)
print(acc_my)

#maybe take a look at GridSearch as well?
print(f'Cross-validation accuracy scores for each fold: {scores}')
print(f'Average accuracy: {scores.mean()}')
