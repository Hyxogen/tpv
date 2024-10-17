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
from pipeline_2 import PipelineWrapper2
from epoch_processor import EpochProcessor
from feature_extractor import FeatureExtractor
from analysis_manager import AnalysisManager


from epoch_extractor import epoch_extractooor, extract_epochs
from feature_extractor import feature_extractor, create_feature_vectors, calculate_mean_power_energy

from pca import My_PCA
from sklearn.preprocessing import FunctionTransformer
from filter_transformer import initial_filter, filter_frequencies
from filter_transformer import InitialFilterTransformer
from epoch_extractor import EpochExtractor


from custom_scaler import CustomScaler
from reshaper import Reshaper

mne.set_log_level(verbose='WARNING')
channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
predict = [
"files/S018/S018R11.edf",
"files/S042/S042R07.edf",
"files/S042/S042R03.edf",
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

def extract_epochs(data):
	event_id = {"T1": 1, "T2": 2}
	events, _ = mne.events_from_annotations(data)
	sfreq = data.info["sfreq"] #this is 160 but we could create a custom dataclass to pass this along, transform only expects an X output
	epochs = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=5.1,
						baseline=None, preload=True)
	return epochs, sfreq


#feature extractor have self,y and self feature matrix
#is ica filter hopeless?
def extract_epochs_and_labelsf(eeg_data):
	'''
	Input: X->filtered eeg data, several eeg files thus we need a loop
	output1: Filtered epochs on which we will run feature extraction (based on different timeframes and associated high/low frequencies)
	output2: the labels of the epoch events which we will use as y_train 
	'''
	epochs_list = []
	labels_list = []
	for filtered_eeg_data in eeg_data:
		# print('TRANSFORM IN EXTRACT EPOCHS')
		epochs, sfreq = extract_epochs(filtered_eeg_data)
		epochs_list.append(epochs)
		labels = epochs.events[:, 2]- 1 #these are all the same, prob enough jsut [2][1]
		labels_list.append(labels)
	# print(f'{epochs_list} is the epochs list from transform')
	print(len(epochs_list), len(labels_list))
	return epochs_list, np.concatenate(labels_list)



# ica = mne.preprocessing.ICA(method="infomax")
#--------------------------------------------------------------------------------------------------------------------------



#beginning of preprocessor class
dataset_preprocessor = Preprocessor()
# raw_data = dataset_preprocessor.load_raw_data(data_path=files)
loaded_data = dataset_preprocessor.load_raw_data(data_path=files) #RETURN DOESNT WORK, IT RETURNS AFTER 1 FILE
filtered_data = dataset_preprocessor.filter_raw_data() #THIS WILL BE INITIAL FILTER TRANSFORMER
epochs, labels = extract_epochs_and_labelsf(filtered_data)

#this is gonna be in the pipeline object as initialFilterTransformer



#we have now the filtered data, we n
# feature_extractor = FeatureExtractor(filtered_data)
feature_extractor1 = FeatureExtractor()
epoch_processor = EpochProcessor(feature_extractor1) #dependency injection, in python not necessary but good habit
analysis_manager = AnalysisManager(epoch_processor)

# # x_train, y_train = get(filtered_data) #analysismanager get_features_and_labels
#we need to extract labels separately
# x_train, y_train = analysis_manager.get_features_and_labels(filtered_data)

# pipeline_custom = PipelineWrapper(n_comps=42)
# pipeline_custom.fit(x_train, y_train)


# print(type(filtered_data))
# print(type(labels))

#https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.FunctionTransformer.html
custom_scaler = CustomScaler()
reshaper = Reshaper()
filter_transformer = FunctionTransformer(initial_filter, validate=False)
epoch_transformer = FunctionTransformer(epoch_extractooor, validate=False)
feature_transformer = FunctionTransformer(feature_extractor)
my_pca = My_PCA(n_comps=42)
mlp_classifier = MLPClassifier(hidden_layer_sizes=(20,10),
							   max_iter=16000,
							   random_state=42
)

dataset_preprocessor = Preprocessor()
# raw_data = dataset_preprocessor.load_raw_data(data_path=files)
loaded_data = dataset_preprocessor.load_raw_data(data_path=files) #RETURN DOESNT WORK, IT RETURNS AFTER 1 FILE
filtered_data = dataset_preprocessor.filter_raw_data() #THIS WILL BE INITIAL FILTER TRANSFORMER
epochs, labels = extract_epochs_and_labelsf(filtered_data)
trained_extracted_features = feature_transformer.transform(epochs)

# pipeline_2 = PipelineWrapper2(filter_new, features_extractor_new)
# pipeline_wrapper = PipelineWrapper2(
# 	n_comps=42,
# 	filter_transformer=filter_transformer,
# 	epoch_extractor=epoch_transformer,
# 	feature_extractor=feature_transformer,
# )


pipeline_wrapper = Pipeline([
	# ('filter', filter_transformer),
	# ('epoch_extractor', epoch_transformer),
	# ('feature_extractor', feature_transformer),
	('scaler', custom_scaler),
	('reshaper', reshaper),
	('pca', my_pca),
	('classifier', mlp_classifier)
])

# pipeline_wrapper = Pipeline([
# 	('filter', filter_transformer),
# 	('epoch_extractor', epoch_transformer),
# 	('feature_extractor', feature_transformer),
# 	('scaler', CustomScaler()),
# 	('reshaper', Reshaper()),
# 	('pca', My_PCA(n_comps=42)),
# 	('classification', MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=16000, random_state=42))
# ])

# print(f'{len(epochs)} are epoch lens,\n{len(labels)} are label lens in training')
pipeline_wrapper.fit(trained_extracted_features, labels)

# shuffle_split_validation = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
# # print(filtered_data)
# scores = cross_val_score(pipeline_wrapper, filtered_data, labels, scoring='accuracy', cv=shuffle_split_validation)


# filter_new = InitialFilterTransformer()
# filtered_data_new = filter_new.transform(loaded_data)

# # epoch_extractor_new = EpochExtractor()
# # epochs = epoch_extractor_new.transform(filtered_data_new) #this just gets the epochs which were previously processed and extracted by analysismanager

# features_extractor_new = FeatureExtractor()
# features = features_extractor_new.transform(epochs)



# pipeline_2 = PipelineWrapper2(filter_new, features_extractor_new)
# pipeline_2.transform(epochs, labels)
# pipeline_2.pipeline[:-1].get_feature_names_out()

#WORKS WITH THIS!!! TOMORROW LETS SEE IF WE CAN PUT INITFILTER,EPOCHEXTRACTOR,FEATUREEXTRACTOR INTO THE PIPELINE AND SAVE IT FOR THE PREDICT SCRIPT
# x_train, y_train = analysis_manager.get_features_and_labels(filtered_data)
# x_train, y_train = features, labels
# # print(x_train)
# # print(y_train)

# pipeline_custom = PipelineWrapper(n_comps=42)
# pipeline_custom.fit(x_train, y_train)
#------------------------------------------------------------------------------------------------------------

#can it be that dataset preprocessor filters/loads wrong data?
predict_raw = dataset_preprocessor.load_raw_data(data_path=predict)
predict_filtered = dataset_preprocessor.filter_raw_data()
epochs_predict, labels_predict = extract_epochs_and_labelsf(predict_filtered)
test_extracted_features = feature_transformer.transform(epochs_predict)
print(f'{len(epochs_predict)} is the len of the EPOCHS extracted from filtered, {len(labels_predict)} is the len of the labels predicted\n')


# filter_transformer = FunctionTransformer(initial_filter, validate=False)
# epoch_transformer = FunctionTransformer(epoch_extractooor, validate=False)
# feature_transformer = FunctionTransformer(feature_extractor)

# filter_predict = initial_filter(predict_filtered)
# epochs_predict = epoch_extractooor(filter_predict)
# features_predict = feature_extractor(epochs_predict)
print(f'{len(test_extracted_features)} is the len of predicted features')



# px_my, py_my = get(predict_filtered)
# px_my, py_my = analysis_manager.get_features_and_labels(predict_filtered)


# k_fold_cross_val = KFold(n_splits=15, shuffle=True, random_state=42)
shuffle_split_validation = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
print(f'PREDICT FILTERED IS\n{predict_filtered}')

# # scoring = ['accuracy', 'precision', 'f1_micro'] this only works for: scores = cross_validate(pipeline_custom, x_train, y_train, scoring=scoring, cv=k_fold_cross_val)
# # scores = cross_val_score(pipeline_custom, x_train, y_train, scoring='accuracy', cv=shuffle_split_validation)
scores = cross_val_score(
	pipeline_wrapper, test_extracted_features, 
	labels_predict, 
	scoring='accuracy', 
	cv=shuffle_split_validation
)
	# n_jobs=-1,
	# verbose=1)

# sorted(scores.keys())

print(scores)
print(f'Average accuracy: {scores.mean()}')

# res_my = pipeline_custom.predict(px_my)
# res_my = pipeline_wrapper.predict(predict_filtered)

# print(labels_predict)
# print(res_my)
# acc_my = accuracy_score(labels_predict, res_my)
# print(acc_my)

#maybe take a look at GridSearch as well?
# print(f'Cross-validation accuracy scores for each fold: {scores}')
# print(f'Average accuracy: {scores.mean()}')

grid_search_params = {
	#num of pca components to try in the pipeline
	'pca__n_comps': [20,30,42,50],
	#hidden layers of multilayer perceptron class
	'classifier__hidden_layer_sizes': [(20, 10), (50, 20), (100, 50)],
	#relu->helps mitigate vanishing gradients, faster convergence
	#tanh->hyperbolic tangent, outputs centered around zero
	'classifier__activation': ['relu', 'tanh'],
	#adam, efficient for large datasets, adapts learning rates
	#stochastic gradient, generalize better, slower convergence
	'classifier__solver': ['adam', 'sgd'],
	'classifier__learning_rate_init': [0.001, 0.01, 0.1]
}

from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
	estimator=pipeline_wrapper,
	param_grid=grid_search_params,
	cv=9,  #9fold cross-val
	scoring='accuracy',  #evalmetric
	n_jobs=-1,  #util all all available cpu cores
	verbose=2  # For detailed output
)

grid_search.fit(test_extracted_features, labels_predict)

# Review Results
print("Best Parameters:")
print(grid_search.best_params_)

print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.2f}")

# Retrieve the Best Pipeline
best_pipeline = grid_search.best_estimator_

# Fit the Best Pipeline on the Entire Dataset (Optional)
best_pipeline.fit(test_extracted_features, labels_predict)