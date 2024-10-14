import mne
import numpy as np
from feature_extractor import FeatureExtractor
from sklearn.base import BaseEstimator, TransformerMixin


class EpochExtractor(BaseEstimator, TransformerMixin):
	def __init__(self):
		# self.feature_extractor = feature_extractor_instance 
		pass



	def extract_epochs(self, data):
		event_id = {"T1": 1, "T2": 2}
		events, _ = mne.events_from_annotations(data)
		sfreq = data.info["sfreq"] #this is 160 but we could create a custom dataclass to pass this along, transform only expects an X output
		epochs = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=5.1,
							baseline=None, preload=True)
		return epochs, sfreq



	# def process_epochs(self, X):
	# 	epochs, sfreq = self.extract_epochs(X)


	# 	#this is already feature extraction

	# 	filtered_epochs = []
		
	# 	for analysis_name, parameters in analysis.items():
	# 		cropped_epochs = epochs.copy().crop(tmin=parameters['tmin'], tmax=parameters['tmax'])
	# 		filtered_epoch = cropped_epochs.filter(h_freq=parameters['hifreq'],
	# 												l_freq=parameters['lofreq'],
	# 												method='iir')
	# 		filtered_epochs.append(filtered_epoch)
			
	# 	return filtered_epochs



#feature extractor have self,y and self feature matrix
#is ica filter hopeless?
	def transform(self, X):
		'''
		Input: X->filtered eeg data, several eeg files thus we need a loop
		output: Filtered epochs (based on different timeframes and associated high/low frequencies)
		'''
		epochs_list = []

		for filtered_eeg_data in X:
			epochs, sfreq = self.extract_epochs(filtered_eeg_data)
			epochs_list.append(epochs)
		# print(f'{epochs_list} is the epochs list from transform')
		return epochs_list
		# all_features = []
		# for filtered_data in X:
		# 	epochs, sfreq = self.extract_epochs(filtered_data)
		# 	epochs_data = epochs.get_data() #https://mne.tools/1.7/generated/mne.Epochs.html#mne.Epochs.get_data (data array of shape (n_epochs, n_channels, n_times))

		# 	analysis = {
		# 		'mrcp': {'tmin': -2, 'tmax': 0, 'lofreq': 3, 'hifreq': 30},
		# 		'erd': {'tmin': -2, 'tmax': 0, 'lofreq': 8, 'hifreq': 30},
		# 		'ers': {'tmin': 4.1, 'tmax': 5.1, 'lofreq': 8, 'hifreq': 30}
		# 	}

		# 	filtered_epochs = []
		
		# 	for analysis_name, parameters in analysis.items():
		# 		start_time = (parameters['tmin'] - epochs_data.tmin) * sfreq) #vectorizing is gonna vbe faster than copying objects
		# 		cropped_epochs = epochs_data.copy().crop(tmin=parameters['tmin'], tmax=parameters['tmax'])
		# 		filtered_epoch = cropped_epochs.filter(h_freq=parameters['hifreq'],
		# 												l_freq=parameters['lofreq'],
		# 												method='iir')
		# 		filtered_epochs.append(filtered_epoch)
				
		# 	return filtered_epochs

			# # current_feature_matrix = self.epoch_extractor.process_epochs(filtered)
			# features.append(current_feature_matrix)
		
		