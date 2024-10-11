import mne
import numpy as np
from feature_extractor import FeatureExtractor

class EpochProcessor:
	def __init__(self, feature_extractor_instance):
		self.feature_extractor = feature_extractor_instance 



	def extract_epochs(self, data):
		event_id = {"T1": 1, "T2": 2}
		events, _ = mne.events_from_annotations(data)
		sfreq = data.info["sfreq"]
		epochs = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=5.1,
							baseline=None, preload=True)
		return epochs, sfreq



#feature extractor have self,y and self feature matrix
#is ica filter hopeless?
	def process_epochs(self, filtered_eeg_data):
		epochs, sfreq = self.extract_epochs(filtered_eeg_data)

		#here we could optimize the create feature vector part
		analysis = {
			'mrcp': {'tmin': -2, 'tmax': 0, 'lofreq': 3, 'hifreq': 30},
			'erd': {'tmin': -2, 'tmax': 0, 'lofreq': 8, 'hifreq': 30},
			'ers': {'tmin': 4.1, 'tmax': 5.1, 'lofreq': 8, 'hifreq': 30}
		}

		feature_matrices = []
		labels = []

		for analysis_name, parameters in analysis.items():
			cropped_epochs = epochs.copy().crop(tmin=parameters['tmin'], tmax=parameters['tmax'])
			filtered_epochs = cropped_epochs.filter(h_freq=parameters['hifreq'],
													l_freq=parameters['lofreq'],
													method='iir')
			#create an object of mne type
			compute_y = (analysis_name == 'ers')
			feature_matrix, y = self.feature_extractor.create_feature_vectors(filtered_epochs, sfreq, compute_y)
			feature_matrices.append(feature_matrix)
			
			if compute_y:
				labels.append(y)

		if labels is None:
			raise ValueError("Labels were not assigned. Ensure that at least one analysis computes labels.")
		#different types of analysis per epoch
		#we could prefilter the epochs outside the loop in create feature vectors
		#we would only do 
		# mrcp_feats, mrcp_y = self.feature_extractor.create_feature_vectors(epochs, -2, 0, 3, 30, 3, sfreq)
		# erd_feats, erd_y = self.feature_extractor.create_feature_vectors(epochs, -2, 0, 8, 30, 2, sfreq)
		# ers_feats, ers_y = self.feature_extractor.create_feature_vectors(epochs, 4.1, 5.1, 8, 30, 1, sfreq)

		# print(f"ers_y:{ers_y}, erd_y:{erd_y}, mrcp_y:{mrcp_y}\n")
		# res = np.concatenate((ers_feats, erd_feats, mrcp_feats), axis=1)
		# return res, ers_y
		res = np.concatenate(feature_matrices, axis=1)
		return res, labels #ers y previously