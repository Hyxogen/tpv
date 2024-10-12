import mne
import numpy as np
from feature_extractor import FeatureExtractor
from sklearn.base import BaseEstimator, TransformerMixin

class EpochProcessor(BaseEstimator, TransformerMixin):
	def __init__(self, feature_extractor_instance):
		self.feature_extractor = feature_extractor_instance 



	def extract_epochs(self, data):
		event_id = {"T1": 1, "T2": 2}
		events, _ = mne.events_from_annotations(data)
		sfreq = data.info["sfreq"]
		epochs = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=5.1,
							baseline=None, preload=True)
		print(sfreq)
		return epochs, sfreq


#feature extractor have self,y and self feature matrix
#is ica filter hopeless?
	def process_epochs(self, filtered_eeg_data):
		epochs, sfreq = self.extract_epochs(filtered_eeg_data)
		
		
		#this is already feature extraction
		analysis = {
			'mrcp': {'tmin': -2, 'tmax': 0, 'lofreq': 3, 'hifreq': 30},
			'erd': {'tmin': -2, 'tmax': 0, 'lofreq': 8, 'hifreq': 30},
			'ers': {'tmin': 4.1, 'tmax': 5.1, 'lofreq': 8, 'hifreq': 30}
		}

		feature_matrices = []
		labels = None

		for analysis_name, parameters in analysis.items():
			cropped_epochs = epochs.copy().crop(tmin=parameters['tmin'], tmax=parameters['tmax'])
			filtered_epochs = cropped_epochs.filter(h_freq=parameters['hifreq'],
													l_freq=parameters['lofreq'],
													method='iir')
			#create an object of mne type?
			compute_y = (analysis_name == 'ers')
			
			#this is now SEPARATE AND PROB WE DONT ASSIGN LABELS
			feature_matrix, y = self.feature_extractor.create_feature_vectors(filtered_epochs, sfreq, compute_y)
			feature_matrices.append(feature_matrix)
			
			if compute_y == True:
				labels = y

		if labels is None:
			raise ValueError("Labels were not assigned. Ensure that at least one analysis computes labels.")
		
		sample_counts = [fm.shape[0] for fm in feature_matrices]
		if not all(count == sample_counts[0] for count in sample_counts):
			raise ValueError("Inconsistent number of samples across analyses. Ensure all have the same number of epochs.")

		res = np.concatenate(feature_matrices, axis=1)
		return res, labels
