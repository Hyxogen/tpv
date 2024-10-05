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
	def process_epochs(self, filtered_eeg_data):
		epochs, sfreq = self.extract_epochs(filtered_eeg_data)
		# self.feature_extractor = FeatureExtractor()
		#ica.fit(data).apply(epochs)
		# TODO we're probably not actually applying the ICA algo
		# TODO use apply!
		#ica.fit(erss)
		#ica.fit(erds)
		#ica.fit(mrcp)
	
		#different types of analysis per epoch
		mrcp_feats, mrcp_y = self.feature_extractor.create_feature_vectors(epochs, -2, 0, 3, 30, 3, sfreq)
		erd_feats, erd_y = self.feature_extractor.create_feature_vectors(epochs, -2, 0, 8, 30, 2, sfreq)
		ers_feats, ers_y = self.feature_extractor.create_feature_vectors(epochs, 4.1, 5.1, 8, 30, 1, sfreq)


		res = np.concatenate((ers_feats, erd_feats, mrcp_feats), axis=1)

		return res, ers_y