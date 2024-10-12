import numpy as np
import mne
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
	# def __init__(self, data_to_extract_from):
	def __init__(self):
		#better not creating a lifetime container here, threading might be tricky, keep it to local variables
		pass

	#feature extractor should have this for sure
	def calculate_mean_power_energy(self, activation, epoch, sfreq):
		mean_act = np.mean(activation, axis=1)
		energy = np.sum(activation ** 2, axis=1)
		power = energy / (len(epoch) * sfreq)
		
		# current_feature_vec = np.array([mean_act, energy, power])
		current_feature_vec = np.zeros((3,8))
		current_feature_vec[0] = mean_act
		current_feature_vec[1] = energy
		current_feature_vec[2] = power
		return current_feature_vec


	#this should be the transform for this class
	def create_feature_vectors(self, epochs, sfreq, compute_y=False):
		y = [] if compute_y else None #we only need this onece, if its ers, since event types are the same across epochs
		feature_matrix = []
		for idx, epoch in enumerate(epochs):
			#epoch is already filtered by now
			mean = np.mean(epoch, axis=0)
			activation = epoch - mean

			current_feature_vec = self.calculate_mean_power_energy(activation, epoch, sfreq)
			feature_matrix.append(current_feature_vec)

			if compute_y == True:
				event_type = epochs.events[idx][2] - 1  #[18368(time)     0(?)     1(event_type)]
				y.append(event_type)
			
		feature_matrix = np.array(feature_matrix)
		y = np.array(y) if compute_y else None

		return feature_matrix, y
	


	def transform(self, X):
		'''
		Input: filtered and cropped list of epochs from EpochExtractor
		Output: a (x,y,z)d np array of created features based on mean, energy, power
		NO LABELS HERE, WILL DO SEPARATE
		'''
		#this is now SEPARATE AND PROB WE DONT ASSIGN LABELS
		sfreq = 160.0
		feature_matrices = []
		for filtered_epoch in X:
			feature_matrix  = self.feature_extractor.create_feature_vectors(X, sfreq)
			feature_matrices.append(feature_matrix)
		
		# if compute_y == True:
		# 	labels = y

		# if labels is None:
		# 	raise ValueError("Labels were not assigned. Ensure that at least one analysis computes labels.")
		
		sample_counts = [fm.shape[0] for fm in feature_matrices]
		if not all(count == sample_counts[0] for count in sample_counts):
			raise ValueError("Inconsistent number of samples across analyses. Ensure all have the same number of epochs.")

		res = np.concatenate(feature_matrices, axis=1)
		return res

