import numpy as np
import mne


class FeatureExtractor:
	def __init__(self, data_to_extract_from):
		self.y = []
		self.feature_matrix = [] #1 separate feature vector per epoch
		self.data_to_extract_from = data_to_extract_from
		

	def calculate_mean_power_energy(self, activation, epoch, sfreq):
		mean_act = np.mean(activation, axis=1)
		energy = np.sum(activation ** 2, axis=1)
		power = energy / (len(epoch) * sfreq)
		
		current_feature_vec = np.array([mean_act, energy, power])
		return current_feature_vec


	#isn epoch type needed here?
	def create_feature_vectors(self, epochs, tmin, tmax, lofreq, hifreq, epoch_type, sfreq):
		epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
		for idx, epoch in enumerate(epochs):
			filtered = mne.filter.filter_data(epoch, method="iir", l_freq=lofreq, h_freq=hifreq,
									sfreq=sfreq)

			mean = np.mean(epoch, axis=0)
			activation = filtered - mean

			current_feature_vec = self.calculate_mean_power_energy(activation, epoch, sfreq)
			event_type = epochs.events[idx][2] - 1
			#features = np.hstack((mean_act, energy, power))

			self.y.append(event_type)
			self.feature_matrix.append(current_feature_vec)

			self.feature_matrix = np.array(self.feature_matrix)
			self.y = np.array(self.y)

