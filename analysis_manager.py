from epoch_processor import EpochProcessor
import numpy as np


class AnalysisManager:
	def __init__(self, epoch_processor_instance):
		self.epoch_processor = epoch_processor_instance



	def get_features_and_labels(self, filtered_data):
			features = []
			y = []
			#this filtered eeg data have to come in 2 sec chunks, create a buffer which holds that
			#implement a buffer where we send overlapping data and feed it to the process part
			for filtered in filtered_data:
				print("Processing epochs, receiving features.")

				x,  epochs = self.epoch_processor.process_epochs(filtered)
				for i in x:
					features.append(i)
				for i in epochs:
					y.append(i)

			return np.array(features), np.array(y)




	# def create_feature_vectors(self, epochs, sfreq, compute_y=False):
	# 	y = [] if compute_y else None #we only need this onece, if its ers, since event types are the same across epochs
	# 	feature_matrix = []
	# 	for idx, epoch in enumerate(epochs):
	# 		#epoch is already filtered by now
	# 		mean = np.mean(epoch, axis=0)
	# 		activation = epoch - mean

	# 		current_feature_vec = self.calculate_mean_power_energy(activation, epoch, sfreq)
	# 		feature_matrix.append(current_feature_vec)

	# 		if compute_y == True:
	# 			event_type = epochs.events[idx][2] - 1  #[18368(time)     0(?)     1(event_type)]
	# 			y.append(event_type)
			
	# 	feature_matrix = np.array(feature_matrix)
	# 	y = np.array(y) if compute_y else None

	# 	return feature_matrix, y
	
