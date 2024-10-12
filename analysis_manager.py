from epoch_processor import EpochProcessor
import numpy as np

class AnalysisManager:
	def __init__(self, epoch_extractor_instance):
		self.epoch_extractor = epoch_extractor_instance



	def get_features_and_labels(self, filtered_data):
			features = []
			y = []
			#this filtered eeg data have to come in 2 sec chunks, create a buffer which holds that
			#implement a buffer where we send overlapping data and feed it to the process part
			for filtered in filtered_data:
				x,  epochs = self.epoch_extractor.process_epochs(filtered)
				print("Processing epochs, receiving features.")
				for i in x:
					features.append(i)
				for i in epochs:
					y.append(i)

			return np.array(features), np.array(y)
