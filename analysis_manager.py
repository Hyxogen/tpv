#this is basically the get method. in case we would extend the functionalities there is at least 
#a central component which we could alter

from epoch_processor import EpochProcessor
import numpy as np

class AnalysisManager:
	def __init__(self, epoch_processor_instance):
		self.epoch_processor = epoch_processor_instance

	# def get_features_and_labels(self, filtered_data):
	def get_features_and_labels(self, filtered_data):
			features = []
			y = []

			for filtered in filtered_data:
				x,  epochs = self.epoch_processor.process_epochs(filtered)
				print("got some features")
				
				for i in x:
					features.append(i)

				for i in epochs:
					y.append(i)

			features = np.array(features)
			# print(features.shape)
			return features, np.array(y)
