#this is basically the get method. in case we would extend the functionalities there is at least 
#a central component which we could alter



def get(arr):
		features = []
		y = []

		for filtered in arr:
			x,  epochs = epoch_extraction(filtered)
			print("got some features")
			
			for i in x:
				features.append(i)

			for i in epochs:
				y.append(i)

		features = np.array(features)
		print(features.shape)
		return features, np.array(y)
