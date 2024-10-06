import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pca import My_PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier


'''
TODO:
#eventually we could implement the pipeline for gridsearch even tho our prediction is working fine for now
#from sklearn.base import BaseEstimator, ClassifierMixin (inheritance for gridsearch)
'''


class Printer(BaseEstimator, TransformerMixin):
	def __init__(self, n_comps = 2):
		return 

	def fit(self, x_features, stupid):
		print("shape of features: {}".format(x_features.shape))
		return self

	def transform(self, x_features, y=None):
		return x_features



class PipelineWrapper:
	def __init__(self, n_comps=2, pca=None, model=None, scalers=None):
		self.n_comps = n_comps
		self.scalers = scalers if scalers is not None else {} #here we use fit_transform for each scaler
		# self.is_training_mode = mode #maybe later a bool
		self.pca = pca if pca is not None else My_PCA(n_comps=n_comps)
		self.model = model if model is not None else MLPClassifier(random_state=42, hidden_layer_sizes=(20, 10), max_iter=16000)
		self.pipeline = Pipeline([("PCA", self.pca), ("Printer", Printer()), ("LogisticRegression", self.model)])



	def fit_scalers(self, x_train):
		x_train_copy = x_train.copy()
		for i in range(x_train.shape[1]):
			self.scalers[i] = StandardScaler()
			x_train_copy[:, i, :] = self.scalers[i].fit_transform(x_train[:, i, :])
		return x_train_copy



	def fit(self, x_train, y_train):
		x_train = self.fit_scalers(x_train)  #assign the scaled data the scaled data back to x_train
		x_train = x_train.reshape((x_train.shape[0], -1))
		self.pipeline.fit(x_train, y_train)




	def transform_scalers(self, x_test):
		x_test_copy = x_test.copy()
		for i in range(x_test.shape[1]):
			x_test_copy[:, i, :] = self.scalers[i].transform(x_test[:, i, :])
		return x_test_copy

	def predict(self, x_test):
		x_test_scaled = self.transform_scalers(x_test)
		x_test_scaled = x_test_scaled.reshape((x_test_scaled.shape[0], -1))
		return self.pipeline.predict(x_test_scaled)



	#https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/pipeline.py
	def get_params(self, deep=True): #allows sckit to use my parameters for cross val score
		return {
			'pca': self.pca,
			'model': self.model,
			'scalers': self.scalers,
			'n_comps': self.n_comps
		  }



	#this would be prob used in GridSearch where it would set our model/scalers/pipeline attributes into different combinations to see which performs the bets
	def set_params(self, **params): #it collects any keywords passed into set params into dicts: ncomps=50 -> 'ncomps':50 
		for parameter, value in params.items(): #in case 
			setattr(self, parameter, value)
		#reinit pipeline with new params?
		return self
		


	def save_pipeline(self, filepath):
		joblib.dump({
			'scalers': self.scalers,
			'pipeline': self.pipeline
		}, filepath)



	def load_pipeline(self, filepath):
		data = 	joblib.load(filepath)
		self.scalers = data['scalers']
		self.pipeline = data['pipeline']


