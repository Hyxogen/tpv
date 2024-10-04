import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pca import My_PCA
from sklearn.base import BaseEstimator, TransformerMixin


class Printer(BaseEstimator, TransformerMixin):
	def __init__(self, n_comps = 2):
		return 

	def fit(self, x_features, stupid):
		print("shape of features: {}".format(x_features.shape))
		return self

	def transform(self, x_features, y=None):
		return x_features



class PipelineWrapper:
	def __init__(self, n_comps=2, mode='training'):
		#if we are in prediction mode, we should load the pipeline first
		self.scalers = {} #here we use fit_transform for each scaler
		self.is_training_mode = mode #maybe later a bool
		self.pca = My_PCA(n_comps=n_comps)
		self.model = LogisticRegression() #can be the neural net later
		self.pipeline = Pipeline([("PCA", self.pca), ("Printer", Printer()), ("LogisticRegression", self.model)])


	def fit_scalers(self, x_train):
		print(f'Shape of x_train inside fit_scalers: {x_train.shape}')
		x_train_copy = x_train.copy()
		for i in range(x_train.shape[1]):
			self.scalers[i] = StandardScaler()
			x_train_copy[:, i, :] = self.scalers[i].fit_transform(x_train[:, i, :])
			print(f'{i} in loop, xshape {x_train.shape}')
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


	def save_pipeline(self, filepath):
		joblib.dump({
			'scalers': self.scalers,
			'pipeline': self.pipeline
		}, filepath)


	def load_pipeline(self, filepath):
		data = 	joblib.load(filepath)
		self.scalers = data['scalers']
		self.pipeline = data['pipeline']


