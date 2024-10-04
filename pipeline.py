import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from pca import My_PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier


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
		# self.model = LogisticRegression(penalty='l1', solver='liblinear') #can be the neural net later
		self.model = MLPClassifier(random_state=42, hidden_layer_sizes=(20, 10), max_iter=16000)
		self.pipeline = Pipeline([("PCA", self.pca), ("Printer", Printer()), ("LogisticRegression", self.model)])


	def fit_scalers(self, x_train):
		x_train_copy = x_train.copy()
		print(f'{x_train_copy} is the x before fitting with class')
		for i in range(x_train.shape[1]):
			self.scalers[i] = StandardScaler()
			x_train_copy[:, i, :] = self.scalers[i].fit_transform(x_train[:, i, :])
		return x_train_copy

	def fit(self, x_train, y_train):
		x_train = self.fit_scalers(x_train)  #assign the scaled data the scaled data back to x_train
		# print('EXITED SCALER')
		# print(f'{x_train} AFTER FITTING IN CLASS')
		x_train = x_train.reshape((x_train.shape[0], -1))
		# print(f'{x_train} are the scaled features in the class\n\n\n')
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


