#!/usr/bin/python
import numpy as np
import mne
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from dataset_preprocessor import Preprocessor
from pipeline import PipelineWrapper
import joblib

mne.set_log_level(verbose='WARNING')

channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]

predict = [
"files/S018/S018R11.edf",
"files/S042/S042R07.edf",
"files/S042/S042R03.edf",
#"/files/S042/S042R11.edf",
#"/files/S052/S052R07.edf",
#"/files/S052/S052R03.edf",
#"/files/S052/S052R11.edf",
#"/files/S104/S104R11.edf",
#"/files/S104/S104R07.edf",
#"/files/S090/S090R11.edf",
#"/files/S086/S086R11.edf",
#"/files/S086/S086R03.edf",
#"/files/S086/S086R07.edf",
#"/files/S017/S017R11.edf",
#"/files/S017/S017R07.edf",
#"/files/S017/S017R03.edf",
#"/files/S013/S013R07.edf",
#"/files/S013/S013R11.edf",
#"/files/S013/S013R03.edf",
#"/files/S055/S055R11.edf",
#"/files/S055/S055R07.edf",
#"/files/S055/S055R03.edf",
#"/files/S016/S016R03.edf",
#"/files/S016/S016R07.edf",
#"/files/S016/S016R11.edf",
#"/files/S103/S103R11.edf",
]

files = [
		"files/S018/S018R07.edf",
		"files/S018/S018R03.edf",
		"files/S104/S104R03.edf",
		"files/S091/S091R11.edf",
		"files/S091/S091R03.edf",
		"files/S091/S091R07.edf",
		"files/S082/S082R11.edf",
		"files/S082/S082R03.edf",
		"files/S082/S082R07.edf",
		"files/S048/S048R03.edf",
		"files/S048/S048R11.edf",
		"files/S048/S048R07.edf",
		"files/S038/S038R11.edf",
		"files/S038/S038R07.edf",
		"files/S038/S038R03.edf",
		"files/S040/S040R03.edf",
		"files/S040/S040R07.edf",
		"files/S040/S040R11.edf",
		"files/S093/S093R07.edf",
		"files/S093/S093R11.edf",
		"files/S093/S093R03.edf",
		"files/S047/S047R11.edf",
		"files/S047/S047R07.edf",
		"files/S047/S047R03.edf",
		"files/S102/S102R07.edf",
		"files/S102/S102R03.edf",
		"files/S102/S102R11.edf",
		"files/S083/S083R11.edf",
		"files/S083/S083R03.edf",
		"files/S083/S083R07.edf",
		"files/S034/S034R07.edf",
		"files/S034/S034R03.edf",
		"files/S034/S034R11.edf",
		"files/S041/S041R07.edf",
		"files/S041/S041R03.edf",
		"files/S041/S041R11.edf",
		"files/S035/S035R07.edf",
		"files/S035/S035R11.edf",
		"files/S035/S035R03.edf",
		"files/S060/S060R07.edf",
		"files/S060/S060R11.edf",
		"files/S060/S060R03.edf",
		"files/S009/S009R11.edf",
		"files/S009/S009R07.edf",
		"files/S009/S009R03.edf",
		"files/S045/S045R11.edf",
		"files/S045/S045R07.edf",
		"files/S045/S045R03.edf",
		"files/S044/S044R03.edf",
		"files/S044/S044R11.edf",
		"files/S044/S044R07.edf",
		"files/S029/S029R11.edf",
		"files/S029/S029R03.edf",
		"files/S029/S029R07.edf",
		"files/S056/S056R03.edf",
		"files/S056/S056R11.edf",
		"files/S056/S056R07.edf",
		"files/S076/S076R07.edf",
		"files/S076/S076R03.edf",
		#"/files/S076/S076R11.edf",
		#"/files/S105/S105R07.edf",
		#"/files/S105/S105R11.edf",
		#"/files/S105/S105R03.edf",
		#"/files/S106/S106R07.edf",
		#"/files/S106/S106R03.edf",
		#"/files/S106/S106R11.edf",
		#"/files/S050/S050R07.edf",
		#"/files/S050/S050R03.edf",
		#"/files/S050/S050R11.edf",
		#"/files/S099/S099R07.edf",
		#"/files/S099/S099R03.edf",
		#"/files/S099/S099R11.edf",
		#"/files/S031/S031R03.edf",
		#"/files/S031/S031R11.edf",
		#"/files/S031/S031R07.edf",
		#"/files/S061/S061R03.edf",
		#"/files/S061/S061R07.edf",
		#"/files/S061/S061R11.edf",
		#"/files/S059/S059R07.edf",
		#"/files/S059/S059R11.edf",
		#"/files/S059/S059R03.edf",
		#"/files/S072/S072R07.edf",
		#"/files/S072/S072R03.edf",
		#"/files/S072/S072R11.edf",
		#"/files/S023/S023R03.edf",
		#"/files/S023/S023R11.edf",
		#"/files/S023/S023R07.edf",
		#"/files/S043/S043R11.edf",
		#"/files/S043/S043R07.edf",
		#"/files/S043/S043R03.edf",
		#"/files/S073/S073R07.edf",
		#"/files/S073/S073R11.edf",
		#"/files/S073/S073R03.edf",
		#"/files/S046/S046R11.edf",
		#"/files/S046/S046R07.edf",
		#"/files/S046/S046R03.edf",
		#"/files/S075/S075R07.edf",
		#"/files/S075/S075R11.edf",
		#"/files/S075/S075R03.edf",
		#"/files/S011/S011R03.edf",
		#"/files/S011/S011R07.edf",
		#"/files/S011/S011R11.edf",
		#"/files/S066/S066R03.edf",
		#"/files/S066/S066R07.edf",
		#"/files/S066/S066R11.edf",
		#"/files/S006/S006R11.edf",
		#"/files/S006/S006R03.edf",
		#"/files/S006/S006R07.edf",
		#"/files/S021/S021R11.edf",
		#"/files/S021/S021R03.edf",
		#"/files/S021/S021R07.edf",
		#"/files/S010/S010R03.edf",
		#"/files/S010/S010R07.edf",
		#"/files/S010/S010R11.edf",
		#"/files/S008/S008R07.edf",
		#"/files/S008/S008R03.edf",
		#"/files/S008/S008R11.edf",
		#"/files/S089/S089R03.edf",
		#"/files/S089/S089R07.edf",
		#"/files/S089/S089R11.edf",
		#"/files/S058/S058R07.edf",
		#"/files/S058/S058R11.edf",
		#"/files/S058/S058R03.edf",
		#"/files/S090/S090R03.edf",
		#"/files/S090/S090R07.edf",
]

class Printer(BaseEstimator, TransformerMixin):
	def __init__(self, n_comps = 2):
		return 

	def fit(self, x_features, stupid):
		print("shape of features: {}".format(x_features.shape))
		return self

	def transform(self, x_features, y=None):
		return x_features

def flipstuff(v):
	max_abs_v_rows = np.argmax(np.abs(v), axis=1)
	shift = np.arange(v.shape[0])
	indices = max_abs_v_rows + shift * v.shape[1]
	signs = np.sign(np.take(np.reshape(v, (-1,)), indices, axis=0))
	v *= signs[:, np.newaxis]
	return v




class My_PCA(BaseEstimator, TransformerMixin):
	def __init__(self, n_comps = 2): #amount of PCAs to select, we have to check later for how much percentage do they cover
		self.n_comps = n_comps
		self.basis = None
		self.current_centered_feature = None

	def fit(self, x_features, y=None):
		self.mean_ = np.mean(x_features, axis=0)
		# When X is a scipy sparse matrix, self.mean_ is a numpy matrix, so we need
		# to transform it to a 1D array. Note that this is not the case when X
		# is a scipy sparse array.
		# TODO: remove the following two lines when scikit-learn only depends
		# on scipy versions that no longer support scipy.sparse matrices.
		self.mean_ = np.reshape(np.asarray(self.mean_), (-1,))

		n_samples=x_features.shape[0]
		C = x_features.T @ x_features
		C -= (
			n_samples
			* np.reshape(self.mean_, (-1, 1))
			* np.reshape(self.mean_, (1, -1))
		)
		C /= n_samples - 1

		x_features = x_features.T
		zerodx = x_features

		for i in range(len(x_features)):
			zerodx[i] -= zerodx[i].mean()


		#compute covariance matrix
		#cov_matrix2 = np.matmul(zerodx, zerodx.T) / (zerodx.shape[1] - 1)
		cov_matrix = np.cov(zerodx) #T is to compute between features instead of datapoints, (rows)

		cov_matrix = C

		#eigenval and eigenvec
		#eig is making imaginary numbers because of floating point precisions
		eigvals, eigvecs = np.linalg.eigh(cov_matrix)

		#eigvals and eigvecs pair up
		#eigvals are sorted in ascending order
		#eigvecs are column vectors
		eigvals = np.reshape(np.asarray(eigvals), (-1,))
		eigvecs = np.asarray(eigvecs)

		#sort eigvals and eigvecs in descending order
		eigvals = np.flip(eigvals, axis=0)
		eigvecs = np.flip(eigvecs, axis=1)

		eigvals[eigvals < 0.0] = 0.0

		Vt = eigvecs.T

		#Vt = flipstuff(Vt)
		
		self.basis = np.asarray(Vt[:self.n_comps, :]) #there war copy=True here but not sure why would we need a copy
		#self.basis = np.asarray(tmp.T[:self.n_comps].T, copy=True)
		#print("ours")
		#print(self.basis)
		#print("---")
		#print()
		
		'''
		5.0001 4.9999
		4.9999 5.0000
		'''

		#sort eigenvalues and vectors, these return indices, not values of eigenvals in descending order
		'''
		eigenvals   eigenvecs
		2.1         v2
		0.8         v1
		0.5         v3
		'''
		#print(np.cov(np.matmul(eigvecs.T, zerodx)))
		return self


	def transform(self, x_features):
		X_transformed = x_features @ self.basis.T

		X_transformed -= np.reshape(self.mean_, (1, -1)) @ self.basis.T

		#x_features = x_features.T
		#zerodx = x_features

		##transform the data by projecting it to PCAs #new eigenvecs are now the new axes, dot projects the features to the new axes
		#for i in range(len(x_features)):
		#    zerodx[i] -= zerodx[i].mean()

		##features_transformed = np.dot(current_centered_feature, self.current_selected_eigenvectors)
		#res = np.matmul(self.basis.T, zerodx).T

		return X_transformed




# def ft_PCA(features): #maybe add n components?
#     #calculate the mean for gene 1 and gene 2

#     n_componens = 2
#     feature_mean = np.mean(features)
#     features_centered = features - feature_mean

#     #compute covariance matrix
#     cov_matrix = np.cov(features_centered.T) #T is to compute between features instead of datapoints, (rows)
#     '''
#     feature vec
#     [ X_11, X_12, X_13, ..., X_1m ]
#     [ X_21, X_22, X_23, ..., X_2m ]
#     [ X_31, X_32, X_33, ..., X_3m ]
#             ...
#     [ X_n1, X_n2, X_n3, ..., X_nm ]


#     centered transposed data
#     Sample1  Sample2  Sample3  Sample4  Sample5
#     A   -2.0     -1.0      0.0      1.0      2.0
#     B   -4.0     -2.0      0.0      2.0      4.0
#     C   -6.0     -3.0      0.0      3.0      6.0

#     covariance matrix:

#         A      B      C
#     A  2.5    5.0    7.5
#     B  5.0   10.0   15.0
#     C  7.5   15.0   22.5

#     Cov(A,A) variance of feature A (with itself)
#     Cov(A,B) covariance of A and B
#     Cov(A,C) covariance of A and C
#     '''

#     #eigenval and eigenvec
#     eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

#     #sort eigenvalues and vectors
#     sorted_indexes = np.argsort(eigenvalues[::-1]) #descending, to explain the most variance (highest vals to vecs)
#     '''
#     eigenvals   eigenvecs
#     2.1         v2
#     0.8         v1
#     0.5         v3
#     '''

#     sorted_eigenvals = eigenvalues[sorted_indexes]
#     sorted_eigenvecs = eigenvectors[:, sorted_indexes]

#     n_selected_eigenvecs = sorted_eigenvecs[:, n_componens]

#     #transform the data by projecting it to PCAs #new eigenvecs are now the new axes, dot projects the features to the new axes
#     features_transformed = np.dot(features_centered, n_selected_eigenvecs)


# FILTER

# from pathlib import Path
# class Preprocessor:
# 	def __init__(self, data_path):
# 		self.data_path = data_path
# 		self.data_channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]
# 		self.raw_data = []

# 	def load_raw_data(self):
# 		for file_path in self.data_path:
# 			file_path = Path(file_path)  # Convert each individual path to a Path object
# 			if not file_path.exists():
# 				raise FileNotFoundError(f"Data path/file '{file_path}' does not exist.")
# 			try:
# 				raw = mne.io.read_raw_edf(file_path, include=self.data_channels)
# 				self.raw_data.append(raw)
# 			except IOError as e:
# 				raise e

# 	def filter_frequencies(self, raw, lo_cut, hi_cut, noise_cut):
# 		filtered_lo_hi = raw.copy().filter(lo_cut, hi_cut)
# 		filter_noise = filtered_lo_hi.notch_filter(noise_cut)
# 		return filter_noise

# 	def filter_raw_data(self):
# 		filtered_data = []
# 		for raw in self.raw_data:
# 			raw.load_data()
# 			filtered_data.append(self.filter_frequencies(raw, lo_cut=0.1, hi_cut=30, noise_cut=50))
# 		self.raw_data = [] #empty memory, wouldnt it leak? 
# 		return filtered_data




# def filter_raw(data):
# 	lo_cut = 0.1
# 	hi_cut = 30

# 	filtered = data.copy().filter(lo_cut, hi_cut)
# 	filtered = filtered.notch_filter(freqs=50.0) # remove line noise

# 	return filtered


# def load_raw(files):
# 	raws = []
# 	for file in files:
# 		raws.append(mne.io.read_raw_edf(file, include=channels))
# 	return raws


# print("loading data")
# raws = load_raw(files)

# raw_filtered = []

# print("filtering...")
# for raw in raws:
# 	raw.load_data()
# 	raw_filtered.append(filter_raw(raw))

# print("done filtering")
# raws = None







# EXTRACT FEATURES



#FEATURE EXTRACTOR CLASS




#EPOCH PROCESSOR CLASS




#PIPELINE CLASS WHICH WOULD SERVE AS ORCHESTRATOR BETWEEN FEATURE AND EPOCH PROCESS
# class PipelineWrapper:
# 	def __init__(self, n_comps=2, mode='training'):
# 		#if we are in prediction mode, we should load the pipeline first
# 		self.scalers = {} #here we use fit_transform for each scaler
# 		self.is_training_mode = mode #maybe later a bool
# 		self.pipeline = Pipeline([("PCA", self.pca), ("Printer", Printer()), ("LogisticRegression", self.model)])
# 		self.pca = My_PCA(n_comps=n_comps)
# 		self.model = LogisticRegression() #can be the neural net later


# 	def fit_scalers(self, x_train):
# 		for i in range(x.shape[1]):
# 			self.scalers[i] = StandardScaler()
# 			x_train[:, i, :] = self.scalers[i].fit_transform(x_train[:, i, :])
# 		return x_train


# 	def fit(self, x_train, y_train):
# 		self.fit_scalers(x_train) #this also stores the individual scalers which we can then save for the prediction
# 		self.pipeline.fit(x_train, y_train)
	

# 	def predict(self, x_test, y):
# 		x_test_scaled = self.fit_scalers(x_test)
# 		return self.pipeline.predict(x_test_scaled)


# 	def save_pipeline(self, filepath):
# 		joblib.dump({
# 			'scalers': self.scalers,
# 			'pipeline': self.pipeline
# 		}, filepath)


# 	def load_pipeline(self, filepath):
# 		data = 	joblib.load(filepath)
# 		self.scalers = data['scalers']
# 		self.pipeline = data['pipeline']




#MAIN HANDLER CLASS, ANALYSYS MANAGER



ica = mne.preprocessing.ICA(method="infomax")

def get_features(epochs, tmin, tmax, lofreq, hifreq, epoch_type, sfreq):
	feat_mat = []
	y = []
	epochs = epochs.copy().crop(tmin=tmin, tmax=tmax)
	for idx, epoch in enumerate(epochs):
		mean = np.mean(epoch, axis=0)

		filtered = mne.filter.filter_data(epoch, method="iir", l_freq=lofreq, h_freq=hifreq,
								 sfreq=sfreq)

		activation = filtered - mean

		mean_act = np.mean(activation, axis=1)
		energy = np.sum(activation ** 2, axis=1)
		power = energy / (len(epoch) * sfreq)

		event_type = epochs.events[idx][2] - 1

		#standarization will probably go wrong...
		#try to make 2d array
		features = np.zeros((3, 8))

		features[0] = mean_act
		features[1] = energy
		features[2] = power

		#print(mean_act.shape)
		#print(energy.shape)
		#print(power.shape)
		#features = np.hstack((mean_act, energy, power))

		y.append(event_type)

		feat_mat.append(features)
	return np.array(feat_mat), np.array(y)



def get_all_features(data):
	event_id = {"T1": 1, "T2": 2}
	events, event_dict = mne.events_from_annotations(data)
	
	sfreq = data.info["sfreq"]

	epochs = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=5.1,
						baseline=None, preload=True)
	
	#ica.fit(data).apply(epochs)
	# TODO we're probably not actually applying the ICA algo
	# TODO use apply!
	#ica.fit(erss)
	#ica.fit(erds)
	#ica.fit(mrcp)

	#different types of analysis per epoch
	mrcp_feats, mrcp_y = get_features(epochs, -2, 0, 3, 30, 3, sfreq)
	erd_feats, erd_y = get_features(epochs, -2, 0, 8, 30, 2, sfreq)
	ers_feats, ers_y = get_features(epochs, 4.1, 5.1, 8, 30, 1, sfreq)


	res = np.concatenate((ers_feats, erd_feats, mrcp_feats), axis=1)

	return res, ers_y

def get(arr):
	features = []
	y = []

	for filtered in arr:
		x,  epochs = get_all_features(filtered)
		print("got some features")
		
		for i in x:
			features.append(i)

		for i in epochs:
			y.append(i)

	features = np.array(features)
	print(features.shape)
	return features, np.array(y)

#--------------------------------------------------------------------------------------------------------------------------
#beginning of preprocessor class
dataset_preprocessor = Preprocessor()
dataset_preprocessor.load_raw_data(data_path=files)
new_filtered = dataset_preprocessor.filter_raw_data()

# x, y = get(raw_filtered)
x_train, y = get(new_filtered)
# scaler = StandardScaler()

#THIS IS WHERE THE PIPELINE WRAPPER COMES INTO PLAY
# scalers = {}
# print(f'{x.shape} is the shape of x')
# for i in range(x.shape[1]):
# 	scalers[i] = StandardScaler()
# 	x[:, i, :] = scalers[i].fit_transform(x[:, i, :])


pipeline_custom = PipelineWrapper()
# print(f'{x.shape} before passing to fit scaler')

# x_train = pipeline_custom.fit_scalers(x) #scalers are saved in the pipeline
#why do we have to reshape it here?
# x = x.reshape((x.shape[0], -1))
# print(f'{x_train.shape} is the shape after)
# x_train = x_train.reshape((x_train.shape[0], -1))
# print(x.shape)
# print(f'{x_train.shape} is x shape after reshaping')


##pca = My_PCA(n_comps=42)
#pca = PCA(n_components=42, svd_solver="covariance_eigh")
##reg = LogisticRegression(penalty='l1', solver='liblinear')
#reg = RandomForestClassifier()
#TODO remove random state
##reg = MLPClassifier(random_state=42, hidden_layer_sizes=(20, 10), max_iter=16000)

np.set_printoptions(threshold=sys.maxsize)
#print("theirs")
#print(PCA(n_components=72).fit(x).components_)
#print("---")
#print()

##pipeline = Pipeline([("PCA", pca), ("Printer", Printer()), ("LogisticRegression", reg)])

##pipeline.fit(x, y)

pipeline_custom.fit(x_train, y)




# predict_raw = dataset_preprocessor.load_raw(predict)
predict_raw = dataset_preprocessor.load_raw_data(data_path=predict)
predict_filtered = dataset_preprocessor.filter_raw_data()

# filtered_predict = []
# for raw in predict_raw:
# 	raw.load_data()
# 	filtered_predict.append(filter_raw(raw))


px, py = get(predict_filtered)


# for i in range(px.shape[1]):
# 	px[:, i, :] = scalers[i].transform(px[:, i, :])

# for i in range(px.shape[1]):
# 	px[:, i, :] = pipeline_custom.scalers[i].transform(px[:, i, :])


# px = px.reshape((px.shape[0], -1))

# res = pipeline.predict(px)
res = pipeline_custom.predict(px)
print(res)
print(py)

acc = accuracy_score(py, res)
print(acc)
#TODO
#   extract features
#   normalize data
#   perform dimension reduction
#   train model
