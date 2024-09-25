#!/usr/bin/python
import numpy as np
import mne
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

import seaborn as sns

mne.set_log_level(verbose='WARNING')

channels = ["Fc3.", "Fcz.", "Fc4.", "C3..", "C1..", "Cz..", "C2..", "C4.."]

predict = [
"data/files/S018/S018R11.edf",
"data/files/S042/S042R07.edf",
"data/files/S042/S042R03.edf",
"data/files/S042/S042R11.edf",
"data/files/S052/S052R07.edf",
"data/files/S052/S052R03.edf",
"data/files/S052/S052R11.edf",
"data/files/S104/S104R11.edf",
"data/files/S104/S104R07.edf",
]

files = [
"data/files/S018/S018R07.edf",
"data/files/S018/S018R03.edf",
#"data/files/S104/S104R03.edf",
#"data/files/S091/S091R11.edf",
#"data/files/S091/S091R03.edf",
#"data/files/S091/S091R07.edf",
#"data/files/S082/S082R11.edf",
#"data/files/S082/S082R03.edf",
#"data/files/S082/S082R07.edf",
#"data/files/S048/S048R03.edf",
#"data/files/S048/S048R11.edf",
#"data/files/S048/S048R07.edf",
#"data/files/S038/S038R11.edf",
#"data/files/S038/S038R07.edf",
#"data/files/S038/S038R03.edf",
#"data/files/S040/S040R03.edf",
#"data/files/S040/S040R07.edf",
#"data/files/S040/S040R11.edf",
#"data/files/S093/S093R07.edf",
#"data/files/S093/S093R11.edf",
#"data/files/S093/S093R03.edf",
#"data/files/S047/S047R11.edf",
#"data/files/S047/S047R07.edf",
#"data/files/S047/S047R03.edf",
#"data/files/S102/S102R07.edf",
#"data/files/S102/S102R03.edf",
#"data/files/S102/S102R11.edf",
#"data/files/S083/S083R11.edf",
#"data/files/S083/S083R03.edf",
#"data/files/S083/S083R07.edf",
#"data/files/S034/S034R07.edf",
#"data/files/S034/S034R03.edf",
#"data/files/S034/S034R11.edf",
#"data/files/S041/S041R07.edf",
#"data/files/S041/S041R03.edf",
#"data/files/S041/S041R11.edf",
#"data/files/S035/S035R07.edf",
#"data/files/S035/S035R11.edf",
#"data/files/S035/S035R03.edf",
#"data/files/S060/S060R07.edf",
#"data/files/S060/S060R11.edf",
#"data/files/S060/S060R03.edf",
#"data/files/S009/S009R11.edf",
#"data/files/S009/S009R07.edf",
#"data/files/S009/S009R03.edf",
#"data/files/S045/S045R11.edf",
#"data/files/S045/S045R07.edf",
#"data/files/S045/S045R03.edf",
#"data/files/S044/S044R03.edf",
#"data/files/S044/S044R11.edf",
#"data/files/S044/S044R07.edf",
#"data/files/S029/S029R11.edf",
#"data/files/S029/S029R03.edf",
#"data/files/S029/S029R07.edf",
#"data/files/S056/S056R03.edf",
#"data/files/S056/S056R11.edf",
#"data/files/S056/S056R07.edf",
#"data/files/S076/S076R07.edf",
#"data/files/S076/S076R03.edf",
#"data/files/S076/S076R11.edf",
#"data/files/S105/S105R07.edf",
#"data/files/S105/S105R11.edf",
#"data/files/S105/S105R03.edf",
#"data/files/S106/S106R07.edf",
#"data/files/S106/S106R03.edf",
#"data/files/S106/S106R11.edf",
#"data/files/S050/S050R07.edf",
#"data/files/S050/S050R03.edf",
#"data/files/S050/S050R11.edf",
#"data/files/S099/S099R07.edf",
#"data/files/S099/S099R03.edf",
#"data/files/S099/S099R11.edf",
#"data/files/S031/S031R03.edf",
#"data/files/S031/S031R11.edf",
#"data/files/S031/S031R07.edf",
#"data/files/S061/S061R03.edf",
#"data/files/S061/S061R07.edf",
#"data/files/S061/S061R11.edf",
#"data/files/S059/S059R07.edf",
#"data/files/S059/S059R11.edf",
#"data/files/S059/S059R03.edf",
#"data/files/S072/S072R07.edf",
#"data/files/S072/S072R03.edf",
#"data/files/S072/S072R11.edf",
#"data/files/S023/S023R03.edf",
#"data/files/S023/S023R11.edf",
#"data/files/S023/S023R07.edf",
#"data/files/S043/S043R11.edf",
#"data/files/S043/S043R07.edf",
#"data/files/S043/S043R03.edf",
#"data/files/S073/S073R07.edf",
#"data/files/S073/S073R11.edf",
#"data/files/S073/S073R03.edf",
#"data/files/S046/S046R11.edf",
#"data/files/S046/S046R07.edf",
#"data/files/S046/S046R03.edf",
#"data/files/S075/S075R07.edf",
#"data/files/S075/S075R11.edf",
#"data/files/S075/S075R03.edf",
#"data/files/S011/S011R03.edf",
#"data/files/S011/S011R07.edf",
#"data/files/S011/S011R11.edf",
#"data/files/S066/S066R03.edf",
#"data/files/S066/S066R07.edf",
#"data/files/S066/S066R11.edf",
#"data/files/S006/S006R11.edf",
#"data/files/S006/S006R03.edf",
#"data/files/S006/S006R07.edf",
#"data/files/S021/S021R11.edf",
#"data/files/S021/S021R03.edf",
#"data/files/S021/S021R07.edf",
#"data/files/S010/S010R03.edf",
#"data/files/S010/S010R07.edf",
#"data/files/S010/S010R11.edf",
#"data/files/S008/S008R07.edf",
#"data/files/S008/S008R03.edf",
#"data/files/S008/S008R11.edf",
#"data/files/S089/S089R03.edf",
#"data/files/S089/S089R07.edf",
#"data/files/S089/S089R11.edf",
#"data/files/S058/S058R07.edf",
#"data/files/S058/S058R11.edf",
#"data/files/S058/S058R03.edf",
#"data/files/S090/S090R03.edf",
#"data/files/S090/S090R07.edf",
#"data/files/S090/S090R11.edf",
#"data/files/S086/S086R11.edf",
#"data/files/S086/S086R03.edf",
#"data/files/S086/S086R07.edf",
#"data/files/S017/S017R11.edf",
#"data/files/S017/S017R07.edf",
#"data/files/S017/S017R03.edf",
#"data/files/S013/S013R07.edf",
#"data/files/S013/S013R11.edf",
#"data/files/S013/S013R03.edf",
#"data/files/S055/S055R11.edf",
#"data/files/S055/S055R07.edf",
#"data/files/S055/S055R03.edf",
#"data/files/S016/S016R03.edf",
#"data/files/S016/S016R07.edf",
#"data/files/S016/S016R11.edf",
#"data/files/S103/S103R11.edf",
]

# FILTER
#can be beneficial later, mnaybe pipeline would not work without it either(fit_transform, fit inheritance)
class My_PCA(BaseEstimator, TransformerMixin):
	def __init__(self, n_comps = 2): #amount of PCAs to select, we have to check later for how much percentage do they cover
		self.n_comps = n_comps
		self.current_selected_eigenvectors = None
		self.current_centered_feature = None

	def fit(self, x_features):
		feature_mean = np.mean(x_features)
		#self.current_centered_feature= x_features - feature_mean, in case wanna try it on random data, generalize this for now
		current_centered_feature = x_features - feature_mean
		#compute covariance matrix
		cov_matrix = np.cov(current_centered_feature.T) #T is to compute between features instead of datapoints, (rows)
		'''
		feature vec
		[ X_11, X_12, X_13, ..., X_1m ]
		[ X_21, X_22, X_23, ..., X_2m ]
		[ X_31, X_32, X_33, ..., X_3m ]
				...
		[ X_n1, X_n2, X_n3, ..., X_nm ]


		centered transposed data
		Sample1  Sample2  Sample3  Sample4  Sample5
		A   -2.0     -1.0      0.0      1.0      2.0
		B   -4.0     -2.0      0.0      2.0      4.0
		C   -6.0     -3.0      0.0      3.0      6.0

		covariance matrix:

			A      B      C
		A  2.5    5.0    7.5
		B  5.0   10.0   15.0
		C  7.5   15.0   22.5

		Cov(A,A) variance of feature A (with itself)
		Cov(A,B) covariance of A and B
		Cov(A,C) covariance of A and C
		'''

		#eigenval and eigenvec
		eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix) #eig is making imaginary numbers because of floating point precisions
		'''
		5.0000 4.9999
		4.9999 5.0000
		'''
		#sort eigenvalues and vectors, these return indices, not values of eigenvals in descending order
		sorted_eigen_val_indexes = np.argsort(eigenvalues[::-1]) #descending, to explain the most variance (highest vals to vecs)
		'''
		eigenvals   eigenvecs
		2.1         v2
		0.8         v1
		0.5         v3
		'''
		sorted_eigenvecs = eigenvectors[:, sorted_eigen_val_indexes]
		self.current_selected_eigenvectors = sorted_eigenvecs[:, :self.n_comps]


	def transform(self, x_features):
		#transform the data by projecting it to PCAs #new eigenvecs are now the new axes, dot projects the features to the new axes
		feature_mean = np.mean(x_features, axis=0)
		current_centered_feature = x_features - feature_mean #this is redundant if we only use it on class data
		
		features_transformed = np.dot(current_centered_feature, self.current_selected_eigenvectors)
		return features_transformed


	#we could also add the transformed features to a self.current_transformed_features
	def fit_transform(self, x_features, y=None):
		self.fit(x_features)
		transformed_features = self.transform(x_features)
		return transformed_features



# def ft_PCA(features): #maybe add n components?
# 	#calculate the mean for gene 1 and gene 2

# 	n_componens = 2
# 	feature_mean = np.mean(features)
# 	features_centered = features - feature_mean

# 	#compute covariance matrix
# 	cov_matrix = np.cov(features_centered.T) #T is to compute between features instead of datapoints, (rows)
# 	'''
# 	feature vec
# 	[ X_11, X_12, X_13, ..., X_1m ]
# 	[ X_21, X_22, X_23, ..., X_2m ]
# 	[ X_31, X_32, X_33, ..., X_3m ]
# 			...
# 	[ X_n1, X_n2, X_n3, ..., X_nm ]


# 	centered transposed data
# 	Sample1  Sample2  Sample3  Sample4  Sample5
# 	A   -2.0     -1.0      0.0      1.0      2.0
# 	B   -4.0     -2.0      0.0      2.0      4.0
# 	C   -6.0     -3.0      0.0      3.0      6.0

# 	covariance matrix:

# 		A      B      C
# 	A  2.5    5.0    7.5
# 	B  5.0   10.0   15.0
# 	C  7.5   15.0   22.5

# 	Cov(A,A) variance of feature A (with itself)
# 	Cov(A,B) covariance of A and B
# 	Cov(A,C) covariance of A and C
# 	'''

# 	#eigenval and eigenvec
# 	eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 	#sort eigenvalues and vectors
# 	sorted_indexes = np.argsort(eigenvalues[::-1]) #descending, to explain the most variance (highest vals to vecs)
# 	'''
# 	eigenvals   eigenvecs
# 	2.1         v2
# 	0.8         v1
# 	0.5         v3
# 	'''
	
# 	sorted_eigenvals = eigenvalues[sorted_indexes]
# 	sorted_eigenvecs = eigenvectors[:, sorted_indexes]

# 	n_selected_eigenvecs = sorted_eigenvecs[:, n_componens]

# 	#transform the data by projecting it to PCAs #new eigenvecs are now the new axes, dot projects the features to the new axes
# 	features_transformed = np.dot(features_centered, n_selected_eigenvecs)





def filter_raw(data):
	lo_cut = 0.1
	hi_cut = 30

	filtered = data.copy().filter(lo_cut, hi_cut)
	filtered = filtered.notch_filter(freqs=50.0) # remove line noise

	return filtered

def load_raw(files):
	raws = []
	for file in files:
		raws.append(mne.io.read_raw_edf(file, include=channels))
	return raws


print("loading data")
raws = load_raw(files)

raw_filtered = []

print("filtering...")
for raw in raws:
	raw.load_data()
	raw_filtered.append(filter_raw(raw))

print("done filtering")
raws = None

# EXTRACT FEATURES


ica = mne.preprocessing.ICA(method="infomax")

def get_features(epochs, lofreq, hifreq, epoch_type, sfreq):
	feat_mat = []
	y = []
	for idx, epoch in enumerate(epochs):
		mean = np.mean(epoch, axis=0)

		filtered = mne.filter.filter_data(epoch, method="iir", l_freq=lofreq, h_freq=hifreq,
								 sfreq=sfreq)

		activation = filtered - mean

		energy = np.sum(activation ** 2, axis=1)
		power = energy / (len(epoch) * sfreq)

		event_type = epochs.events[idx][2] - 1

		features = np.hstack((mean, energy, power))
		y.append(event_type)

		feat_mat.append(features)
		# print(f'{feat_mat} is feat matrix')
		# print(f'{y} is the event type prediction')
		# if idx == 1:
		# 	# scatter_plot_epochs(feat_mat[0], feat_mat[5])
		# 	heatmap(features)
		# 	break 



	return np.array(feat_mat), np.array(y)

def standardize(arr):
	scaler = StandardScaler()
	return scaler.fit_transform(arr)

def get_all_features(data):
	event_id = {"T1": 1, "T2": 2}
	events, event_dict = mne.events_from_annotations(data)
	
	sfreq = data.info["sfreq"]

	erss = mne.Epochs(data, events, event_id=event_id, tmin=4.1, tmax=5.1,
					  baseline=None)
	#mrcp = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=0,
	#                  baseline=None)
	#erds = mne.Epochs(data, events, event_id=event_id, tmin=-2, tmax=0.0,
	#                  baseline=None)

	ica.fit(erss)
	#ica.fit(erds)
	#ica.fit(mrcp)

	ers_feats, ers_y = get_features(erss, 8, 30, 1, sfreq)
	#eds_feats, eds_y = get_features(erds, 8, 30, 2, sfreq)
	#mrcp_feats, mrcp_y = get_features(erds, 3, 30, 3, sfreq)

	return ers_feats, ers_y



def scatter_plot_epochs_mean(feat_mat_epoch1, feat_mat_epoch2):
	feat_mat_epoch1 = np.array(feat_mat_epoch1)
	feat_mat_epoch2 = np.array(feat_mat_epoch2)

	# Extract mean, energy, and power values for both epochs
	mean_values_epoch1 = feat_mat_epoch1[0]  # Mean for Epoch 1
	mean_values_epoch2 = feat_mat_epoch2[0]  # Mean for Epoch 2

	energy_values_epoch1 = feat_mat_epoch1[1]  # Energy for Epoch 1
	energy_values_epoch2 = feat_mat_epoch2[1]  # Energy for Epoch 2

	power_values_epoch1 = feat_mat_epoch1[2]  # Power for Epoch 1
	power_values_epoch2 = feat_mat_epoch2[2]  # Power for Epoch 2

	# Create a scatter plot for combined mean, energy, and power
	plt.figure(figsize=(10, 6))

	# Plot mean values
	plt.scatter(mean_values_epoch1, mean_values_epoch2, color='green', label='Mean', s=100, alpha=0.7)

	# Plot energy values
	plt.scatter(energy_values_epoch1, energy_values_epoch2, color='blue', label='Energy', s=100, alpha=0.7)

	# Plot power values
	plt.scatter(power_values_epoch1, power_values_epoch2, color='red', label='Power', s=100, alpha=0.7)

	# Add labels, title, and legend
	plt.xlabel("Epoch 1")
	plt.ylabel("Epoch 2")
	plt.title("Combined Mean, Energy, and Power: Epoch 1 vs Epoch 2")
	plt.legend()

	plt.grid(True)
	plt.show()


def scatter_plot_epochs(feat_mat_epoch1, feat_mat_epoch2):
	"""
	Create a 2D scatter plot to compare feature values between two epochs.

	Parameters:
	- feat_mat_epoch1: Feature matrix for Epoch 1 (mean, energy, and power).
	- feat_mat_epoch2: Feature matrix for Epoch 2 (mean, energy, and power).
	"""

	# convert to numpy
	feat_mat_epoch1 = np.array(feat_mat_epoch1)
	feat_mat_epoch2 = np.array(feat_mat_epoch2)

	# extract mean for epoch1, epoch2
	mean_values_epoch1 = feat_mat_epoch1[0]  # Mean for Epoch 1
	mean_values_epoch2 = feat_mat_epoch2[0]  # Mean for Epoch 2


	energy_values_epoch1 = feat_mat_epoch1[1]  # Energy for Epoch 1
	energy_values_epoch2 = feat_mat_epoch2[1]  # Energy for Epoch 2

	power_values_epoch1 = feat_mat_epoch1[2]  # Power for Epoch 1
	power_values_epoch2 = feat_mat_epoch2[2]  # Power for Epoch 2

	# Create scatter plots for mean, energy, and power comparisons
	fig, axes = plt.subplots(1, 3, figsize=(18, 6))

	# Mean comparison plot
	axes[0].scatter(mean_values_epoch1, mean_values_epoch2, color='green', label='Mean', s=100, alpha=0.7)
	axes[0].set_xlabel("Mean (Epoch 1)")
	axes[0].set_ylabel("Mean (Epoch 5)")
	axes[0].set_title("Mean: Epoch 1 vs Epoch 5")
	axes[0].grid(True)

	# Energy comparison plot
	axes[1].scatter(energy_values_epoch1, energy_values_epoch2, color='blue', label='Energy', s=100, alpha=0.7)
	axes[1].set_xlabel("Energy (Epoch 5)")
	axes[1].set_ylabel("Energy (Epoch 5)")
	axes[1].set_title("Energy: Epoch 1 vs Epoch 5")
	axes[1].grid(True)

	# Power comparison plot
	axes[2].scatter(power_values_epoch1, power_values_epoch2, color='red', label='Power', s=100, alpha=0.7)
	axes[2].set_xlabel("Power (Epoch 1)")
	axes[2].set_ylabel("Power (Epoch 5)")
	axes[2].set_title("Power: Epoch 1 vs Epoch 5")
	axes[2].grid(True)

	# Display all plots
	plt.tight_layout()
	plt.show()


# def plot_features(features, labels):
# 	#mean, energy, and power stacked together
# 	# mean_values = features[:, 0]  #first col mean
# 	print(type(features))
# 	features = np.array(features)

# 	mean_values = features[:, 0] #mean
# 	energy_values = features[:, 1]  #second col energy
# 	power_values = features[:, 2]  #third col power

# 	#energy vs power
# 	plt.figure(figsize=(10, 6))
# 	sns.scatterplot(x=energy_values, y=power_values, hue=labels, palette="viridis")
# 	plt.xlabel("energy")
# 	plt.ylabel("power")
# 	plt.title("energy vs power of EEG features")
# 	plt.show()

def heatmap(features):


	# Create labels for the features (grouped in 3s: mean, energy, power)
	# Create labels for the features (grouped in 3s: mean, energy, power)
	# features = np.array(features)
	# labels = []
	# for i in range(features.shape[1] // 3):  # Adjust for the correct number of columns
	# 	labels.extend(["Mean", "Energy", "Power"])

	# # Create a heatmap
	# plt.figure(figsize=(10, 6))
	# ax = sns.heatmap(features, cmap="viridis", cbar=True)

	# # Set tick positions for every third feature
	# tick_positions = np.arange(1.5, features.shape[1], 3)

	# # Ensure that tick labels match the number of ticks (grouped in 3s)
	# ax.set_xticks(tick_positions)
	# ax.set_xticklabels(labels, rotation=90)

	# # Set axis labels and title
	# plt.title("Heatmap of EEG Features Grouped by Mean, Energy, Power")
	# plt.xlabel("Feature Type (Grouped in 3s)")
	# plt.ylabel("Epoch Index")
	# plt.show()

	plt.figure(figsize=(10, 6))
	sns.heatmap(features, cmap="viridis", annot=False)
	plt.title("Heatmap of EEG Features")
	plt.xlabel("Feature Index")
	plt.ylabel("Epoch Index")
	plt.show()


def get(arr):
	x = []
	y = []

	for filtered in arr:
		a, b = get_all_features(filtered)
		
		for i in a:
			x.append(i)

		for i in b:
			y.append(i)
		print("got features")
		# plot_features(x[:-1], y[:-1])
		# break
	
	x = np.array(x)
	y = np.array(y)
	return x, y

x, y = get(raw_filtered)

# pca = PCA()
ft_pca = My_PCA()
scaler = StandardScaler()
reg = LogisticRegression(penalty='l1', solver='liblinear')

pipeline = Pipeline([("Scaler", scaler), ("CSP", ft_pca), ("LogisticRegression", reg)])

pipeline.fit(x, y)

#is this after creating the feature vector?
predict_raw = load_raw(predict)
filtered_predict = []
for raw in predict_raw:
	raw.load_data()
	filtered_predict.append(filter_raw(raw))


px, py = get(filtered_predict)

res = pipeline.predict(px)
print(res)
print(py)

acc = accuracy_score(py, res)
print(acc)
#TODO
#   extract features
#   normalize data
#   perform dimension reduction
#   train model

