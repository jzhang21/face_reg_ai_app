import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image

#Load numpy zip
data = np.load('./data/data_10000_norm.npz')

X = data['arr_0'] #independent features
y = data['arr_1'] #dependent

#Eigen Image
X1 = X - X.mean(axis=0)

from sklearn.decomposition import PCA

pca = PCA(n_components=None, whiten=True, svd_solver='auto')

x_pca = pca.fit_transform(X1)

eigen_ratio = pca.explained_variance_ratio_
eigen_ratio_cum = np.cumsum(eigen_ratio)

pca_50 = PCA(n_components=50, whiten=True, svd_solver='auto')
x_pca_50 = pca_50.fit_transform(X1)

#Save pca
import pickle 
pickle.dump(pca_50, open('./model/pca_50.pickle', 'wb'))

#Consider 50 components and inverse transform
x_pca_inv = pca_50.inverse_transform(x_pca_50)

#Save
np.savez('./data/data_pca_50_y_mean.pickle', x_pca_50, y, X.mean(axis=0))
