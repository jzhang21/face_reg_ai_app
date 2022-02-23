import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image
import pickle

#Load pickle data
df = pickle.load(open('./data/dataframe_images_100_100.pickle', 'rb'))

#Remove missing values
df.dropna(axis=0, inplace=True)

#Data Normalization
##split the data into two parts
X = df.iloc[:,1:].values #independent features
y = df.iloc[:,0].values #dependent

##min max scaling
Xnorm = X / X.max()
y_norm = np.where(y=='female',1,0)

##Save x and y
np.savez('./data/data_10000_norm', Xnorm, y_norm)