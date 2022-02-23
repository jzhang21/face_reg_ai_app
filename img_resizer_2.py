import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image

female = glob('./data/crop/female_crop/*.png')
male = glob('./data/crop/male_crop/*.png')

path = female + male
        
#Get Size of Image
def getSize(path):
    img = Image.open(path)
    return img.size[0]

#Create Dataframe
df = pd.DataFrame(data=path, columns=['path'])

df['size'] = df['path'].apply(getSize)
        
#Remove Imgs < 54x54
df_new = df[df['size'] > 60]

#Add Gender to DF
def gender(string):
    try:
        return string.split('_')[0].split('/')[-1]
    except:
        return None
    
df['gender'] = df['path'].apply(gender)
        
#Resize Images
def resize_img(path_to_resize):
    try:
        #Step 1: Read Image
        img = cv2.imread(path_to_resize)
        #Step 2: Convert into Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Step 3: Resize
        size = gray.shape[0]
        if size >= 100:
            gray_re = cv2.resize(gray, (100,100), cv2.INTER_AREA)
        else:
            gray_re = cv2.resize(gray, (100,100), cv2.INTER_CUBIC)
        #Step 4: Flatten Image
        return gray_re.flatten()
    except:
        return None

df_new['gender'] = df_new['path'].apply(gender)
df_new['structure_data'] = df_new['path'].apply(resize_img)
        
#Copy and Expand Columns
df1 = df_new['structure_data'].apply(pd.Series)
df2 = pd.concat((df_new['gender'], df1), axis=1)

import pickle
pickle.dump(df2, open('./data/dataframe_images_100_100.pickle', 'wb'))