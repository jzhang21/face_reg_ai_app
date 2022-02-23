import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

femalepath = glob('./data/female/*.jpg')
malepath = glob('./data/male/*.jpg')

haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')

# find face then crop and save
def extract_images(path, gender, i):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, 1.5, 5)
    for x,y,w,h in faces:
        roi = img[y:y+h, x:x+w]
        if gender == 'male':
            cv2.imwrite('./data/crop/male_crop/{}_{}.png'.format(gender,i), roi)
        else:
            cv2.imwrite('./data/crop/female_crop/{}_{}.png'.format(gender,i), roi)

#extract_images(femalepath[0], 'female', 1)

#crop all female pictures
for i, path in enumerate(femalepath):
    try:
        extract_images(path, 'female', i)
        print('INFO: {}/{} processed successfuly'.format(i, len(femalepath)))
    except:
        print('INFO: {}/{} cannot be processed'.format(i, len(femalepath)))
        
#crop all male pictures
for i, path in enumerate(malepath):
    try:
        extract_images(path, 'male', i)
        print('INFO: {}/{} processed successfuly'.format(i, len(malepath)))
    except:
        print('INFO: {}/{} cannot be processed'.format(i, len(malepath)))
            
        
        
        
        
        
        
        
        
        
        
        
        
        