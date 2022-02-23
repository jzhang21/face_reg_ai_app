import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
import cv2

#Load Models & Pickle Files
haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')
mean = pickle.load(open('./model/mean_preprocess.pickle', 'rb'))
model_svm = pickle.load(open('./model/model_svm.pickle', 'rb'))
model_pca = pickle.load(open('./model/pca_50.pickle', 'rb'))
print('Models Loaded Succesffuly')

gender_pre = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX

def pipeline_model(img, color='rgb'):
    #Convert to Gray Scale
    if color == 'bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: 
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #Crop Face
    faces = haar.detectMultiScale(gray, 1.5, 3)
    for x,y,w,h in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2) #draw rectangle
        roi = gray[y:y+h, x:x+w] #crop image
        #Normalize
        roi = roi / 255.0
        #Resize to 100x100
        if roi.shape[1] > 100:
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi, (100,100), cv2.INTER_CUBIC)
        #Flattening 1x10000
        roi_reshape = roi_resize.reshape(1,10000) #1, -1
        #Subtract Mean
        roi_mean = roi_reshape - mean
        #Get Eigen Image
        eigen_image = model_pca.transform(roi_mean)
        #Pass to ML Model (SVM)
        results = model_svm.predict_proba(eigen_image)[0]
        #Organize Results
        predict = results.argmax() # Max Value. 0 = Male, 1 = Female 
        score = results[predict]
        #Output
        text = "%s : %0.2f"%(gender_pre[predict], score)
        cv2.putText(img, text, (x, y), font, 1, (0,255,0), 2)
    return img
    
#cv2.imshow('Gender Prediction', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

from PIL import Image

#Test Image
test_data_path = './data/male_000281.jpg'
color = 'bgr'
#Read Image
img = Image.open(test_data_path)
img = np.array(img)
img = pipeline_model(img)
plt.imshow(img)


#Apply Pipeline Model to Video
cap = cv2.VideoCapture('./data/video.mp4')

while True:
    ret, frame = cap.read() #bgr
    if ret == False:
        break
    frame = pipeline_model(frame, color='bgr')
    cv2.imshow('Gender Detector', frame)
    if cv2.waitKey(10) == 27: #esc key
        break
    
cv2.destroyAllWindows()
cap.release()




