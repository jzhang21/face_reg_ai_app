import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = np.load('./data/data_pca_50_y_mean.pickle.npz')

X = data['arr_0']
y = data['arr_1']
mean = data['arr_2']

#SVM Model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

#Training ML Model
from sklearn.svm import SVC
model = SVC(C=1.0, kernel='rbf', gamma=0.01, probability=True)
model.fit(x_train, y_train)

#Score - Model Accuracy
print(model.score(x_train, y_train))
print(model.score(x_test, y_test))

#Model Evaluation
'''
- Confusion Matrix
- Classification Report
- Kappa Score
- ROC and AUC (Probability)
'''

from sklearn import metrics
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)

cm = metrics.confusion_matrix(y_test, y_pred)
cm = np.concatenate((cm, cm.sum(axis=0).reshape(1,-1)), axis=0)
cm = np.concatenate((cm, cm.sum(axis=1).reshape(-1,1)), axis=1)
print(cm)

cr = metrics.classification_report(y_test, y_pred, target_names=['male','female'], output_dict=True)
print(pd.DataFrame(cr).T)

print(metrics.cohen_kappa_score(y_test, y_pred))

#female
fpr,tpr,thresh = metrics.roc_curve(y_test, y_prob[:,1])
auc_s = metrics.auc(fpr, tpr)
plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, '-.')
plt.plot([0,1],[0,1],'b--')
for i in range(0, len(thresh), 20):
    plt.plot(fpr[i], tpr[i], '^')
    plt.text(fpr[i], tpr[i], "%0.2f"%thresh[i])

plt.legend(['AUC Score = %0.2f'%auc_s])
    
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics')
plt.show()

# Hyper Parameter Tuning
model_tune = SVC()
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[1,10,20,30,50,100],
              'kernel':['rbf', 'poly'],
              'gamma':[0.1,0.05,0.01,0.001,0.002,0.005],
              'coef0':[0,1]}
model_grid = GridSearchCV(model_tune, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1)
model_grid.fit(X,y)
model_grid.best_index_
model_grid.best_params_
model_grid.best_score_

#Build ML Model w/ Best Params
model_best = SVC(C=10, kernel='rbf', gamma=0.005, probability=True) #coef0 is 0 by default
model_best.fit(x_train, y_train)
print(model_best.score(x_test, y_test))

#Save Model
import pickle
pickle.dump(model_best, open('model_svm.pickle', 'wb'))
pickle.dump(mean, open('./model/mean_preprocess.pickle', 'wb'))   
