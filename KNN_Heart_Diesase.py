import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# 2. Importing Dataset
dataset = pd.read_csv('Heart_Disease.csv')
features =dataset.iloc[:,0:9].values
labels= dataset.iloc[:,-1].values

import statsmodels.regression.linear_model as sm
features= np.append(arr=np.ones((462,1)).astype(int),values=features,axis=1)
features_obj=features[:,[0,1,2,3,4,5,6,7,8,9]]

while True:
    labels=labels.astype(float)
    features_obj=features_obj.astype(float)
    ols=sm.OLS(endog = labels , exog=features_obj).fit()
    p_values=ols.pvalues
    if p_values.max()> 0.05:
        features_obj = np.delete(features_obj, p_values.argmax(), 1)
    else:
        break
    
features_obj=features_obj[:,[1,2,3,4,5]]   

from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features_obj,labels,test_size=0.25,random_state=0)

from sklearn.svm import SVC
classifier=SVC(kernel = "rbf" , random_state=0)
classifier.fit(features_train,labels_train)

##Predicting the class labels 
labels_pred =classifier.predict(features_test)

#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels_test,labels_pred)

#Accuracy Score
from sklearn.metrics import accuracy_score
ac = accuracy_score(labels_test,labels_pred)

