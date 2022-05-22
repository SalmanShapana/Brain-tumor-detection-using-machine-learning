import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics, tree



#Installing Dataset

#i create two folders , traning and put inside yes_truma all this in main folder brain_tumor, no_truma and create folder testing and also put inside yes_truma, no_truma

path = os.listdir('brain_tumor/Traning/')
classes = {'no_truma':0, 'yes_truma':1}
X = []
Y = []

for cls in classes:
    pth = 'brain_tumor/Traning/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])


#Extract features in the tumor brain detection


X = np.array(X)
Y = np.array(Y)

X_updated = X.reshape(len(X), -1)
print(np.unique(Y))

print(pd.Series(Y).value_counts())
print(X.shape, X_updated.shape)
plt.imshow(X[0], cmap='gray')
# plt.show()

X_updated = X.reshape(len(X), -1)
print(X_updated.shape)

xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,test_size=.20)

print(xtrain.shape, xtest.shape)
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())

xtrain = xtrain/255
xtest = xtest/255

print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
print(xtrain.shape, xtest.shape)

pca = PCA(.98)

pca_train = xtrain
pca_test = xtest


warnings.filterwarnings('ignore')


#Model Training"LogisticRegression, SVC, GaussianNB, tree.DecisionTreeClassifier

lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)

sv = SVC()
sv.fit(xtrain, ytrain)

knn_1=GaussianNB()
knn_1.fit(xtrain,ytrain)

Dt = tree.DecisionTreeClassifier()
Dt.fit(xtrain, ytrain)

#Now compare scores accurate of above four models
def scores_models():
 print("Training Score of Logistic:", lg.score(xtrain, ytrain)*100)
 print("Testing Score  of Logistic:", lg.score(xtest, ytest)*100)

 print("\nTraining Score of SVM:", sv.score(xtrain, ytrain)*100)
 print("Testing Score of SVM:", sv.score(xtest, ytest)*100)

 print("\nTraining Score of KNN:", knn_1.score(xtrain, ytrain)*100)
 print("Testing Score OF KNN:", knn_1.score(xtest, ytest)*100)

 print("\nTraining Score of Dt:", Dt.score(xtrain, ytrain)*100)
 print("Testing Score OF Dt:", Dt.score(xtest, ytest)*100)


#Predict test dataset 

pred = sv.predict(xtest)
print(pred)

misclassified=np.where(ytest!=pred)
print(misclassified)

print("Total Misclassified Samples: ",len(misclassified[0]))
print(pred[36],ytest[36])


#Result

dec = {0:'No Tumor', 1:'Positive Tumor'}

def testing_no():
 plt.figure(figsize=(18,10))
 p = os.listdir('brain_tumor/Testing/')
 c=1
 for i in os.listdir('brain_tumor/Testing/no_truma/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread('brain_tumor/Testing/no_truma/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1
    plt.show()

def testing_yes():
 plt.figure(figsize=(18,10))
 p = os.listdir('brain_tumor/Testing/')
 c=1
 for i in os.listdir('brain_tumor/Testing/yes_truma/')[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread('brain_tumor/Testing/yes_truma/'+i,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1
    plt.show()

scores_models()
testing_no()
testing_yes()


