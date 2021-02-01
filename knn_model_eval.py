'''
Functions to evaluate a K-Nearest Neighbors (KNN) model for a machine learning classification problem.
'''
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score
import pandas as pd
from importlib import reload
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

plt.rcParams['figure.figsize'] = (9, 6)
sns.set(context='notebook', style='whitegrid', font_scale=1.2)

def confusion_matrix_generator(confusion_matrix, name):
    '''
    Arguments: takes in the basic confusion matrix, and the type of regression.
    Returns: a visually appealing Seaborn confusion matrix.
    '''
    plt.figure(dpi=150)
    sns.heatmap(confusion_matrix, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
           xticklabels=['Non-Show', 'Show'],
           yticklabels=['Non-Show', 'Show'])

    plt.xlabel('Predicted Finish - Show or No Show')
    plt.ylabel('Actual Finish - Show or No Show')
    plt.title('{} confusion matrix'.format(name));
    

def KNN_accuracy_scorer(X, y, n = 5):
    '''
    Arguments: takes in a set of features X and a target variable y.  Y is a classification (0/1).  Default n is 5, can be changed.
    Returns: Performs K nearest neighbors classification and returns the feature coefficeints and returns the score.
    '''
    #Splitting into train and val sets:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5)

    #Standard Scaling of Features
    std = StandardScaler()
    std.fit(X_train.values)
    X_train_scaled = std.transform(X_train.values)
    X_val_scaled = std.transform(X_val.values)
    
    #Running KNN:
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_val_scaled)

    #Precision:
    precision = precision_score(y_val, y_pred)
    
    #Recall:
    recall = recall_score(y_val, y_pred)
    
    #Confusion Matrix:
    cm = confusion_matrix(y_val, y_pred)
    
    #scoring:
    print("The accuracy score for kNN is")
    print("Training: {:6.2f}%".format(100*knn.score(X_train_scaled, y_train)))
    print("Validation set: {:6.2f}%".format(100*knn.score(X_val_scaled, y_val)))
    print("Validation Set F1 Score: {:6.4f}:".format(f1_score(y_val, y_pred)))
    print("Validation set Precision: {:6.4f}".format(precision))
    print("Validation set recall: {:6.4f} \n".format(recall))
    confusion_matrix_generator(cm, 'KNN')
