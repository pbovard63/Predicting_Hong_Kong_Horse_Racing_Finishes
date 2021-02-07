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
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

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
    

def KNN_accuracy_scorer(X, y, n = 5, beta=0.5):
    '''
    Arguments: takes in a set of features X and a target variable y.  Y is a classification (0/1).  Default n is 5, can be changed.
    Returns: Performs K nearest neighbors classification and returns the feature coefficeints and returns the score.
    '''
    #Splitting into train and val sets:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)

    #Standard Scaling of Features
    std = StandardScaler()
    X_train_scaled = std.fit_transform(X_train.values)
    X_val_scaled = std.transform(X_val.values)
    
    #Running KNN:
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_val_scaled)

    #Precision:
    precision = precision_score(y_val, y_pred)
    
    #Recall:
    recall = recall_score(y_val, y_pred)

    #Scoring F1 and Fbeta
    f1score = f1_score(y_val, y_pred)
    fbetascore = fbeta_score(y_val, y_pred, beta=beta)
    
    #Confusion Matrix:
    cm = confusion_matrix(y_val, y_pred)
    
    #scoring:
    print("The accuracy score for kNN is")
    print("Training: {:6.2f}%".format(100*knn.score(X_train_scaled, y_train)))
    print("Validation set: {:6.2f}%".format(100*knn.score(X_val_scaled, y_val)))
    print("Validation Set F1 Score: {:6.4f}:".format(f1_score(y_val, y_pred)))
    print("Validation Set Fbeta Score (beta={}): {:6.4f}".format(beta, fbetascore))
    print("Validation set Precision: {:6.4f}".format(precision))
    print("Validation set recall: {:6.4f} \n".format(recall))
    print(classification_report(y_val, y_pred))
    confusion_matrix_generator(cm, 'KNN')
    
def KNN_accuracy_scorer_f_fold(X, y, n = 5, k=5):
    '''
    Arguments: takes in a set of features X and a target variable y.  Y is a classification (0/1).  Default n is 5, can be changed.
    Returns: Performs K nearest neighbors classification and returns the feature coefficeints and returns the score.
    '''

    #Standard Scaling of Features
    std = StandardScaler()
    X_scaled = std.fit_transform(X.values)
    
    #Creating CV arrays:
    X_cv, y_cv = np.array(X_scaled), np.array(y)
    kf = KFold(n_splits=k, shuffle=True, random_state = 12)
    
    #Setting up empty lists for the stats:
    cv_knn_acc = []
    cv_knn_prec = []
    cv_knn_rec = []
    cv_knn_fbeta = []
    cv_knn_f1 = []
    
    i = 1
    #K-Fold Loop:
    for train_ind, val_ind in kf.split(X_cv, y_cv):
        X_train_scaled, y_train = X_cv[train_ind], y_cv[train_ind]
        X_val_scaled, y_val = X_cv[val_ind], y_cv[val_ind] 
    
        #Running KNN:
        knn = KNeighborsClassifier(n_neighbors = n)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_val_scaled)

            
        #Accuracy:
        cv_knn_acc.append(knn.score(X_val_scaled, y_val))
                          
        #Precision:
        cv_knn_prec.append(precision_score(y_val, y_pred))
    
        #Recall:
        cv_knn_rec.append(recall_score(y_val, y_pred))

        #Scoring F1 and Fbeta
        cv_knn_f1.append(f1_score(y_val, y_pred))
        cv_knn_fbeta.append(fbeta_score(y_val, y_pred, beta=0.5))

        #Printing Confusion Matrix for each round:
        cm = confusion_matrix(y_val, y_pred)
        print("Confusion Matrix for Fold {}".format(i))
        print(cm)
        print('\n')
        i += 1

    
    #Reporting Results:
    print('KNN Classification w/ KFOLD CV Results (k={}):'.format(k))
    print('KNN Accuracy scores: ', cv_knn_acc, '\n')
    print(f'Simple mean cv accuracy: {np.mean(cv_knn_acc):.3f} + {np.std(cv_knn_acc):.3f} \n')
    print('KNN Precision scores: ', cv_knn_prec, '\n')
    print(f'Simple mean cv precision: {np.mean(cv_knn_prec):.3f} +- {np.std(cv_knn_prec):.3f} \n')
    print('KNN Recall scores: ', cv_knn_rec, '\n')
    print(f'Simple mean cv recall: {np.mean(cv_knn_rec):.3f} +- {np.std(cv_knn_rec):.3f} \n')
    print('KNN Fbeta (beta=0.5) scores: ', cv_knn_fbeta, '\n')
    print(f'Simple mean cv Fbeta (beta=0.5): {np.mean(cv_knn_fbeta):.3f} +- {np.std(cv_knn_fbeta):.3f} \n')
    print('KNN F1 scores: ', cv_knn_f1, '\n')
    print(f'Simple mean cv F1: {np.mean(cv_knn_f1):.3f} +- {np.std(cv_knn_f1):.3f} \n')

    return knn

