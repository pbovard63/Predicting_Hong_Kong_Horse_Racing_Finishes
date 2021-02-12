'''
Functions to evaluate an XGBoost classification model for train/validation purposes.
'''
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score, auc
import pandas as pd
from importlib import reload
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from matplotlib import pyplot
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from xgboost import plot_importance

def confusion_matrix_generator(confusion_matrix, name):
    '''
    Arguments: takes in the basic confusion matrix, and the name of the model to title the output graph.
    Returns: a visually appealing Seaborn confusion matrix graph.
    '''
    plt.figure(dpi=150)
    sns.heatmap(confusion_matrix, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
           xticklabels=['Non-Show', 'Show'],
           yticklabels=['Non-Show', 'Show'])

    plt.xlabel('Predicted Finish - Show or No Show')
    plt.ylabel('Actual Finish - Show or No Show')
    plt.title('{} confusion matrix'.format(name));

def xgboost_eval(X,y):
    '''
    Arguments: takes in a set of features X and a target variable y.  Y is a classification (0/1).  
    Returns: Performs XGBoost classification and returns the scores. 
    The XGBoost parameters used are: n_estimators=100, learning_rate = 0.05, max_depth=6, min_child_weight = 3, use_label_encoder = False, objective = 'binary:logistic'
    '''
    #SPlit the data into train and validation sets:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)

    #Fitting the Model:
    xgb_model = XGBClassifier(n_estimators=100, learning_rate = 0.05, max_depth=6,
                         min_child_weight = 3, use_label_encoder = False, objective = 'binary:logistic')
    xgb_model.fit(X_train, y_train)

    #Scoring the Model:
    y_pred = xgb_model.predict(X_val)
    train_accuracy = xgb_model.score(X_train, y_train)
    val_accuracy = xgb_model.score(X_val, y_val)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    fbetascore = fbeta_score(y_val, y_pred, beta=0.5)
    f1score = f1_score(y_val, y_pred)

    #Printing Scores:
    print('Model Training Set Accuracy: {}'.format(train_accuracy))
    print('Model Validation Set Accuracy: {}'.format(val_accuracy))
    print('Model Precision Score: {}'.format(precision))
    print('Model Fbeta (beta = 0.5) Score: {}'.format(fbetascore))
    print('Model Recall Score: {}'.format(recall))
    print('Model F1 Score: {}'.format(f1score))
    
    #Confusion Matrix:
    cm = confusion_matrix(y_val, y_pred)
    confusion_matrix_generator(cm, 'XGBoost')

    return xgb_model

def xgboost_feature_importance(model):
    '''
    Arguments: a model and a list of feature names.
    Returns: a plot with feature importance given the model.
    '''
    xgb.plot_importance(model)
    xgb.plot_importance(mode, importance_type='gain')

def xgboost_eval_kfold(X,y, k=5, threshold = 0.5):
    '''
    Arguments: takes in a set of features X and a target variable y.  Y is a classification (0/1).  Threshold can be changed from 0.5 default.  Uses k=5 as default k for K-Fold cross validation.
    Returns: Performs XGBoost classification and returns the scores. 
    The XGBoost parameters used are: n_estimators=100, learning_rate = 0.05, max_depth=6, min_child_weight = 3, use_label_encoder = False, objective = 'binary:logistic'
    '''
    X_cv, y_cv = np.array(X), np.array(y)
    kf = KFold(n_splits=k, shuffle=True, random_state = 12)
    
    #Setting up empty lists:
    cv_xg_acc = []
    cv_xg_prec = []
    cv_xg_rec = []
    cv_xg_fbeta = []
    cv_xg_f1 = []
    
    #K-Fold Loop:
    i = 1
    for train_ind, val_ind in kf.split(X_cv,y_cv):
        X_train, y_train = X_cv[train_ind], y_cv[train_ind]
        X_val, y_val = X_cv[val_ind], y_cv[val_ind] 
    
        #Running Model and making predictions:
        xgb_model = XGBClassifier(n_estimators=100, learning_rate = 0.05, max_depth=6,
                         min_child_weight = 3, use_label_encoder = False, objective = 'binary:logistic')
        xgb_model.fit(X_train, y_train)

        y_pred = (xgb_model.predict_proba(X_val)[:, 1] >= threshold)

        #Printing Confusion Matrix for each round:
        cm = confusion_matrix(y_val, y_pred)
        print("Confusion Matrix for Fold {}".format(i))
        print(cm)
        print('\n')
        i += 1
    
        #Scores:
        cv_xg_acc.append(xgb_model.score(X_val, y_val))
        cv_xg_prec.append(precision_score(y_val, y_pred))
        cv_xg_rec.append(recall_score(y_val, y_pred))
        cv_xg_fbeta.append(fbeta_score(y_val, y_pred, beta=0.5))
        cv_xg_f1.append(f1_score(y_val, y_pred))
    
    print('XGBoost Classification w/ KFOLD CV Results (k={}, threshold = {}):'.format(k, threshold))
    print('XG Boost Accuracy scores: ', cv_xg_acc, '\n')
    print(f'Simple mean cv accuracy: {np.mean(cv_xg_acc):.3f} + {np.std(cv_xg_acc):.3f}')
    print('XG Boost Precision scores: ', cv_xg_prec, '\n')
    print(f'Simple mean cv precision: {np.mean(cv_xg_prec):.3f} +- {np.std(cv_xg_prec):.3f}')
    print('XG Boost Recall scores: ', cv_xg_rec, '\n')
    print(f'Simple mean cv recall: {np.mean(cv_xg_rec):.3f} +- {np.std(cv_xg_rec):.3f}')
    print('XG Boost Fbeta (beta=0.5) scores: ', cv_xg_fbeta, '\n')
    print(f'Simple mean cv Fbeta (beta=0.5): {np.mean(cv_xg_fbeta):.3f} +- {np.std(cv_xg_fbeta):.3f}')
    print('XG Boost F1 scores: ', cv_xg_f1, '\n')
    print(f'Simple mean cv F1: {np.mean(cv_xg_f1):.3f} +- {np.std(cv_xg_f1):.3f}')

    return xgb_model