'''
Functions on running Random Forest classification modeling on a binary machine-learning problem.
'''
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score, auc
import pandas as pd
from importlib import reload
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot

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

def random_forest_eval(X,y):
    '''
    Takes in a X and y data set, returns a random forest model eval.
    '''
    #SPlit the data into train and validation sets:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)

    #Fitting the Model:
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    #Scoring the Model:
    y_pred = rf_model.predict(X_val)
    train_accuracy = rf_model.score(X_train, y_train)
    val_accuracy = rf_model.score(X_val, y_val)
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
    confusion_matrix_generator(cm, 'Random Forest')

    return rf_model

def rf_model_feature_importance(features, model):
    '''
    Arguments: a model and a list of feature names.
    Returns: a plot with feature importance.
    '''
    importance = model.feature_importances_
    pyplot.bar(features, importance)
    pyplot.xticks(ticks = features, labels=features, rotation=90)
    pyplot.title('Feature Importance Chart - Random Forest')
    pyplot.show()

def random_forest_eval_oversampled(X,y):
    '''
    Takes in a X and y data set, returns a random forest model eval.
    '''
    from imblearn.over_sampling import RandomOverSampler
    #SPlit the data into train and validation sets:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)

    #Oversampling Train Data:
    ros = RandomOverSampler(random_state=0)
    X_train_resampled, y_train_resampled = ros.fit_sample(X_train,y_train)

    #Fitting the Model:
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_resampled, y_train_resampled)

    #Scoring the Model:
    y_pred = rf_model.predict(X_val)
    train_accuracy = rf_model.score(X_train_resampled, y_train_resampled)
    val_accuracy = rf_model.score(X_val, y_val)
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
    confusion_matrix_generator(cm, 'Random Forest')

    return rf_model