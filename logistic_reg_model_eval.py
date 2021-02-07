'''
Functions on running classification modeling on a binary machine-learning problem.
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
from sklearn.metrics import log_loss
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

def log_precision_and_recall_curves(X,y,C=0.95):
    '''
    Arguments: takes in a model name, validation set of data (labels and features).
    Returns: a plot of precision and recall curves.
    '''
    
    #Split Data, fit log regression model:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)
    
    #Standard Scaling of Features
    std = StandardScaler()
    X_train_scaled = std.fit_transform(X_train.values)
    X_val_scaled = std.transform(X_val.values)
    
    logit = LogisticRegression(solver='lbfgs', C = C)
    logit.fit(X_train_scaled, y_train)
    y_pred = logit.predict_proba(X_val_scaled)[:,1]
    
    #Generate curves:
    precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_val, y_pred)

    #Plot:
    plt.figure(dpi=80)
    plt.plot(threshold_curve, precision_curve[1:],label='precision')
    plt.plot(threshold_curve, recall_curve[1:], label='recall')
    plt.legend(loc='lower left')
    plt.xlabel('Threshold (above this probability, label as Show)');
    plt.title('Precision and Recall Curves:');

def log_precision_recall_curve_generator(X,y,C=0.95):
    '''
    Arguments: takes in a model name, and a validation set of data (labels and features).
    Returns: a precision and recall curve.
    '''
    #Split Data, fit log regression model:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)
    
    #Standard Scaling of Features
    std = StandardScaler()
    X_train_scaled = std.fit_transform(X_train.values)
    X_val_scaled = std.transform(X_val.values)
    
    logit = LogisticRegression(solver='lbfgs', C = C)
    logit.fit(X_train_scaled, y_train)

    #Making predictions on the validation set:
    y_preds = logit.predict_proba(X_val_scaled)[:,1]
    no_skill = len(y_val[y_val==1]) / len(y_val)
    
    #Generate curves:
    precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_val, y_preds)

    #Plot:
    plt.figure(dpi=80)
    plt.plot([0,1], [no_skill, no_skill], linestyle = '--', label = 'Chance')
    plt.plot(recall_curve, precision_curve,label='Log. Regression Model')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Precision-Recall Curve");

def log_accuracy_scorer(X, y, threshold=0.5, C=0.95, beta=0.5):
    '''
    Arguments: takes in a set of features X and a target variable y.  Y is a classification (0/1).  Default C is 0.95, can be changed.
    Also includes a threshold, default of 0.5, 
    Returns: Performs logistic regression classification and returns the feature coefficeints and returns the score.
    '''
    #Splitting into train and val sets:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)
    
    #Standard Scaling of Features
    std = StandardScaler()
    X_train_scaled = std.fit_transform(X_train.values)
    X_val_scaled = std.transform(X_val.values)
    
    #Running Logistic Regression, fitting the model:
    logit = LogisticRegression(solver='lbfgs', C = C)
    logit.fit(X_train_scaled, y_train)
        
    #Confusion Matrix, utilizing threshold:
    y_predict = (logit.predict_proba(X_val_scaled)[:,1] >= threshold)
    cm = confusion_matrix(y_val, y_predict)

    #Calculating Validation Accuracy, w/ threshold:
    acc = 0
    y_val_array = np.asarray(y_val, dtype=bool)
    for i, y in enumerate(y_predict):
        if y == y_val_array[i]:
            acc += 1
    val_accuracy = acc/y_predict.shape[0]

    #Precision:
    precision = precision_score(y_val, y_predict)
    
    #Recall:
    recall = recall_score(y_val, y_predict)

    #Scoring F1 and Fbeta
    f1score = f1_score(y_val, y_predict)
    fbetascore = fbeta_score(y_val, y_predict, beta=beta)
    
    #Reporting Results:
    print("The accuracy score for logistic regression w/ threshold of {} is:".format(threshold))
    #print("Training set accuracy: {:6.2f}%".format(100*logit.score(X_train, y_train)))
    print("Validation set accuracy: {:6.2f}%".format(100*val_accuracy))
    print("Additional Model Metrics:")
    print("Validation Set F1 Score: {:6.4f}:".format(f1score))
    print("Validation Set Fbeta Score (beta={}): {:6.4f}".format(beta, fbetascore))
    print("Validation set Precision: {:6.4f}".format(precision))
    print("Validation set recall: {:6.4f} \n".format(recall))
    print("Validation set log-loss score: {:6.4f}".format(log_loss(y_val, y_predict)))
    
    print("Confusion Matrix, Threshold = {}".format(threshold))
    confusion_matrix_generator(cm, 'Logistic Regression')

def log_roc_curve_generator(X,y,C=0.95):
    '''
    Arguments: takes in a model name, and a validation set of data (labels and features).
    Returns: a ROC curve for the data.
    '''
    #Splitting into train and val sets:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)
    
    #Standard Scaling of Features
    std = StandardScaler()
    X_train_scaled = std.fit_transform(X_train.values)
    X_val_scaled = std.transform(X_val.values)
    
    #Running Logistic Regression, fitting the model:
    logit = LogisticRegression(solver='lbfgs', C = C)
    logit.fit(X_train_scaled, y_train)
    y_preds = logit.predict_proba(X_val_scaled)[:,1]
    
    #calculating false positive rate, true positive rate, and the thresholds:
    fpr, tpr, thresholds = roc_curve(y_val, y_preds)
    J = tpr-fpr
    opt = np.argmax(J)
    optimal_threshold = thresholds[opt]

    #Max. TPR (Precision):
    max_pres = thresholds[np.argmax(tpr)]
    
    #Plotting:
    plt.plot(fpr, tpr,lw=2, label = 'ROC Curve')
    plt.plot([0,1],[0,1],c='violet',ls='--', label = 'Chance Predictions')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.legend()
    
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for Horses Showing - Logistic Regression Model');
    print("ROC AUC score = ", roc_auc_score(y_val, y_preds))
    print('Optimal Threshold: {:6.4f}'.format(optimal_threshold))
    print('Threshold for Max Precision: {:6.4f}'.format(max_pres))
    
def log_accuracy_scorer_k_fold(X, y, k=5, threshold=0.5, C=0.95):
    '''
    Arguments: takes in a set of features X and a target variable y.  Y is a classification (0/1).  Default C is 0.95, can be changed.
    Also includes a threshold, default of 0.5, and a k value (default 5) for number of folds to use in k-fold cross validation.
    Returns: Performs logistic regression classification and returns the feature coefficeints and returns the score.
    '''
      
    #Standard Scaling of Features
    std = StandardScaler()
    X_scaled = std.fit_transform(X.values)
    
    #Creating CV arrays:
    X_cv, y_cv = np.array(X_scaled), np.array(y)
    kf = KFold(n_splits=k, shuffle=True, random_state = 12)
    
    #Setting up empty lists for the stats:
    cv_log_acc = []
    cv_log_prec = []
    cv_log_rec = []
    cv_log_fbeta = []
    cv_log_f1 = []
    cv_log_ll = []
    

    counter = 1
    #K-Fold Loop:
    for train_ind, val_ind in kf.split(X_cv, y_cv):
        X_train_scaled, y_train = X_cv[train_ind], y_cv[train_ind]
        X_val_scaled, y_val = X_cv[val_ind], y_cv[val_ind]
        
        #Running Logistic Regression, fitting the model:
        logit = LogisticRegression(solver='lbfgs', C = C)
        logit.fit(X_train_scaled, y_train)

        #Confusion Matrix, utilizing threshold:
        y_predict = (logit.predict_proba(X_val_scaled)[:,1] >= threshold)
        
        #Calculating Validation Accuracy, w/ threshold:
        acc = 0
        y_val_array = np.asarray(y_val, dtype=bool)
        for i, y in enumerate(y_predict):
            if y == y_val_array[i]:
                acc += 1
        val_accuracy = acc/y_predict.shape[0]
        cv_log_acc.append(val_accuracy)
        
        #Precision:
        cv_log_prec.append(precision_score(y_val, y_predict))
    
        #Recall:
        cv_log_rec.append(recall_score(y_val, y_predict))

        #Scoring F1 and Fbeta
        cv_log_f1.append(f1_score(y_val, y_predict))
        cv_log_fbeta.append(fbeta_score(y_val, y_predict, beta=0.5))
        
        #Log Loss:
        cv_log_ll.append(log_loss(y_val, y_predict))

        #Printing Confusion Matrix for each round:
        cm = confusion_matrix(y_val, y_predict)
        print("Confusion Matrix for Fold {}".format(counter))
        print(cm)
        print('\n')
        counter += 1
    
    #Reporting Results:
    print('Logistic Regression Classification w/ KFOLD CV Results (k={}):'.format(k))
    print('Log. Reg Accuracy scores: ', cv_log_acc, '\n')
    print(f'Simple mean cv accuracy: {np.mean(cv_log_acc):.3f} + {np.std(cv_log_acc):.3f} \n')
    print('Log. Reg Precision scores: ', cv_log_prec, '\n')
    print(f'Simple mean cv precision: {np.mean(cv_log_prec):.3f} +- {np.std(cv_log_prec):.3f} \n')
    print('Log. Reg Recall scores: ', cv_log_rec, '\n')
    print(f'Simple mean cv recall: {np.mean(cv_log_rec):.3f} +- {np.std(cv_log_rec):.3f} \n')
    print('Log. Reg Fbeta (beta=0.5) scores: ', cv_log_fbeta, '\n')
    print(f'Simple mean cv Fbeta (beta=0.5): {np.mean(cv_log_fbeta):.3f} +- {np.std(cv_log_fbeta):.3f} \n')
    print('Log. Reg F1 scores: ', cv_log_f1, '\n')
    print(f'Simple mean cv F1: {np.mean(cv_log_f1):.3f} +- {np.std(cv_log_f1):.3f} \n')
    print('Log. Reg Log Loss scores: ', cv_log_ll, '\n')
    print(f'Simple mean cv log loss: {np.mean(cv_log_ll):.3f} +- {np.std(cv_log_ll):.3f} \n')
    
    return logit