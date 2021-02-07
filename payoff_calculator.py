'''
Functions to calculate the gambling payoff of a horse racing model.
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
from sklearn.model_selection import KFold

def payoff_calculator_with_ss(model, df, feature_columns):
    '''
    Arguments: takes in a dataframe and a model, as well as a list of columns in the dataframe to use as features.  Will prompt user to input a bet amount.
    NOTE: THIS REQUIRES A MODEL THAT NEEDS TO HAVE STANDARD SCALED FEATURES
    Returns: the payout, based on the bets.
    '''
    #Standard Scaling the features
    X_new = df[feature_columns]
    std = StandardScaler()
    X_new_scaled = std.fit_transform(X_new.values)

    #Making predictions:
    df['prediction'] = model.predict(X_new_scaled)
    
    #Making a new pandas dataframe, to run calculations on:
    calculator_df = df[df.place_odds.notnull() & df.prediction == 1][['place_odds', 'prediction', 'show']]

    #Calculating:
    bet_size = input('How much would you like to bet per horse?    ')
    bet = int(bet_size)
    #Subtracting the total bets:
    wallet = -(bet*calculator_df.shape[0])
    for i, value in enumerate(calculator_df['show']):
        if calculator_df.iloc[i, 2] == 1:
            wallet += (bet*(calculator_df.iloc[i, 0]))
    print('Total Bets Placed: {}'.format(calculator_df.shape[0]))
    print('Total Amount Wagered: ${}'.format(calculator_df.shape[0]*bet))
    print('Total Winnings: ${}'.format(wallet))

def payoff_calculator_without_ss(model, df, feature_columns):
    '''
    Arguments: takes in a dataframe and a model, as well as a list of columns in the dataframe to use as features.  Will prompt user to input a bet amount.
    NOTE: THIS REQUIRES A MODEL THAT DOES NOT NEED TO HAVE STANDARD SCALED FEATURES
    Returns: the payout, based on the bets.
    '''
    #Standard Scaling the features
    X_new = df[feature_columns]

    #Making predictions:
    df['prediction'] = model.predict(X_new)
    
    #Making a new pandas dataframe, to run calculations on:
    calculator_df = df[df.place_odds.notnull() & df.prediction == 1][['place_odds', 'prediction', 'show']]

    #Calculating:
    bet_size = input('How much would you like to bet per horse?    ')
    bet = int(bet_size)
    #Subtracting the total bets:
    wallet = -(bet*calculator_df.shape[0])
    for i, value in enumerate(calculator_df['show']):
        if calculator_df.iloc[i, 2] == 1:
            wallet += (bet*(calculator_df.iloc[i, 0]))
    print('Total Bets Placed: {}'.format(calculator_df.shape[0]))
    print('Total Amount Wagered: ${}'.format(calculator_df.shape[0]*bet))
    print('Total Winnings: ${}'.format(wallet))

