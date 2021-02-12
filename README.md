# Predicting Hong Kong Horse Racing Finishes :horse_racing::  
## By: Patrick Bovard  
## *Metis Data Science Bootcamp Winter 2021 Project 3*  

**Description**:  
The goal of this project is to use Horse racing data from the Hong Kong Jockey Club races to predict whether a horse will show (i.e. place first, second, or third) or not in a given race.  This was done by building iteratively improving classification models, using race finish position as the target (show or not show) with various features about the horse as features.  

**Features and Target Variables**:  
- Target Classification: will a given racehorse **Show** (i.e. place 1st, 2nd, or 3rd) or **Not Show** (i.e. place 4th or lower) in a given race  
- Features: Horse Age, Horse Rating, Declared weight of horse and jockey, horse type (i.e. sex), actual weight applied to horse (i.e. race handicapping), etc.  *More to come*  
  
**Data Used**:  
- Data for this project was pulled from [Graham Daley's Horse Racing in HK Kaggle Dataset](https://www.kaggle.com/gdaley/hkracing).  

**Tools Used:**  
- Data Analysis and Model Building: Python, Pandas, Numpy, Scikit-learn, PostgreSQL, SQL Alchemy in Pandas
- Classification Models Attempted: K Nearest Neighbors (KNN), Logistic Regression, Naive Bayes, Random Forest, XGBoost
  - Code for KNN, Logistic Regression, Random Forest, and XGBoost evaluating functions are in the Python_Functions folder
- Visualization: Matplotlib, Seaborn  
  
**Possible Impacts of this project:**  
Possible impacts of this project include utilization on the gambling side of horse racing, to determine if a certain horse should be bet on.  Additionally, a horse owner could use this model to help predict whether their horse is likely to have a good performance in a given race.  While this data focuses on Hong Kong racing, it is hopeful that similar techniques could be applied to other area, such as the United States. 

**Navigating the Repo:**
- SQL Queries are in the SQL_Queries folder.  This includes two files:
  - initial_sql_and_data_scoping.ipynb: initial look through the data, and creating tables to use in machine learning
  - feature_engineering_sql.ipynb: Contains more advanced SQL queries to engineer new features for previous race performances (i.e. rolling average), how a horse compared to the other horses in the race (i.e. comparison of weight, rating, and age to field average), and various other stats
- Code used to train and validate various model types are in the Python_Functions folder.  This includes code for the followind model types:
  - K Nearest Neighbors (knn_model_eval.py)
  - Logistic Regression (logistic_reg_model_eval.py)
  - Random Forest Classification (random_forest_evaluator.py)
  - XGBoost Classification (xgboost_evaluator.py)
- Slides prepared for my presentation of this project are in the Final_Presentation folder.  This includes both Powerpoint and PDF files.
- Data utilized in this project is in the Data folder.  This includes the original race and run csv files, as well as the various round of Pandas dataframes I engineerd in .pkl files.  The data used in the final model was **training_horses_final.pkl** for training/validation and **testing_horses_final.pkl** for testing the model.
- Early Models: earlier models are contained in the Early_Models folder of the repo.  This contains the following, in order of earliest to latest (earliest at the top of the model):
  - initial_model_eda.ipynb
  - second_model.ipynb
  - model_3_random_forest.ipynb
  - xgboost_model.ipynb
  - race_split_data_test.ipynb
