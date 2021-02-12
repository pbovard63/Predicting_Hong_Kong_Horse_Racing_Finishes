# Predicting Hong Kong Horse Racing Finishes :horse_racing::  
## By: Patrick Bovard  
## *Metis Data Science Bootcamp Winter 2021 Project 3*  

### Project Introduction: 
**Description**:  
The goal of this project is to use Horse racing data from the Hong Kong Jockey Club races to predict whether a horse will show (i.e. place first, second, or third) or not in a given race.  This was done by building iteratively improving classification models, using race finish position as the target (show or not show) with various features about the horse as features.  

**Features, Classification Classes, and Key Evaluation Metrics**:  
- Target Classification: will a given racehorse **Show** (i.e. place 1st, 2nd, or 3rd) or **Not Show** (i.e. place 4th or lower) in a given race  
- Features: several features were used in the final model, which can be found in Final_Modeling_nb.ipynb in this repo.  These features fell into a few main categories:
  - Data directly from the Original Dataset:
    - Horse Country, Horse Age, Horse Rating, Declared Weight (i.e. weight of horse and jockey), Actual Weight (applied weight to horse), Draw (starting position), Place Odds, Race Distance, Horses in the Race, Horse Sex
  - Engineered Features Using SQL Queries in feature_engineering_sql.ipynb (in SQL_Queries folder):
    - Three Race Rolling Averages (i.e. each horse's performance in the three previous races): Finish Position, Lengths Behind Leader, Finish Time, Race Distance/Finish Time, Change in Time/Lengths Behind Leader within the race
    - Other Race Stats: Career Races, Career Top 3 Finishes, Top 3 Finishes in the Last 5 races, Days Since Last Race, 
    - Comparison to the Race Field (rank and difference from field's average value): Age, Horse Declared Weight, Horse Applied Weight
    - Top 10 Jockey and Trainer (yes/no), based on the jockey/trainer's record of success
    - Some Strong Interaction Terms
- Key Evaluation Metrics: for this project, the major evaluation metrics for models were **Precision** and **FBeta (beta = 0.5)**.  In this use case of sports betting, incorrect predictions to show mean lost money, so precision is more important than recall.  However, I want to also ensure the model is making predictions, so FBeta with a Beta of 0.5 keeps an eye on recall, while weighting precision about twice as heavily.
  
**Data Used**:  
- Data for this project was pulled from [Graham Daley's Horse Racing in HK Kaggle Dataset](https://www.kaggle.com/gdaley/hkracing).  

**Tools Used:**  
- Data Analysis and Model Building: Python, Pandas, Numpy, Scikit-learn, PostgreSQL, SQL Alchemy in Pandas
- Classification Models Attempted: K Nearest Neighbors (KNN), Logistic Regression, Naive Bayes, Random Forest, XGBoost
  - Code for KNN, Logistic Regression, Random Forest, and XGBoost evaluating functions are in the Python_Functions folder
- Visualization: Matplotlib, Seaborn  

**Possible Impacts of this project:**  
Possible impacts of this project include utilization on the gambling side of horse racing, to determine if a certain horse should be bet on.  With a precision of 59.1%, this model is showing success, as it is predicting more successful picks than incorrect.

Additionally, a horse owner could use this model to help predict whether their horse is likely to have a good performance in a given race.  While this data focuses on Hong Kong racing, it is hopeful that similar techniques could be applied to other area, such as the United States.   

### Final Model Results:
The final model for this project utilized an XGBoost Classifier, with hyper parameters tuned using RandomizedSearchCV.  The XGBoost was chose for a number of reasons that are outlined in the modeling notebooks, but these include: 
- Strong Train/Validation Accuracy Performance (60.3% train/validation precision in the final form, as compared to 58.3% for RandomForest Classification
- Higher Performance on FBeta as compared to other models (0.474 FBeta in train/validation)
- Ability to handle multi-class (i.e. ordinal, numerical, and categorical data) all in one  

On the final training set, the XGBoost classification model had a **59.1% precision** with an **FBeta Score of 0.454**.  

Final Model Confusion Matrix on the Test Set: 
![](images/final_model_confusion_matrix_image.png)  

Tuned XGBoost Hyperparameters used:
| Hyperparameter      | Value Used in Final Model |
|---------------------|---------------------------|
| n_estimators        | 132                       |
| learning rate       | 0.05                      |
| max_depth           | 3                         |
| min_child_weight    | 6                         |
| subsample           | 0.4                       |
| colsample_bytree    | 0.4                       |

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
  - initial_model_eda.ipynb: Logistic Regression, K-Nearest Neighbors (KNN)
  - second_model.ipynb: Logistic Regression, KNN
  - model_3_random_forest_naive_bayes.ipynb: RandomForest, Naive Bayes
  - model_4_xgboost.ipynb: XGBoost
  - race_split_data_test.ipynb: Model Comparison between KNN, RandomForest, XGBoost, and Logistic Regression 
- Final Model: selection of final model, and tuning of the XGBoost classifier, along with Test Set performance, results, and error analysis.
  - Final_Modeling_nb.ipynb
