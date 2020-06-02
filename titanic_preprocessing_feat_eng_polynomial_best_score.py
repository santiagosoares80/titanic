# -*- coding: utf-8 -*-
"""
Created on Fri May 29 12:16:06 2020

@author: santiago
"""
import pandas as pd
import numpy as np
import re
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def substitute_title(title):
    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir', 'Mr']:
        return 'Mr'
    elif title in ['Mme', 'Lady', 'Mrs', 'Dona', 'the Countess']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms', 'Miss']:
        return 'Miss'
    elif title in ['Master']:
        return 'Master'
    elif title in ['Dr']:
        return 'Dr'
    else:   
        return 'No title'
    
# def select_group_age(age):
#     print(age)
#     if age < 16:
#         return 0
#     elif age < 30:
#         return 1
#     elif age < 45:
#         return 2
#     elif age < 60:
#         return 3
#     else:
#         return 4
        
train_filepath = 'train.csv'
test_filepath = 'test.csv'

X = pd.read_csv(train_filepath, index_col=['PassengerId'])
X_test = pd.read_csv(test_filepath, index_col=['PassengerId'])

# Remove rows with missing target
X.dropna(axis=0, subset=['Survived'], inplace=True)

## Separate target from predictors
y = X.Survived
X.drop(['Survived'], axis=1, inplace=True)

# Creating a column with titles
titles=[]
for title in X['Name']:
    titles.append(substitute_title(re.split(r"[,.] ",str(title))[1]))
X['Title'] = titles

titles_test=[]
for title in X_test['Name']:
    titles_test.append(substitute_title(re.split(r"[,.] ",str(title))[1]))
X_test['Title'] = titles_test

# Fill missing ages based on mean age by title
mean_age_by_title = {'Dr': 44, 'Master': 6, 'Mrs': 37, 'Ms': 37, 'Miss': 22, 'Mr': 32}

ages=[]
for age,title in zip(X['Age'],X['Title']):
    if np.isnan(age):
        ages.append(mean_age_by_title[title])
    else:
        ages.append(age)
X['Ages']=ages
    
ages_test=[]
for age,title in zip(X_test['Age'],X_test['Title']):
    if np.isnan(age):
        ages_test.append(mean_age_by_title[title])
    else:
        ages_test.append(age)
X_test['Ages']=ages_test

X['Relatives'] = X.SibSp + X.Parch
X_test['Relatives'] = X_test.SibSp + X_test.Parch

# Drop features that are irrelevant. Decided to drop Cabin, too many missing values
X.drop(['Ticket', 'Name', 'Cabin', 'Age', 'Title'], axis=1, inplace=True)
X_test.drop(['Ticket', 'Name', 'Cabin', 'Age', 'Title'], axis=1, inplace=True)

# Preprocessing data
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64','float64']]

categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']

numerical_transformer = Pipeline(steps=[
    ('imputer',  SimpleImputer(strategy='mean')),
    ('normalizer', StandardScaler()),
    ('poly', PolynomialFeatures())
    ])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
        ])

# Define model
model = XGBClassifier(learning_rate = 0.04, n_estimators=10)

# Define Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
    ])

# Preprocessing of training data, fit model 
model_pipeline.fit(X, y)

# Preprocessing of test data, fit model
preds_test = model_pipeline.predict(X_test) 

# Save test predictions to file
output = pd.DataFrame({'PassengerId': X_test.index,
                       'Survived': preds_test})
output.to_csv('submission.csv', index=False)