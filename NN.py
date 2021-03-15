# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 20:23:01 2021

@author: Joey
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import imblearn
from mlrose import NeuralNetwork
import mlrose 

LE = LabelEncoder()

survey_response = pd.read_csv("C:/Users/Joey/Downloads/archive(1)/responses.csv")

survey_response.describe()
survey_response = survey_response.drop(columns={"House - block of flats"})


null_count = survey_response.isnull().sum()

target = 'Village - town'
survey_response[target] = ['rural' if x == "village" else 'city' for x in survey_response[target] ]
survey_response = survey_response[survey_response[target].notnull()]



survey_response['Punctuality'] = LE.fit_transform(survey_response['Punctuality'].tolist())
survey_response['Lying'] = LE.fit_transform(survey_response['Lying'].tolist())
survey_response['Internet usage'] = LE.fit_transform(survey_response['Internet usage'].tolist())
survey_response['Left - right handed'] = LE.fit_transform(survey_response['Left - right handed'].tolist())
survey_response['Smoking'] = LE.fit_transform(survey_response['Smoking'].tolist())
survey_response['Alcohol'] = LE.fit_transform(survey_response['Alcohol'].tolist())
survey_response['Gender'] = LE.fit_transform(survey_response['Gender'].tolist())
survey_response['Education'] = LE.fit_transform(survey_response['Education'].tolist())
survey_response['Only child'] = LE.fit_transform(survey_response['Only child'].tolist())
survey_response['Village - town'] = LE.fit_transform(survey_response['Village - town'].tolist())


survey_response = survey_response.dropna()
ros = imblearn.over_sampling.RandomOverSampler(random_state = 2047)
ros.fit(survey_response.iloc[:,:-1], survey_response[target])
X, Y = ros.fit_resample(survey_response.iloc[:,:-1], survey_response[target])


temp = pd.DataFrame(pd.DataFrame(Y)[target].value_counts())
temp = temp.reset_index()
temp  = temp.rename(index={0:'city', 1:'rural'})
temp = temp.drop(columns = {'index'})


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state  = 2047)
x_train = x_train.reset_index(drop=True)
x_test = x_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


for i in [0.0001, 0.001, 0.01, 0.9]:

    nn_model1 = NeuralNetwork(hidden_nodes = [], activation = 'sigmoid',
                                     algorithm = 'random_hill_climb', max_iters = 1000,
                                     bias = True, is_classifier = True, learning_rate = i,
                                     early_stopping = True, max_attempts = 100,
    				 random_state = 8)
    
    nn_model1.fit(x_train, y_train)
    
    y_test_pred = nn_model1.predict(x_test)
    y_test_f1 = f1_score(y_test, y_test_pred)
    
    print(y_test_f1)
    print('Random Hill Climbing - Classification Report = \n {}'.format(classification_report(y_test, y_test_pred)))





for i in [0.0001, 0.001, 0.01, 0.9]:
    for j in [10,100, 250]:
        nn_model2 = NeuralNetwork(hidden_nodes = [], activation = 'sigmoid',
                                         algorithm = 'simulated_annealing', max_iters = 1000,
                                         bias = True, schedule = mlrose.ExpDecay(init_temp = j, exp_const = 0.9), is_classifier = True, learning_rate = i, max_attempts = 100,
        				 random_state = 8)
        
        nn_model2.fit(x_train, y_train)
        
        y_test_pred = nn_model2.predict(x_test)
        y_test_accuracy = f1_score(y_test, y_test_pred)
        
        print(y_test_accuracy)
        print('Simulated Annealing - Classification Report = \n {}'.format(classification_report(y_test, y_test_pred)))



for i in [0.0001, 0.001, 0.01, 0.9]:
    for j in [100,250, 500]:
        nn_model3 = NeuralNetwork(hidden_nodes = [], activation = 'sigmoid',
                                         algorithm = 'genetic_alg', max_iters = 1000,  mutation_prob = 0.3,
                                         bias = True, is_classifier = True, learning_rate = i, max_attempts = 100,pop_size = j,
        				 random_state = 8)
        
        nn_model3.fit(x_train, y_train)
        
        y_test_pred = nn_model3.predict(x_test)
        y_test_accuracy = f1_score(y_test, y_test_pred)
        
        print('Genetic Algorithm - Classification Report = \n {}'.format(classification_report(y_test, y_test_pred)))



