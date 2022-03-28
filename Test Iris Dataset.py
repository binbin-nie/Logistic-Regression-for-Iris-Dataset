# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 19:09:42 2022

@author: Binbin Nie
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# import MyPackage
from LogisticRegression.LogisticRegression import LogisticRegression

def iris_data_preprocess():
    # load iris dataset
    iris_data = load_iris()
    iris_target = iris_data.target
    iris_features = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)

    # copy type 0 and type 1 since we will do two-type classification
    iris_all = iris_features.copy()    # shallow copy
    iris_all['target'] = iris_data.target   # add a new feature named target
    iris_features_part = iris_features.iloc[:100]
    iris_features_part['one'] = 1
    iris_target_part = iris_target[:100]
    
    # train test set split
    X_train, X_test, y_train, y_test = train_test_split(iris_features_part, iris_target_part, test_size=0.3)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


X_train, X_test, y_train, y_test = iris_data_preprocess()
iris_classification = LogisticRegression(X_train, X_test, y_train, y_test)
N = 1000000
iris_classification.propagate(N)
print('Classification accuracy is {}%'.format(iris_classification.predict()*100))
iris_classification.confuseMatrix()