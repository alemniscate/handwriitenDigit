from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

def fit_predict_eval(model, features_train, features_test, target_train, target_test, score_list, print_flag):
    # here you fit the model
    model.fit(features_train, target_train)
    # make a prediction
    model.predict(features_test)
    # calculate accuracy and save it to score
    score = model.score(features_test, target_test)
    if print_flag:
        print(f'Model: {model}\nAccuracy: {score}\n')

    score_list.append((model, score))

def model_test(X_train, X_test, Y_train, Y_test, score_list, print_flag):
    fit_predict_eval(KNeighborsClassifier(), X_train, X_test, Y_train, Y_test, score_list, print_flag)
    fit_predict_eval(DecisionTreeClassifier(random_state=40), X_train, X_test, Y_train, Y_test, score_list, print_flag)
    fit_predict_eval(LogisticRegression(random_state=40, solver="liblinear"), X_train, X_test, Y_train, Y_test, score_list, print_flag)
    fit_predict_eval(RandomForestClassifier(random_state=40), X_train, X_test, Y_train, Y_test, score_list, print_flag)

    max_score = -1
    max_model = None

    for model, score in score_list:
        if score > max_score:
            max_score = score
            max_model = model

    return (max_model, max_score)
    
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

uc = list(np.unique(y_train))
x_train = x_train.reshape([60000, 784])

X_train, X_test, Y_train, Y_test = train_test_split(x_train[:6000], y_train[:6000], test_size=0.3, random_state=40)

transformer = Normalizer()
X_train_norm = transformer.transform(X_train)
X_test_norm = transformer.transform(X_test)

param_grid = {'n_neighbors': [3, 4], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'brute']}
knc = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, scoring='accuracy', n_jobs=-1)

param_grid = {'n_estimators': [300, 500], 'max_features': ['auto', 'log2'], 'class_weight': ['balanced', 'balanced_subsample']}
rfc = GridSearchCV(estimator=RandomForestClassifier(random_state=40), param_grid=param_grid, scoring='accuracy', n_jobs=-1)

score_list = []
knc.fit(X_train, Y_train)
fit_predict_eval(knc.best_estimator_, X_train, X_test, Y_train, Y_test, score_list, False)
rfc.fit(X_train, Y_train)
fit_predict_eval(rfc.best_estimator_, X_train, X_test, Y_train, Y_test, score_list, False)
knc.fit(X_train_norm, Y_train)
fit_predict_eval(knc.best_estimator_, X_train_norm, X_test_norm, Y_train, Y_test, score_list, False)
rfc.fit(X_train_norm, Y_train)
fit_predict_eval(rfc.best_estimator_, X_train_norm, X_test_norm, Y_train, Y_test, score_list, False)

print("K-nearest neighbours algorithm")
print(f"best estimator: {knc.best_estimator_}")
_, accuracy1 = score_list[0]
_, accuracy2 = score_list[2]
accuracy = max(accuracy1, accuracy2)
print(f"accuracy: {np.round(accuracy, 3)}")
print()

print("Random forest algorithm")
print(f"best estimator: {rfc.best_estimator_}") 
_, accuracy1 = score_list[1]
_, accuracy2 = score_list[3]
accuracy = max(accuracy1, accuracy2)
print(f"accuracy: {np.round(accuracy, 3)}")

pass