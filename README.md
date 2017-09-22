# titanic-machine-learning

### Introduction

This repository tackles Kaggle's Titanic Machine Learning problem. The (somewhat morbid) task is to predict if a passenger will survive given a number attributes for that passenger. 

``titanic-ml.py`` reads in ``train.csv`` which contains a list of passengers their attributes (e.g. Age, Gender, Cabin Type, etc) and if they survived. This data is then used to train a number classification algorithms and build models that are used to predict survival of each passenger listed in ``test.csv``.

### Algorithms tested

A number of the classification algorithms found in Python's ``sci-kit-learn`` library were used, including:

Logistic Regression
Support Vector Machines
k-Nearest Neighbor
Gaussian Naive Bayes
Perceptron
Stochastic Gradient Descent
Random Forest

The accuracy of each algorithm is reported in order to choose the optimal algorithm for this problem. 

### Train the Model

```
python titanic-ml.py
```

Example output:
```
Model: Logistic Regression
Score: 80.36
-------------
Model: Support Vector Machines
Score: 83.84
-------------
Model: Random Forest
Score: 86.76
-------------
Best Model: Random Forest
Score: 86.76
```

code reads in ``train.csv`` and uses to train multiple classification models. These trained models are then applied to passenger data in ``test.csv`` and writes out prediction on the survival of each passenger for the best model in ``submission.csv`` for submission to Kaggle's contest.


### Resources

https://www.kaggle.com/c/titanic
http://scikit-learn.org
