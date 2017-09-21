# titanic-machine-learning

### Introduction

Kaggle's Titanic Machine Learning problem. The (somewhat morbid) task is to predict if a passenger will survive given a number attributes that passenger. 

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

reads in ``train.csv`` and uses to train models. Applies trained models to passenger data in ``test.csv`` and writes out prediction on the survival of each passenger in ``submission.csv`` for submission to Kaggle's contest.


### Resources

https://www.kaggle.com/c/titanic
http://scikit-learn.org
