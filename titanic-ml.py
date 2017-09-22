
# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

class data():
    def __init__(self,pNf_train,pNf_test):

        self.train_df = pd.read_csv(str(pNf_train))
        self.test_df = pd.read_csv(str(pNf_test))
        self.combine = [self.train_df, self.test_df]

        # drop Ticket and Cabin data - not relevant
        self.train_df = self.train_df.drop(['Ticket', 'Cabin'], axis=1)
        self.test_df = self.test_df.drop(['Ticket', 'Cabin'], axis=1)
        self.combine = [self.train_df, self.test_df]

        # cleanup titles
        for dataset in self.combine:
            dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
            #print dataset['Title']
        c = 0
        for dataset in self.combine:
            #print c
            c +=1
            #print 
            #print dataset['Title']
            # for replace titles that occure infrequently with the 'Rare' string
            dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

            dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
            dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
            #print dataset['Title']

        self.train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

        # There are 5 titles now, map each title to a number
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        for dataset in self.combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)

        #print train_df.head()

        self.train_df = self.train_df.drop(['Name', 'PassengerId'], axis=1)
        self.test_df = self.test_df.drop(['Name'], axis=1)
        self.combine = [self.train_df, self.test_df]
        self.train_df.shape, self.test_df.shape

        #print train_df.head()

        # map sec to a number
        for dataset in self.combine:
            dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

        self.train_df.head()

        # guess age for missing age

        guess_ages = np.zeros((2,3))
        guess_ages

        for dataset in self.combine:
            for i in range(0, 2):
                for j in range(0, 3):
                    guess_df = dataset[(dataset['Sex'] == i) & \
                                        (dataset['Pclass'] == j+1)]['Age'].dropna()

                    # age_mean = guess_df.mean()
                    # age_std = guess_df.std()
                    # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

                    age_guess = guess_df.median()

                    # Convert random age float to nearest .5 age
                    guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
                    
            for i in range(0, 2):
                for j in range(0, 3):
                    dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                            'Age'] = guess_ages[i,j]

            dataset['Age'] = dataset['Age'].astype(int)

        self.train_df.head()


        ###############

        self.train_df['AgeBand'] = pd.cut(self.train_df['Age'], 5) # break up age into 5 bins
        self.train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


        ###############

        for dataset in self.combine:    
            dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
            dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
            dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
            dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
            dataset.loc[ dataset['Age'] > 64, 'Age']
        self.train_df.head()

        ################

        self.train_df = self.train_df.drop(['AgeBand'], axis=1)
        self.combine = [self.train_df, self.test_df]
        self.train_df.head()

        # set FamilySize
        for dataset in self.combine:
            dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 #why +1? Including current passenger in Family.

        self.train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

        #print train_df[['FamilySize', 'Survived']].groupby(['Survived'], as_index=False).mean()

        for dataset in self.combine:
            dataset['IsAlone'] = 0
            dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

        self.train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

        self.train_df = self.train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
        self.test_df = self.test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
        self.combine = [self.train_df, self.test_df]

        self.train_df.head()

        ################

        for dataset in self.combine:
            dataset['Age*Class'] = dataset.Age * dataset.Pclass

        self.train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


        freq_port = self.train_df.Embarked.dropna().mode()[0]
        freq_port

        for dataset in self.combine:
            dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
            
        #print train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


        #################

        for dataset in self.combine:
            dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

        self.train_df.head()


        ##################

        self.test_df['Fare'].fillna(self.test_df['Fare'].dropna().median(), inplace=True)
        self.test_df.head()

        ##################

        self.train_df['FareBand'] = pd.qcut(self.train_df['Fare'], 4)
        self.train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

        ###################
        # bin fare
        for dataset in self.combine:
            dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
            dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
            dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
            dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
            dataset['Fare'] = dataset['Fare'].astype(int)

        self.train_df = self.train_df.drop(['FareBand'], axis=1)
        self.combine = [self.train_df, self.test_df]
            
        self.train_df.head(10)

        ###################

        self.X_train = self.train_df.drop("Survived", axis=1)
        self.Y_train = self.train_df["Survived"]
        self.X_test  = self.test_df.drop("PassengerId", axis=1).copy()
        self.X_train.shape, self.Y_train.shape, self.X_test.shape


        return

class predict_survival():
    def __init__(self,data,model='Logistic Regression'):

        self.data = data
        self.model = model

        if model == 'Logistic Regression':
            # Logistic Regression
            logreg = LogisticRegression()
            logreg.fit(data.X_train, data.Y_train)
            self.Y_pred = logreg.predict(data.X_test)
            self.score = round(logreg.score(data.X_train, data.Y_train) * 100, 2)
            
        elif model == 'Support Vector Machines':
            # Support Vector Machines
            svc = SVC()
            svc.fit(data.X_train, data.Y_train)
            self.Y_pred = svc.predict(data.X_test)
            self.score = round(svc.score(data.X_train, data.Y_train) * 100, 2)
            
        elif model == 'Random Forest':
            # Random Forest
            random_forest = RandomForestClassifier(n_estimators=100)
            random_forest.fit(data.X_train, data.Y_train)
            self.Y_pred = random_forest.predict(data.X_test)
            self.score = round(random_forest.score(data.X_train, data.Y_train) * 100, 2)
            
        else:
            print str(model)+' not recognized'
            self.score = None
            self.Y_pred = None

        print 'Model: '+str(model)
        print 'Score: '+str(self.score)
        print '-------------'
        return

    def write_submission(self, pNf='submission.csv'):
        
        submission = pd.DataFrame({
            "PassengerId": self.data.test_df["PassengerId"],
            "Survived": self.Y_pred
            })
        submission.to_csv(pNf, index=False)

passengers = data('train.csv','test.csv') # reads in and prepares data for modeling

model_list = []
model_obj = predict_survival(passengers,'Logistic Regression')
model_list.append(model_obj)
model_obj = predict_survival(passengers,'Support Vector Machines')
model_list.append(model_obj)
model_obj = predict_survival(passengers,'Random Forest')
model_list.append(model_obj)

# Search for best model
best_score = 0
best_model = None
for model in model_list:
    if model.score > best_score:
        best_score = model.score
        best_model = model

print 'Best Model: '+str(best_model.model)
print 'Score: '+str(best_score)
best_model.write_submission()