#coding:utf-8
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

train_url = 'http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv'
test_url = 'http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv'

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# Impute the Embarked variable
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train['Embarked'].fillna('S')

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

#Print the Sex and Embarked columns
# print train["Sex"]
# print train["Embarked"]

print train
# Create the target and features numpy arrays: target, features_one
target = train['Survived'].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
print(my_tree_one.feature_importances_)
print(my_tree_one.score(features_one, target))


# Impute the missing value with the median
test.Fare[152] = test.Fare.median()

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age"]].values

# Make your prediction using the test set and print them.
my_prediction = my_tree_one.predict(test_features)
print(my_prediction)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
print(my_solution)

# Check that your data frame has 418 entries
print(my_solution.shape)

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("my_solution_one.csv", index_label = ["PassengerId"])