#coding:utf-8
import pandas as pd

train_url = 'http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv'
test_url = 'http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv'

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

# .describe() summarizes the columns/features of the dataframe
# print train.describe()
# print train.shape

# print train['Survived'].value_counts()  # absolute numbers
# print train['Survived'].value_counts(normalize=True)  # percentages:549 individuals died (62%) and 342 survived (38%)

# print train['Survived'][train['Sex'] == 'male'].value_counts()
# print train['Survived'][train['Sex'] == 'female'].value_counts()

# train['Child'] = 0  # create a new column
#
# train["Child"][train['Age']< 18] = 1
# print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))
#
# # Print normalized Survival Rates for passengers 18 or older
# print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))

test_one = test.copy()
test_one['Survived'] = 0
test_one['Survived'][test_one['Sex']=='female'] = 1
print(test_one['Survived'])