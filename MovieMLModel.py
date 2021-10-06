# Prediction algorithm using ML

import pandas
from sklearn.tree import DecisionTreeClassifier

moviedata=pandas.read_csv('moviechoice.csv')
#print(moviedata)

#training data
features=moviedata.drop(columns=['genre'])
labels=moviedata['genre']

#build a model
model=DecisionTreeClassifier()
model.fit(features,labels)

#test data and predict
result=model.predict([[1,55,1],[2,52,1]])
print(result)

'''
output -> ['Horror' 'Drama']
that means a person who is male, age 55 and lives in city will like Horror
a person who is female of age 52 and urban will prefer watching Drama genre

1 for male & 2 for female
1 for urban & 2 for rural
'''
