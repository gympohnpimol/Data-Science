import pandas as pd

penguin = pd.read_csv("/Users/gym/Desktop/Data-Science/WebApp-for-DataSci/penguin-app/PENGUINS.csv")
df = penguin.copy()
target = 'species'
encode = ['sex', 'island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

x = df.drop('species', axis=1)
y = df['species']

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(x,y)

import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))