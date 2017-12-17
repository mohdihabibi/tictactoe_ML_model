import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


data = pd.read_csv('tictac_single.txt', sep=" ", header=None)

board_stat = data.iloc[:, :9]
next_move = data.iloc[:, 9]


X_train, X_test, Y_train, Y_test = train_test_split(board_stat,next_move, test_size=0.2, random_state=42)


forest = RandomForestClassifier(n_estimators = 50, random_state = 0)
# forest.fit(X_train, Y_train)
forest.fit(board_stat, next_move)

print("Accuracy on test set: {:.3f}".format(forest.score(X_test, Y_test)))

print(forest.predict([[1, -1, 0, 0, 0, 0, -1, 1, 0]]))

filename = 'finalized_random_forest_model.sav'
joblib.dump(forest, filename)

# print data.iloc[0,9]

