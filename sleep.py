import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection

import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import linear_model
import pickle

data = pd.read_csv('sleep.csv', sep=',')
data = data[["day", "start", "length", "cycles", "deep"]]

predict = "deep"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

learn = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
x_train, x_test, y_train, y_test = learn

best = 0
iter = 30

for i in range(iter):
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    if acc > best:
        best = acc
        with open("sleep.pickle", "wb") as f:
            pickle.dump(linear, f)

print('linear score: {}'.format(best))

# pickle_in = open("studentmodel.pickle", "rb")
# linear = pickle.load(pickle_in)

print('\n')
print(linear.coef_)
print(linear.intercept_)
print('\n')

predictions = linear.predict(x_test)
counter = 0
total_error = 0
for i in range(len(predictions)):
    counter += 1
    predicted = round(predictions[i], 2)
    actual = y_test[i]
    error = round(abs((actual - predicted)/actual), 2)
    total_error += error

    print(predicted, x_test[i], actual, error)
print('error: {}'.format(round(total_error/counter, 2)))

p = 'start'

style.use("ggplot")
plt.scatter(data[p], data["deep"])
plt.xlabel(p)
plt.ylabel("deep sleep")
plt.show()