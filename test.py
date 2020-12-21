import pandas as pd
import numpy as np
import sklearn
from sklearn import model_selection

import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import linear_model
import pickle

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] #attributes
print(data)

predict = "G3"

x = np.array(data.drop([predict], 1)) #remove G3 - use other data to predict G3
y = np.array(data[predict]) #actual G3 values

learn = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
x_train, x_test, y_train, y_test = learn #must get only a section of data so data left to test outcomes (10 percent of data as test samples), randomly selected 0.1

best = 0
for _ in range(30):
    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train) # fit data using x_train and y_train
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print(linear.coef_)
print(linear.intercept_)

predictions = linear.predict(x_test) #take array

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x]) #compare....

p = 'absences'

style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()