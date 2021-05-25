import pandas as pd
from sklearn import linear_model
import sklearn
import numpy as np

data = pd.read_csv("day.csv")

data = data[
    ["season", "yr", "holiday", "weekday", "workingday", "temp", "windspeed", "cnt"]
]

pred = "cnt"

x = np.array(data.drop(["cnt"], 1))
y = np.array(data[pred])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    x, y, test_size=0.1
)

model = linear_model.LinearRegression()

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)

prediction = model.predict(x_test)

for x in range(len(prediction)):
    print("Pred: ", prediction[x], "Actual: ", y_test[x])
