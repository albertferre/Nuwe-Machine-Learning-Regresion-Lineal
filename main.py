import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

# importing data
train = pd.read_csv('data/j0001_train.csv')
test = pd.read_csv('data/j0001_X_test.csv')

# Splitting data
X = train.drop(columns=['target'])
y = train.target
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42, test_size=0.2)

# Fitting the model
model = LinearRegression()
model.fit(X_train, y_train)

# Printing results
print('Coefficients: \n', model.coef_)

y_pred = model.predict(X_valid)
mse = mean_squared_error(y_pred, y_valid)
print('MSE: \n', mse)

# Plots
plt.scatter(y_pred, y_valid,  color='black')
plt.plot(y_pred,y_valid, color='red', linewidth=3)
plt.savefig('plots/regression.png')

# Saving results
pd.DataFrame(model.predict(test),
             columns=['results']).to_csv('results.csv')
