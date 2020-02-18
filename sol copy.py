# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
Y = dataset.iloc[:, 2].values
X = dataset.iloc[:, 1:]
X = X.drop("Customer Lifetime Value", axis = 1)
X = X.drop("Effective To Date", axis = 1)
X = X.values

# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
for i in range(0,21):
    if(type(X[1, i]) == str):
        labelencoder = LabelEncoder()
        X[:, i] = labelencoder.fit_transform(X[:, i])

onehotencoder = OneHotEncoder(categorical_features = [0,1,2,3,4,5,7,8,14,15,16,17,19,20])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 100)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# +
# Random Forest Regressor
from sklearn.tree import DecisionTreeRegressor
#regressor = RandomForestRegressor(n_estimators = 200, random_state = 0, max_depth = 17, min_samples_split = 2)
regressor = DecisionTreeRegressor()

regressor.fit(X_train, Y_train)

Y_pred_1 = regressor.predict(X_test)

Y_pred_2 = regressor.predict(X_train)

from sklearn.metrics import r2_score
r2_test = r2_score(Y_test, Y_pred_1)
r2_train = r2_score(Y_train, Y_pred_2)
print(r2_test, r2_train)
# -


from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, Y_train)
Y_pred_1 = regressor.predict(X_test)
Y_pred_2 = regressor.predict(X_train)
r2_test = r2_score(Y_test, Y_pred_1)
r2_train = r2_score(Y_train, Y_pred_2)
print(r2_test, r2_train)

from sklearn.metrics import mean_squared_error


def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    prediction_test = estimator.predict(x_tst)
    return {
        'r2' : r2_score(prediction_test, y_tst), 
        'RMSE' : np.sqrt(mean_squared_error(prediction_test, y_tst))
    }



