from ols_output import regression_testing
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def visualizations(true, predicted, title):
        plt.figure(figsize=(8, 6))
        plt.scatter(true, predicted, alpha=0.6, color='blue', edgecolor='k')
        plt.plot([true.min(), true.max()],
                [true.min(), true.max()],
                'r--', linewidth=2)  # reference line y=x
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{title}')
        plt.grid(True)
        return plt.show()

np.random.seed(13)
model, X_train, X_test, y_train, y_test = regression_testing()

# evaluating OLS on the training set
ols = LinearRegression()
ols = ols.fit(X_train, y_train)
y_train_pred = ols.predict(X_train)
print(f"ols training average mean error: ${round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2)}")
visualizations(y_train, y_train_pred, 'Predicted vs Actual Values, Train Set OLS')

# evaluating OLS on test set
y_test_pred = ols.predict(X_test)
print(f"ols test mean error: ${round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2)}")
visualizations(y_test, y_test_pred, 'Predicted vs Actual Values, Test Set OLS')


# random forest
rfmodel = RandomForestRegressor(n_jobs = -1, random_state = 1, criterion = "squared_error")
parameters = {"n_estimators": np.geomspace(25, 500, 15, dtype = int), # trees in forest
              "max_depth": np.linspace(4, 50, 15, dtype = int),
              "max_features": np.linspace(1, 51, 10, dtype = int)
              }
# first finding the best parameter combination
output = GridSearchCV(rfmodel, 
                      parameters, 
                      scoring = "neg_mean_squared_error", 
                      cv = 5, 
                      n_jobs = -1).fit(X_train, y_train)
rfmodel_trained = RandomForestRegressor(n_jobs = -1, 
                                        random_state = 1,
                                        criterion = "squared_error",
                                        max_depth = output.best_params_["max_depth"],
                                        n_estimators = output.best_params_["n_estimators"],
                                        max_features = output.best_params_["max_features"]).fit(
                                            X_train,
                                             y_train)
y_train_pred = rfmodel_trained.predict(X_train)
print(f"random forest train mean error: ${round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2)}")
visualizations(y_train, y_train_pred, 'Predicted vs Actual Values, Train Set Random Forest')


y_test_pred = rfmodel_trained.predict(X_test)
print(f"random forest test mean error: ${round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2)}")
visualizations(y_test, y_test_pred, 'Predicted vs Actual Values, Test Set Random Forest')


# models that did not make improvements relative to OLS
# Support Vector Machines Regression
# SVM is sensitive to scaling in features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr = SVR()
parameters = {"kernel": ("linear", "poly", "rbf"),
              "gamma": ("scale", "auto"),
              "C": np.geomspace(.001, 500, 100, dtype = float)} 
output = GridSearchCV(svr, 
                      parameters, 
                      scoring = "neg_mean_squared_error",
                      n_jobs = -1,
                      cv = 5).fit(X_train_scaled, y_train)

print(output.best_params_)

svr = SVR(C = output.best_params_["C"],
        gamma = output.best_params_["gamma"], 
        kernel = output.best_params_["kernel"]).fit(X_train_scaled, y_train)
y_train_pred = svr.predict(X_train_scaled)
print(f"svr train mean error: ${round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2)}")
visualizations(y_train, y_train_pred, 'Predicted vs Actual Values, Train Set SVM')

y_test_pred = svr.predict(X_test_scaled)
print(f"svr test mean error: ${round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2)}")
visualizations(y_test, y_test_pred, 'Predicted vs Actual Values, Test Set SVM')

# pickling random forest
joblib.dump(rfmodel_trained, "rfmodel.pkl")