from creating_data import donation_data
from cleaned_data import cleaning_data
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from visualizations import visualizations

np.random.seed(13)
X_train, X_test, y_train, y_test = cleaning_data(donation_data())

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