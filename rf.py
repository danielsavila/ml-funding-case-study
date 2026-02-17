import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from cleaned_data import cleaning_data
from creating_data import donation_data
from visualizations import visualizations
import joblib

X_train, X_test, y_train, y_test = cleaning_data(donation_data())

# random forest
rfmodel = RandomForestRegressor(n_jobs = -1, random_state = 13, criterion = "squared_error")
parameters = {"n_estimators": np.geomspace(25, 500, 15, dtype = int), # trees in forest
              "max_depth": np.linspace(4, 50, 15, dtype = int),
              }
# first finding the best parameter combination
output = GridSearchCV(rfmodel, 
                      parameters, 
                      scoring = "neg_mean_squared_error", 
                      cv = 5, 
                      n_jobs = -1).fit(X_train, y_train)
rfmodel_trained = RandomForestRegressor(n_jobs = -1, 
                                        random_state = 13,
                                        criterion = "squared_error",
                                        max_depth = output.best_params_["max_depth"],
                                        n_estimators = output.best_params_["n_estimators"]).fit(
                                            X_train,
                                             y_train)
y_train_pred = rfmodel_trained.predict(X_train)
print(f"random forest train mean error: ${round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2)}")
visualizations(y_train, y_train_pred, 'Predicted vs Actual Values, Train Set Random Forest')


y_test_pred = rfmodel_trained.predict(X_test)
print(f"random forest test mean error: ${round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2)}")
visualizations(y_test, y_test_pred, 'Predicted vs Actual Values, Test Set Random Forest')

# pickling random forest
joblib.dump(X_train.columns.tolist(), "feature_columns.pkl")
joblib.dump(rfmodel_trained, "rfmodel.pkl")