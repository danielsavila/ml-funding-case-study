from ols_output import regression_testing
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from visualizations import visualizations

np.random.seed(13)
model, X_train, X_test, y_train, y_test = regression_testing()

# evaluating OLS on the training set
X_train = sm.add_constant(X_train) # need to add this back in for OLS
y_train_pred = model.predict(X_train)
print(f"ols training average mean error: ${round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 2)}")
visualizations(y_train, y_train_pred, 'Predicted vs Actual Values, Train Set OLS')

# evaluating OLS on test set
y_test_pred = model.predict(X_test)
print(f"ols test mean error: ${round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 2)}")
visualizations(y_test, y_test_pred, 'Predicted vs Actual Values, Test Set OLS')