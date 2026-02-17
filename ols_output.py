import pandas as pd
import numpy as np
from cleaned_data import cleaning_data
from creating_data import donation_data
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML

np.random.seed(10101)

def regression_testing():
    X_train, X_test, y_train, y_test = cleaning_data(donation_data(), ols = True)
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)
    model = sm.OLS(y_train, X_train).fit()


    # capturing statistically significant outputs in html
    results = pd.DataFrame({
        "Coefficient": model.params,
        "Std. Error": model.bse,
        "t-stat": model.tvalues,
        "p-value": model.pvalues
    })

    sig_results = results[results["p-value"] < 0.05].round(3)
    html = sig_results.to_html(border=1, justify="center")

    with open("significant_output.html", "w") as f:
        f.write(html)

    X_train = X_train.drop("const", axis = 1)
    
    return  model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = regression_testing()
