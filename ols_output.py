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
    df = cleaning_data(donation_data())

    y = df["donation_amount"]
    X = df.drop(["donor_id", "donation_amount", "donation_id", "donation_date"], axis = 1)
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .7, random_state = 10101, shuffle = True)
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
    
    return  model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test = regression_testing()