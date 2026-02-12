import pandas as pd
import numpy as np
from creating_data import donation_data
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML

np.random.seed(10101)
# some data formatting
df = donation_data()
test_set = donation_data(a = True)

df["month"] = df["donation_date"].dt.month
df["day"] = df["donation_date"].dt.day

# note that default is the following...
# payment_method : 0 == "card"
# city : 0 == "Chicago"
# year : 0 == "2020"
# month = : 0 == "1" i.e. January
# day = : 0 == "1"
df = pd.get_dummies(df, columns = ["payment_method"], drop_first = True, dtype = int)
df = pd.get_dummies(df, columns = ["city"], drop_first = True, dtype = int)
df = pd.get_dummies(df, columns = ["year"], drop_first = True, dtype = int)
df = pd.get_dummies(df, columns = ["month"], drop_first = True, dtype = int)
df = pd.get_dummies(df, columns = ["day"], drop_first = True, dtype = int)


y = df["donation_amount"]
X = df.drop(["donor_id", "donation_amount", "donation_id", "donation_date"], axis = 1)
X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .7, random_state = 10101, shuffle = True)

model = sm.OLS(y_train, X_train).fit()
print(model.summary())



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

