from creating_data import donation_data
import pandas as pd
from sklearn.model_selection import train_test_split

def cleaning_data(uncleaned_dataframe, ols = False, production = False):
    uncleaned_dataframe["donation_date"] = pd.to_datetime(uncleaned_dataframe["donation_date"])
    uncleaned_dataframe["month"] = uncleaned_dataframe["donation_date"].dt.month
    uncleaned_dataframe["day"] = uncleaned_dataframe["donation_date"].dt.day
    categorical_cols = ["payment_method", "city", "month"]

    #this is for ols and svr
    if ols == True:
        # note that default is the following...
        # payment_method : 0 == "card"
        # city : 0 == "Chicago"
        # year : 0 == "2020"
        # month = : 0 == "1" i.e. January
        # day = : 0 == "1"
        cleaned_df = pd.get_dummies(uncleaned_dataframe, columns=categorical_cols, drop_first=True, dtype=int)
        y = cleaned_df["donation_amount"]
        X = cleaned_df.drop(["donor_id", "donation_amount", "donation_id", "donation_date"], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .7, random_state = 10101, shuffle = True)
        
        return X_train, X_test, y_train, y_test
    
    #this is relevant in fastapi data pipeline (app.py)
    elif production == True:
        cleaned_df = pd.get_dummies(uncleaned_dataframe, columns=categorical_cols, dtype=int)
        X = cleaned_df.drop(["donor_id", "donation_id", "donation_date"], axis = 1)

        return X

    else:
        cleaned_df = pd.get_dummies(uncleaned_dataframe, columns=categorical_cols, dtype=int) #drop_first is missing to enable one-hot encoding
        y = cleaned_df["donation_amount"]
        X = cleaned_df.drop(["donor_id", "donation_amount", "donation_id", "donation_date"], axis = 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .7, random_state = 10101, shuffle = True)
    
        return X_train, X_test, y_train, y_test