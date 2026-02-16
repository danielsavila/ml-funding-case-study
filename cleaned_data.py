from creating_data import donation_data
import numpy as np
import pandas as pd

def cleaning_data():
    df = donation_data()
    
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

    return df

if __name__ == "__main__":
    df = cleaning_data()