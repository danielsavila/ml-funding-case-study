from creating_data import donation_data
import numpy as np
import pandas as pd

def cleaning_data(uncleaned_dataframe):
    
    uncleaned_dataframe["month"] = uncleaned_dataframe["donation_date"].dt.month
    uncleaned_dataframe["day"] = uncleaned_dataframe["donation_date"].dt.day

    # note that default is the following...
    # payment_method : 0 == "card"
    # city : 0 == "Chicago"
    # year : 0 == "2020"
    # month = : 0 == "1" i.e. January
    # day = : 0 == "1"
    categorical_cols = ["payment_method", "city", "year", "month", "day"]
    cleaned_df = pd.get_dummies(uncleaned_dataframe, columns=categorical_cols, drop_first=True, dtype=int)
    
    return cleaned_df