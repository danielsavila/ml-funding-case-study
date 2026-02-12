import pandas as pd
import numpy as np 
import seaborn as sns
from creating_data import donation_data
import matplotlib.pyplot as plt

np.random.seed(1)
df = donation_data()
test = donation_data(a = True)
df.columns

sns.countplot(df, x = "payment_method")
plt.show()

sns.barplot(df, x = "city", 
            y = "donation_amount", 
            hue = "campaign_indicator")
plt.show()

sns.scatterplot(df, x = "donation_id",
                y = "donation_amount", 
                hue = "payment_method",
                size = "major_donor_flag")
plt.show()

sns.scatterplot(df, x = "year", 
                y = "donation_amount", 
                hue = "campaign_indicator", 
                size = "major_donor_flag")
plt.show()