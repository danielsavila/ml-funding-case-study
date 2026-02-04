import pandas as pd
import numpy as np

np.random.seed(10101)
donation_id = np.linspace(1, 1000, 1000, dtype = int)
donation_dates = pd.date_range(start="2020-03-15", end = "2026-03-03", freq = "D")
donation_dates = pd.Series(donation_dates).sample(1000, replace = True).sort_values().reset_index(drop = True)
donor_id = np.random.randint(1, 75, size = 1000).astype(int)
payment_method = np.random.choice(["card", "check", "wire", "cash"], size = 1000, p = [.5, .3, .15, .05])
campaign_indicator = np.random.choice([0, 1], 1000, p = [.4, .6])
region = np.random.choice(["Chicago", "New York", "San Francisco", "Milwaukee", "Detroit"], size = 1000, p = [.45, .15, .1, .2, .1])

df = pd.DataFrame({"donation_id": donation_id,
                   "donor_id": donor_id,
                   "donation_date": pd.to_datetime(donation_dates),
                   "payment_method": payment_method,
                   "city": region,
                   "campaign_indicator": campaign_indicator})

donation_choicelist = [np.random.choice(np.arange(25, 2500)), 
                          np.random.choice(np.arange(2000, 7500)),
                          np.random.choice(np.arange(3000, 5000)),
                          np.random.choice(np.arange(500))]

df["donation_amount"] = np.select([df["payment_method"] == "card",
                                   df["payment_method"] == "check",
                                   df["payment_method"] == "wire",
                                   df["payment_method"] == "cash"], donation_choicelist)

