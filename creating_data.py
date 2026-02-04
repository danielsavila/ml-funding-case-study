import pandas as pd
import numpy as np

def donation_data(a = False):
    donation_id = np.linspace(1, 1000, 1000, dtype = int)

    donation_dates = pd.date_range(start="2020-03-15", end = "2026-03-03", freq = "D")
    donation_dates = pd.Series(donation_dates).sample(
        1000, replace = True).sort_values(
        ).reset_index(drop = True)
    
    donor_id = np.random.randint(1, 75, size = 1000).astype(int)

    payment_method = np.random.choice(["card", "check", "digital", "cash"], size = 1000, p = [.3, .3, .35, .05])

    cities = ["Chicago", "New York", "San Francisco", "Milwaukee", "Detroit"]
    region = np.random.choice(cities, size = 1000, p = [.45, .15, .1, .2, .1])

    df = pd.DataFrame({"donation_id": donation_id,
                    "donor_id": donor_id,
                    "donation_date": pd.to_datetime(donation_dates),
                    "payment_method": payment_method,
                    "city": region})

    donation_amount = []
    for i in range(len(df)):
        value = df.loc[i, "payment_method"]
        if value == "card":
            amount = np.random.choice(np.arange(25, 2500))
            donation_amount.append(amount)
        elif value == "check":
            amount =  np.random.choice(np.arange(2000, 7500))
            donation_amount.append(amount)
        elif value == "digital":
            amount = np.random.choice(np.arange(3000, 5000))
            donation_amount.append(amount)
        else:
            amount = np.random.choice(np.arange(500))
            donation_amount.append(amount)

    df["donation_amount"] = donation_amount

    # making adjustments for chicago and san francisco
    for i in range(len(df)):
        city = df.loc[i, "city"]
        if city == "San Francisco":
            df.loc[i, "donation_amount"] = (df.loc[i, "donation_amount"] * .65).astype(int)
        if city == "Detroit":
            df.loc[i, "donation_amount"] = (df.loc[i, "donation_amount"] * 1.15).astype(int)

    lf_donations = df.groupby(["donor_id"], as_index = False).count()[["donor_id", "donation_id"]].rename(columns={"donation_id": "count_lifetime_donations"})
    df = df.merge(lf_donations, on = "donor_id", how = "left")

    df["year"] = df["donation_date"].dt.year
    donations_by_year = df.groupby(["donor_id", "year"], as_index = False).count()[["donor_id", "year", "donation_id"]].rename(columns = {"donation_id": "count_donations_by_year"})
    df = df.merge(donations_by_year, on = ["donor_id", "year"], how = 'left')

    # creating campaing indicator around specific dates
          # holiday/tax year end donation campaign, every year
    df["campaign_indicator"] = 0
    for i in df["year"].unique():
        df.loc[(df["donation_date"] >= f"{i}-11-02") & (df["donation_date"] <= f"{i + 1}-1-15"), "campaign_indicator"] = np.random.binomial(1, .55)


    # turn this on to create a test set with slight variation in amounts from chicago and san francisco
    if a:
        for i in range(len(df)):
            city = df.loc[i, "city"]
            if city == "Chicago":
                df.loc[i, "donation_amount"] = (df.loc[i, "donation_amount"] * np.random.uniform(1.05, 1.35)).astype(int)
            elif city == "San Francisco":
                df.loc[i, "donation_amount"] = (df.loc[i, "donation_amount"] * np.random.uniform(1.1, 1.25)).astype(int)
    
    return df

if __name__ == "__main__":
    df = donation_data()