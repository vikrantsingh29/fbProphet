
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
from fbprophet import Prophet

# Load the dataset using pandas
data = pd.read_csv("/Users/vikrant.singh/Downloads/TSLA (1).csv")

# Select only the important features i.e. the date and price
data = data[["Date","Close"]] # select Date and Price

# Rename the features: These names are NEEDED for the model fitting
data = data.rename(columns = {"Date":"ds","Close":"y"})
#renaming the columns of the dataset


m = Prophet(daily_seasonality=True)# the Prophet class (model)
m.fit(data) # fit the model using all data

future = m.make_future_dataframe(periods=365) #we need to specify the number of days in future
prediction = m.predict(future)
m.plot(prediction)
plt.title("Prediction of the TESLA Stock Price using the Prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show()
