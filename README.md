import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
# Load the data
data = pd.read_csv('human_behavior.csv')
# Explore the data
print(data.head())
print(data.describe())
# Create a scatter plot of the data
sns.scatterplot(x="x", y="y", data=data)
plt.show()
# Create a linear regression model
model = LinearRegression()
# Fit the model to the data
model.fit(data'x', data['y'])
# Print the model coefficients
print(model.coef_)
print(model.intercept_)
# Make predictions
predictions = model.predict(data'x')
# Plot the predictions
sns.scatterplot(x="x", y="y", data=data)
sns.lineplot(x="x", y="y", data=predictions)
plt.show()
# Evaluate the model
print(mean_squared_error(data['y'], predictions))
# Create an interface for the user to input data
def get_user_input():
    x = float(input("Enter a value for x: "))
    return x
# Make a prediction based on the user input
def make_prediction(x):
    prediction = model.predict(x)
    return prediction
# Print the prediction
def print_prediction(prediction):
    print(f"The predicted value for y is {prediction}.")
# Run the program
if __name__ == "__main__":
    x = get_user_input()
    prediction = make_prediction(x)
    print_prediction(prediction)
