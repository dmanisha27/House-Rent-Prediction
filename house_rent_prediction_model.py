import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
data=pd.read_csv('D:\Projects\House rent prediction dataset\House_Rent_Dataset.csv')
print(data.head())
print("checking if the data contains null values or not")
print(data.isnull().sum())
print("the descriptive statistics of the data")
print(data.describe())
print("the mean, median, highest, and lowest rent of the houses")
print(f"Mean Rent: {data.Rent.mean()}")
print(f"Median Rent: {data.Rent.median()}")
print(f"Highest Rent: {data.Rent.max()}")
print(f"Lowest Rent: {data.Rent.min()}")
#the rent of the houses in different cities according to the number of bedrooms, halls, and kitchens
figure = px.bar(data, x=data["City"],
                y = data["Rent"],
                color = data["BHK"],
            title="Rent in Different Cities According to BHK")
figure.show()
#the rent of the houses in different cities according to the area type
figure = px.bar(data, x=data["City"],
                y = data["Rent"],
                color = data["Area Type"],
            title="Rent in Different Cities According to Area Type")
figure.show()
#the rent of the houses in different cities according to the size of the house
figure = px.bar(data, x=data["City"],
                y = data["Rent"],
                color = data["Size"],
            title="Rent in Different Cities According to Size")
figure.show()
#the number of houses available for rent in different cities according to the dataset
cities = data["City"].value_counts()
label = cities.index
counts = cities.values
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=label, values=counts, hole=0.5)])
fig.update_layout(title_text='Number of Houses Available for Rent')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()
#HOUSE RENT PREDICTION MODEL
#Converting all the categorical features into numerical features that we need to train a house rent prediction model
data["Area Type"]=data["Area Type"].map({"Super Area":1,"Carpet Area":2,"Build Area":3})
data["City"]=data["City"].map({"Mumbai":4000,"Chennai":6000,"Bangalore":5600,"Hyderabad":5000,"Delhi":1100,"Kolkata":7000})
data["Furnishing Status"]=data["Furnishing Status"].map({"Unfurnished":0,"Semi-Furnished":1,"Furnished":2})
data["Tenant Preferred"]=data["Tenant Preferred"].map({"Bachelors/Family":2,"Bachelors":1,"Family":3})
#splittling data into training and test sets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
x=np.array(data[["BHK","Size","Area Type","City","Furnishing Status","Tenant Preferred","Bathroom"]])
y=np.array(data[["Rent"]]).ravel()
x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.10,random_state=42)
#Initialize the Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
#Train the model
model.fit(xtrain, ytrain)
#Make predictions on the test set
ypred = model.predict(xtest)
#Evaluate the model
mae = mean_absolute_error(ytest, ypred)
mse = mean_squared_error(ytest, ypred)
rmse = np.sqrt(mse)
print()
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print()
# Predict Rent based on User Input
print("Enter House Details to Predict Rent")
a = int(input("Number of BHK: "))
b = int(input("Size of the House: "))
c = int(input("Area Type (Super Area = 1, Carpet Area = 2, Built Area = 3): "))
d = int(input("Pin Code of the City: "))
e = int(input("Furnishing Status of the House (Unfurnished = 0, Semi-Furnished = 1, Furnished = 2): "))
f = int(input("Tenant Type (Bachelors = 1, Bachelors/Family = 2, Only Family = 3): "))
g = int(input("Number of bathrooms: "))
# Create feature array for prediction
features = np.array([[a, b, c, d, e, f, g]])
# Predict and display the rent
predicted_rent = model.predict(features)
print("Predicted House Rent_p =", predicted_rent[0])