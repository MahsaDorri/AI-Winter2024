#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 17:29:33 2024

@author: mahsadorri
"""

## Enter Code Here ##
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx'
df_house= pd.read_excel(url)
print(df_house.head())
df_house
## Enter Code Here ##
df_house.head(5)
### Part 2: Data Preprocessing
## get the shape of data
df_house.shape
#  get the column values
print(df_house.columns.values)
# or
for col in df_house.columns: 
    print(col) 
#create summaries of data
df_house.describe()
# get the types of columns
df_house.dtypes
# replacing missing value with mean first finding missing value 2nd replacing missing value with mean
print(df_house.mean())
#Replace mean
df_house.fillna(df_house.mean(),inplace=True)
df_house.head(5)
#Correlation
corr_matrix = df_house.corr()
corr_matrix['Y house price of unit area'].sort_values(ascending=False)
#visualaize correlation matrix
import seaborn as sns
import matplotlib.pyplot as plt

# # Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()
#Diagram
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_house, x='X2 house age', y='Y house price of unit area')
plt.title('House Age vs Price')
plt.show()
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_house, x='X1 transaction date', y='Y house price of unit area')
plt.title('Transaction Date vs Price')
plt.show()
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_house, x='X3 distance to the nearest MRT station', y='Y house price of unit area')
plt.title('Distance to MRT vs Price')
plt.show()
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_house, x='X4 number of convenience stores', y='Y house price of unit area')
plt.title('Number of Convenience Stores vs Price')
plt.show()


### Part 3: Linear Regression

## Enter Code Here ##
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn import preprocessing

# Get column names first
names = df_house.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(df_house)

corr_matrix = df_house.corr()
corr_matrix['Y house price of unit area'].sort_values(ascending=False)
corr_matrix
X = df_house[['X4 number of convenience stores', 'X5 latitude', 'X6 longitude', 'X3 distance to the nearest MRT station']]
y = df_house['Y house price of unit area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#regression
lin_regg = LinearRegression()

lin_regg.fit(X_train, y_train)

y_pred1 = lin_regg.predict(X_test)


#DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
y_pred2= tree_reg.predict(X_test)
#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(X_train, y_train)
y_pred3 = forest_reg.predict(X_test)

### Part 4: Model Evaluation ###

mse1 = mean_squared_error(y_test, y_pred1)
lin_rmse1 = np.sqrt(mse1)
print('Mean Squared Error:', mse1)
print('Root Mean Squared Error:', lin_rmse1)

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred1})
print(comparison.head())

# for regression we have rmse which will help us to compare with other models.
#DecisionTreeRegressor evaluation
mse2 = mean_squared_error(y_test, y_pred2)
lin_rmse2 = np.sqrt(mse2)
print('Mean Squared Error:', mse2)
print('Root Mean Squared Error:', lin_rmse2)

#ForestReg
mse3= mean_squared_error(y_test, y_pred3)
lin_rmse3 = np.sqrt(mse3)
print('Mean Squared Error:', mse3)
print('Root Mean Squared Error:', lin_rmse3)

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
print(comparison.head())

#Cross-Validation
from sklearn.model_selection import cross_val_score

# For Linear Regression
lin_scores = cross_val_score(lin_regg, X_train, y_train,scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)

# For Decision Tree Regressor
tree_scores = cross_val_score(tree_reg, X_train, y_train,scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)

# For Forest Regressor
forest_scores = cross_val_score(tree_reg, X_train, y_train,scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)

# Function to display the scores
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# Display the results
print("Linear Regression Cross-Validation RMSE Scores:")
display_scores(lin_rmse_scores)

print("\nDecision Tree Regressor Cross-Validation RMSE Scores:")
display_scores(tree_rmse_scores)

print("\nRandom Forest Cross-Validation RMSE Scores:")
display_scores(forest_scores)
#given the lowest test RMSE and assuming the primary goal is to minimize prediction error,
# the Random Forest Regressor is the most appropriate model for your data but next mol is better baseon the results we have so we will fit it antest it
### Part 5: Model Tuning ###
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

#  GridSearchCV
param_grid = [
    {'n_estimators': [5, 15, 45], 'max_features': [3, 5, 6, 9]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [3, 4, 5]},
]

#  RandomForestRegressor
forest_reg = RandomForestRegressor(random_state=42)

# Setup GridSearchCV
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True)

grid_search.fit(X_train, y_train)
grid_search.best_params_
grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]): 
 print(np.sqrt(-mean_score), params)

 #best feature
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances

### Part 6: Model Deployment ###
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("Final MSE:", final_mse)
print("Final RMSE:", final_rmse)
comparison_final = pd.DataFrame({'Actual': y_test, 'Predicted': final_predictions})
print(comparison_final.head())
feature_importances = final_model.feature_importances_
features = list(X_train.columns)
importances = sorted(zip(feature_importances, features), reverse=True)
print("Feature Importances:")
for importance, feature in importances:
    print(f"{feature}: {importance}")
plt.figure(figsize=(10, 6))
sns.barplot(x=[importance for importance, _ in importances], y=[feature for _, feature in importances])
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance Ranking')
plt.show()
#predict
print("\nUsers can enter the value for the new real estate:")
attributes = []

# Allow the user to input the values
for col in X.columns:
    val = float(input(f"Enter value for the {col}: "))
    attributes.append(val)

# Deploy the actual dataset
predicted_price = final_model.predict([attributes])[0]
print("\nPredicted house price:", predicted_price)
