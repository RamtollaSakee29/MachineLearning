# Start of Package import #
import pandas as panda

import matplotlib.pyplot as plot
import sys

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# End of Package import #

# Logics Of linear and polynomial approaches =>

# Load data into the Dataframe
Dataframe = panda.read_csv('marks_v2.csv')
# TEST: Check if the Dataset is feeded correctly 
# # print(Dataframe.to_string())

# TEST: Check if dataset is well plotted
# # Plot 'hours' as independent variables, 'Marks' as dependent Variables
# # Dataframe.plot(kind='scatter', x='hours',y='marks')
# # plot.show()

# Convert to array - PANDAS <=> NUMPY
# reshape(-1,1) - Array is converted to 1 column and 55 rows
hours = Dataframe['hours'].to_numpy().reshape(-1,1)
marks = Dataframe['marks'].to_numpy()
# TEST: Check if independent variable data has been converted to array 
# # hours_col = hours.reshape(-1,1)
# # print(hours_col)

# Train,Test set : 70,30 (Size - 0.3) 
# Random_state - 42 to ensure we get the same train and test set across different executions
hours_train, hours_test, marks_train, marks_test = train_test_split(hours,marks,test_size=0.3,random_state=42)

# 1. Linear approaches =>
modelLinear = LinearRegression().fit(hours_train,marks_train)
predictMarks = modelLinear.predict(hours_test)

# Output : Evaluations of model
# Finding the slope and intercept of the line of best fit
print("Intercept line of best fit   -", modelLinear.intercept_)
print("Coefficient line of best fit -", modelLinear.coef_)
# The R2 is high - Line of best fit is not biased
# Evaluated error between test data and predicted data 
print('R2 value          -', metrics.r2_score(marks_test, predictMarks))
print('Mean square Error -', metrics.mean_squared_error(marks_test, predictMarks))

#Plotting the results 
plot.figure(figsize = (10,6))
# Mark all the points
plot.scatter(hours, marks , color = 'green')
# Display the best fit line
plot.plot(hours_train, modelLinear.predict(hours_train) , color = 'b' , linewidth = 1)
# Label the axes
plot.xlabel('Hours' , size = 20)
plot.ylabel('Marks', size = 20)
plot.show()

print('-----------------------End of Linear Regression------------------------------------')
# End of Linear Approaches # 


# 2. Polynomial approaches =>
# Make X-axis as 1D for poly training
poly_hour_train = hours_train.flatten()

# Output : Evaluation of models for different degree of polynomial
# Get best fit line
# R2 and MSE of predicted value to check if the predicted values are accurate
# # R2 may not be so reliable as it evaluates the scatter of the data around the fitted regression line
# # MSE evaluates the differences predicted and the actual values

# Degree 2 
modelPoly2 = np.poly1d(np.polyfit(poly_hour_train,marks_train,2))
predictMarks_poly2 = modelPoly2(hours_test)
print('Degree 2, R2 Value          -',metrics.r2_score(marks_test, predictMarks_poly2))
print('Degree 2, Mean square Error -', metrics.mean_squared_error(marks_test, predictMarks_poly2))

# Degree 3
modelPoly3 = np.poly1d(np.polyfit(poly_hour_train,marks_train,3))
predictMarks_poly3 = modelPoly3(hours_test)
print('Degree 3, R2 Value          -',metrics.r2_score(marks_test, predictMarks_poly3))
print('Degree 3, Mean square Error -', metrics.mean_squared_error(marks_test, predictMarks_poly3))

# Degree 4
modelPoly4 = np.poly1d(np.polyfit(poly_hour_train,marks_train,4))
predictMarks_poly4 = modelPoly4(hours_test)
print('Degree 4, R2 Value          -',metrics.r2_score(marks_test, predictMarks_poly4))
print('Degree 4, Mean square Error -', metrics.mean_squared_error(marks_test, predictMarks_poly4))

#Plotting the result of polynomial with degree 4 as it is the better fit
# myline = np.linspace(1, 100, 100)
# plot.scatter(hours, marks)
# plot.xlabel('Hours' , size = 20)
# plot.ylabel('Marks', size = 20)
# plot.plot(myline, modelPoly4(myline))
# plot.show()

print('-----------------------End of Polynomial Regression------------------------------------')
# End of Polynomial Approaches # 




