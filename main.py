#Breast Cancer Detection 

#Logistic Regression

#Importing Necessary Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

#Loading Dataset
col_names = ['ID','Diagnosis','radius1','texture1','perimeter1','area1','smoothness1','compactness1','concavity1','concave_points1','symmetry1','fractal_dimension1',
             'radius2','texture2','perimeter2','area2','smoothness2','compactness2','concavity2','concave_points2','symmetry2','fractal_dimension2',
             'radius3','texture3','perimeter3','area3','smoothness3','compactness3','concavity3','concave_points3','symmetry3','fractal_dimension3']
data = pd.read_csv("wdbc.data",header=None, names=col_names)
#print(data.head())
#print()

#Converting target char value to numerical value 
#Mapping 'B' to 0 and 'M' to 1
data['Diagnosis'] = data['Diagnosis'].map({'B':0, 'M': 1})

#Splitting dataset into features and target variables
X = data.drop(['ID','Diagnosis'],axis=1) #30 features
y = data['Diagnosis'] # target
#print(x.columns)
#print(y)

#Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Feature Scaling to ensure that all features equally contribute to target column
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Create and train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

#Predictions
y_pred = logreg.predict(X_test_scaled)

print("==== Actual vs Predicted Result ====")
for i in range(len(y_pred)):
    print(f"Actual Result: {y_test.iloc[i]}  Predicted Result: {y_pred[i]}")
print()

#Evaluate the model
cnf = confusion_matrix(y_pred, y_test)
print("Confusion Matrix: ")
print(f"{cnf}")

accuracy = accuracy_score(y_test, y_pred)
accuracy_percent = accuracy * 100
print(f"Accuracy of the model: {accuracy_percent:.2f}%")

