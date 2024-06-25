#Breast Cancer Detection 

#Logistic Regression

#Importing Necessary Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Loading Dataset
col_names = ['ID','Diagnosis','radius1','texture1','perimeter1','area1','smoothness1','compactness1','concavity1','concave_points1','symmetry1','fractal_dimension1',
             'radius2','texture2','perimeter2','area2','smoothness2','compactness2','concavity2','concave_points2','symmetry2','fractal_dimension2',
             'radius3','texture3','perimeter3','area3','smoothness3','compactness3','concavity3','concave_points3','symmetry3','fractal_dimension3']
data = pd.read_csv("wdbc.data",header=None, names=col_names)
#print(data.head())
#print()

#Converting target char value to numerical value 
data['Diagnosis'] = data['Diagnosis'].map({'B':0, 'M': 1})

#Splitting dataset into features and target variables
x = data.drop(['ID','Diagnosis'],axis=1) #30 features
y = data['Diagnosis'] # target
#print(x.columns)
#print(y)

#Visualizing dataset with scatterplot
#plt.scatter(x,y)
#plt.title("Scatter Plot of Breast Cancer Prediction Logistic Regression")
#plt.show()