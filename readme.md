# Breast Cancer Prediction Classification - Logistic Regression

Logistic Regression is similar to Linear Regression but it's target values are categories not continuous numerical number. In this project, logistic regression is trained on [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) to predict breast tumor diagnosis based on the features from a digitized image of a breast mass.

### Steps to build Logistic Regression Model
1. Importing Libraries
2. Loading Dataset
   1. As there is no header row in the dataset, add column names into the dataset
3. Convert the target character values to numerical values (0 and 1) 
4. Split data columns into features and target
5. Split dataset into training and testing data
6. Feature Scaling as there may be a wide range of values over 30 columns
7. Create and train logistic regression model
8. Predict X_test_scaled and compare y_pred with actual y_test data
9. Evaluate the model

### Result
The logistic model in this project is 95% accurate.


### References

1. https://www.datacamp.com/tutorial/understanding-logistic-regression-python?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720830&utm_adgroupid=157156377311&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=703052949528&utm_targetid=dsa-2218886984100&utm_loc_interest_ms=&utm_loc_physical_ms=9190385&utm_content=&utm_campaign=230119_1-sea~dsa~tofu_2-b2c_3-us_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na-june24&gad_source=1&gclid=CjwKCAjw1emzBhB8EiwAHwZZxchsP-k6qmm8McIu46PVLAPd3xB6LGFvMPpKmf0xUgA9QPk7IWuBiBoCfUEQAvD_BwE
2. Wolberg,William, Mangasarian,Olvi, Street,Nick, and Street,W.. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.
   
