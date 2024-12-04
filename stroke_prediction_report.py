#Stroke Prediction Model - Main File

#This Python script replicates the original RMarkdown file.

#add libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import joblib
from imblearn.over_sampling import SMOTE

#1: Import Data and Data Preprocessing

#load dataset
file_path='healthcare-dataset-stroke-data.csv' #Update with the correct file path
data=pd.read_csv(file_path)

#display first few rows
print(data.head())

#summary statistics
print(data.describe())

#convert categorical variables into numeric where needed
categorical_columns=['gender','ever_married','work_type','Residence_type','smoking_status']
data=pd.get_dummies(data,columns=categorical_columns,drop_first=True)

#clean + handle missing values
data.fillna(data.mean(),inplace=True)
print("Missing values handled.")

#2: Data Analysis

#visualize #1: dataset distributions
plt.figure(figsize=(10, 6))
sns.histplot(data['age'],kde=True,bins=30)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#visualize #2: correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(),annot=True,fmt=".2f",cmap="coolwarm")
plt.title('Correlation Heatmap')
plt.show()

#3: Build and Evaluate Prediction Model

#split data into training and testing sets
X=data.drop('stroke',axis=1) #independent variables
y=data['stroke'] #target variable

#handle class imbalance and resample if needed
smote=SMOTE(random_state=42)  #SMOTE to generate synthetic samples
X_res, y_res = smote.fit_resample(X, y)  #resample the data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#train Random Forest model (I didn't train the other model since I don't know how to...)
model=RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)

#predict and evaluate
y_pred=model.predict(X_test)
print("Model Accuracy:",accuracy_score(y_test,y_pred))
print("Classification Report:",classification_report(y_test,y_pred))

#save model
joblib.dump(model, 'stroke_prediction_model.pkl')  #save the model
joblib.dump(X.columns.tolist(), 'model_feature_names.pkl')  #save feature names