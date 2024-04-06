import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import zipfile
import plotly.express as px
from jupyterthemes import jtplot
jtplot.style(theme = 'monokai', context = 'notebook', ticks = True, grid = False)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Read the dataset
cancer_df = pd.read_csv('cervical_cancer.csv')
print("Dataset loaded successfully.")

#Exploratory Data Analysis
cancer_df = cancer_df.replace('?', np.nan)

#Print heat map of null values
# plt.figure(figsize=(20,20))
# sns.heatmap(cancer_df.isnull(),yticklabels=False)

# Drop columns with many null values
cancer_df = cancer_df.drop(columns= ['STDs: Time since first diagnosis','STDs: Time since last diagnosis'])

# Convert to numeric data
cancer_df = cancer_df.apply(pd.to_numeric)

#Replace nulls with the median
cancer_df = cancer_df.fillna(cancer_df.median())
print("Data preprocessing completed.")

'''
corr_matrix = cancer_df.corr()
# OPTIONAL: print the correlational data
print("Corr Matrix:", corr_matrix)
plt.figure(figsize=(30,30))
sns.heatmap(corr_matrix, annot=True)
plt.show()
cancer_df.hist(bins=10, figsize=(30,30), color='b')
'''

# Split the data into training and testing sets
target_df = cancer_df['Biopsy']
input_df = cancer_df.drop(columns=['Biopsy'])

x = np.array(input_df).astype('float32')
y = np.array(target_df).astype('float32')

scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20)
x_test, x_val, y_test, y_val = train_test_split(x_test,y_test,test_size = 0.5)
print("Data split into training and testing sets.")

# Train the XGBoost model
model = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 5, n_estimators = 50)
model.fit(x_train,y_train)
print("Model trained successfully.")

# Evaluate the model
result_train = model.score(x_train, y_train)
result_test = model.score(x_test, y_test)
print("Model evaluation:")
print(" - Accuracy on training data:", result_train)
print(" - Accuracy on testing data:", result_test)

# Function to preprocess input data and predict cervical cancer risk
def predict_cervical_cancer_risk(age, smoke, first_intercourse, num_pregnancies, iud):
    # Create a dictionary containing input data
    user_input_data = {
        'Age': [age],
        'Number of sexual partners': cancer_df['Number of sexual partners'].median(), 
        'First sexual intercourse': [first_intercourse], 
        'Num of pregnancies': [num_pregnancies], 
        'Smokes': [1 if smoke.lower() == 'y' else 0],
        'Smokes (years)': cancer_df['Smokes (years)'].median(), 
        'Smokes (packs/year)': cancer_df['Smokes (packs/year)'].median(), 
        'Hormonal Contraceptives': cancer_df['Hormonal Contraceptives'].median(), 
        'Hormonal Contraceptives (years)': cancer_df['Hormonal Contraceptives (years)'].median(), 
        'IUD': [1 if iud.lower() == 'y' else 0], 
        'IUD (years)': cancer_df['IUD (years)'].median(), 
        'STDs': cancer_df['STDs'].median(),
        'STDs (number)': cancer_df['STDs (number)'].median(), 
        'STDs:condylomatosis': cancer_df['STDs:condylomatosis'].median(), 
        'STDs:cervical condylomatosis': cancer_df['STDs:cervical condylomatosis'].median(), 
        'STDs:vaginal condylomatosis': cancer_df['STDs:vaginal condylomatosis'].median(), 
        'STDs:vulvo-perineal condylomatosis': cancer_df['STDs:vulvo-perineal condylomatosis'].median(), 
        'STDs:syphilis': cancer_df['STDs:syphilis'].median(), 
        'STDs:pelvic inflammatory disease': cancer_df['STDs:pelvic inflammatory disease'].median(), 
        'STDs:genital herpes': cancer_df['STDs:genital herpes'].median(), 
        'STDs:molluscum contagiosum': cancer_df['STDs:molluscum contagiosum'].median(), 
        'STDs:AIDS': cancer_df['STDs:AIDS'].median(), 
        'STDs:HIV': cancer_df['STDs:HIV'].median(), 
        'STDs:Hepatitis B': cancer_df['STDs:Hepatitis B'].median(), 
        'STDs:HPV': cancer_df['STDs:HPV'].median(), 
        'STDs: Number of diagnosis': cancer_df['STDs: Number of diagnosis'].median(), 
        'Dx:Cancer': cancer_df['Dx:Cancer'].median(), 
        'Dx:CIN': cancer_df['Dx:CIN'].median(), 
        'Dx:HPV': cancer_df['Dx:HPV'].median(), 
        'Dx': cancer_df['Dx'].median(), 
        'Hinselmann': cancer_df['Hinselmann'].median(), 
        'Schiller': cancer_df['Schiller'].median(), 
        'Citology': cancer_df['Citology'].median()
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame(user_input_data)

    # Predict probability of positive outcome (cancer)
    prediction = model.predict_proba(input_df)[:, 1]
    
    return prediction

# User input
age = float(input("Enter age: "))
smoke = input("Do you smoke? (y/n): ")
first_intercourse = float(input("Enter age of first sexual intercourse: "))
num_pregnancies = float(input("Enter number of pregnancies: "))
iud = input("Do you use IUD? (y/n): ")

risk_prediction = predict_cervical_cancer_risk(age, smoke, first_intercourse, num_pregnancies, iud)
risk_percentage = risk_prediction * 100
formatted_percentage = "{:.2f}%".format(risk_percentage[0])
print("Predicted risk of cervical cancer:", formatted_percentage)
