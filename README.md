# Cervical Cancer Risk Prediction
[Overview](#overview) | [Key Features](#key-features) | [How to Install](#how-to-install) | [Credits](#credits)

## Overview

Cervical cancer poses a significant health risk to women's health, claiming thousands of lives annually both in the United States and globally. This script creates a Machine Learning model, utilizing an XGBoost classifier, used to predict the likelihood of an individual developing cervical cancer. The script asks the user for their age, smoking status, age of first sexual intercourse, number of pregancies, and IUD status and then predicts their risk of cervical cancer using the model.

## Key Features

- Importing dataset using Python libraries.
- Preprocessing data, including handling missing values and converting categorical variables to numeric.
- Splitting and standardizing the training and testing dataset.
- Training the XGBoost model using Scikit-Learn.
- Evaluating and analyzing the performance of the classifier models.
- Predicting the risk of cervical cancer based on user input.
    ![Screenshot of cervical cancer script running](/cervicalCancerRun.png)

## How to Install

1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed. You can install them using pip:

```bash
    pip install pandas numpy seaborn matplotlib plotly xgboost jupyterthemes
```

3. Run the provided Python script (cervical_cancer_prediction.py) in your preferred Python environment.
4. Follow the prompts to input relevant information such as age, smoking habits, age of first sexual intercourse, number of pregnancies, and use of intrauterine device (IUD).
5. After providing the required information, the script will predict the risk of cervical cancer and display the result as a percentage.

## Credits

This project is based on the Cervical Cancer Risk Prediction Using Machine Learning Coursera Project Network. The dataset (cervical_cancer.csv) is borrowed from the same source.