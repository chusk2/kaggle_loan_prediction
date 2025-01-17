# Kaggle Competion Notebook
## [Loan Prediction](https://www.kaggle.com/competitions/playground-series-s4e10)

This repository contains a Jupyter notebook (`kaggle_loan_prediction.ipynb`) that demonstrates the process of training a machine learning model for a Kaggle competition. The notebook includes data preprocessing, model training, evaluation, and deployment steps.

## Table of Contents

- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Model Deployment](#model-deployment)
- [Endpoint Invocation](#endpoint-invocation)
- [Conclusion](#conclusion)

## Introduction

The goal of this notebook is to build a machine learning model to predict loan status based on various features. The notebook covers the following steps:
1. Data Preprocessing
2. Model Training
3. Model Evaluation
4. Model Deployment
5. Endpoint Invocation

## Data Preprocessing

In this section, we load the dataset and perform necessary preprocessing steps, including handling missing values, encoding categorical features, and scaling numerical features.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('./data/loan_prediction/train.csv')

# Split the data into features and target
X = data.drop(columns=["loan_status"])
y = data["loan_status"]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X.select_dtypes('number').columns),
        ('cat', OneHotEncoder(), X.select_dtypes('object').columns))
    ]
)
```

## Model Training
We define a function to create a pipeline that includes preprocessing and the model. We then train the model using the training data.

```
from lightgbm import LGBMClassifier

# Define the function to create the pipeline
def create_pipeline(model):
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])
    return pipeline

# Create the pipeline with LGBMClassifier
model = LGBMClassifier(random_state=42)
pipeline = create_pipeline(model)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)
```

## Model Evaluation

We evaluate the model using the validation data and plot the ROC curve.

```python
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

# Plot the ROC curve using the fitted pipeline
fig, ax = plt.subplots(figsize=(8, 8))
RocCurveDisplay.from_estimator(pipeline, X_val, y_val, ax=ax)
ax.set_title('ROC Curve')
plt.show()
```

## Model Deployment

We convert the preprocessed data to JSON format and prepare it for endpoint invocation.

```python
import json

# Convert the preprocessed data to JSON format
json_data = preprocessed_df.to_json(orient='records')
data = json.loads(json_data)
```

## Endpoint Invocation

We send a POST request to the deployed server endpoint to get predictions.

```python
import requests

# Define the endpoint URL
url = "http://localhost:5001/invocations"

# Send a POST request to the server
response = requests.post(url, data=json_data, headers={"Content-Type": "application/json"})

# Check the response
if response.status_code == 200:
    print("Response from server:", response.json())
else:
    print("Failed to get a response. Status code:", response.status_code)
    print("Response content:", response.content)
```

# Conclusion

This notebook demonstrates the complete workflow of building, evaluating, and deploying a machine learning model for a Kaggle competition. By following the steps outlined in this notebook, you can train a model, evaluate its performance, and deploy it for making predictions on new data.

Feel free to explore the notebook and modify the code to suit your specific use case.