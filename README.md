Customer Churn Prediction
Project Overview

Customer churn is when a customer stops using a company’s services. Predicting churn helps businesses retain valuable customers and reduce revenue loss. This project uses machine learning to predict whether a customer is likely to churn based on historical customer data. A neural network model is trained to classify customers and provide churn probabilities.

Dataset

The dataset includes customer information such as:

Demographics: Age, Gender, Geography

Account info: Credit Score, Tenure, Balance, Number of Products

Behavior: Active Member, Has Credit Card, Estimated Salary

Target variable: Exited (1 = churn, 0 = stay)

The dataset is preprocessed by removing unnecessary columns and encoding categorical variables.

Features

Data cleaning and preprocessing (drop irrelevant columns, encode categories)

Train/test split and feature scaling for neural network efficiency

Deep learning model using TensorFlow/Keras

Predict churn probability for new customers

Evaluate model performance using accuracy and metrics

Requirements

Python 3.11+

Libraries: pandas, numpy, scikit-learn, tensorflow, matplotlib

Install dependencies via pip:

pip install pandas numpy scikit-learn tensorflow matplotlib

Usage

Load and preprocess the dataset.

Split data into training and testing sets.

Scale features using StandardScaler.

Train the neural network:

model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)


Predict churn on new customers:

prob = model.predict(new_customer_scaled)
prediction = 1 if prob > 0.5 else 0


Evaluate model performance with accuracy score:

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

Notes

Ensure new customer inputs are scaled and encoded exactly like the training data.

The model helps businesses identify at-risk customers and design retention strategies.
