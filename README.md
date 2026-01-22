Customer Churn Prediction
Project Overview

Customer churn occurs when customers stop using a company’s products or services. Predicting churn is crucial because retaining existing customers is often more cost-effective than acquiring new ones. This project uses machine learning to predict whether a customer is likely to leave, helping businesses take proactive retention measures.

The model is built using historical customer data, including demographic details, account information, and behavioral indicators. By analyzing these features, the neural network learns patterns that distinguish between customers who stay and those who churn.

Dataset

The dataset includes the following features:

Feature	Description
CreditScore	Customer’s credit score
Age	Customer age
Tenure	Number of years the customer has been with the bank
Balance	Account balance
NumOfProducts	Number of products the customer uses
HasCrCard	Whether the customer has a credit card (1 = Yes, 0 = No)
IsActiveMember	Whether the customer is active (1 = Yes, 0 = No)
EstimatedSalary	Estimated annual salary
Geography	Customer location (encoded as Germany/Spain/France)
Gender	Customer gender (encoded as Male/Female)
Exited	Target variable: 1 = churn, 0 = stay

The dataset is preprocessed by removing irrelevant columns and encoding categorical variables.

Features

Data cleaning and preprocessing

Train/test split and feature scaling for neural network input

Neural network classifier using TensorFlow/Keras

Predict churn probabilities for new customers

Evaluate model performance using accuracy

Requirements

Python 3.11+

Libraries: pandas, numpy, scikit-learn, tensorflow, matplotlib

Install dependencies:

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


Evaluate performance:

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

Notes

Important: New customer data must be preprocessed, encoded, and scaled exactly like the training data before making predictions.

This model helps businesses identify at-risk customers and design targeted retention strategies, reducing revenue loss and improving customer satisfaction.
