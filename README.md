# Bank Churn Classifier

This repository contains a Bank Churn Classifier built using an Artificial Neural Network (ANN) to predict if a customer will churn (leave the bank) or not. The model is trained on customer data to identify patterns and help in retention strategies. The classifier is designed with TensorFlow and trained on a labeled dataset, leveraging deep learning to achieve high accuracy.

## Overview

Customer churn is a critical metric for banks and financial institutions, representing the customers who stop using the services. By predicting churn, businesses can implement targeted retention strategies to improve customer loyalty and reduce turnover. This project utilizes an ANN to classify whether a customer is likely to churn based on various features.

## Features

- **Artificial Neural Network (ANN):** Implemented a multi-layer ANN model for classification.
- **TensorBoard for Training Visualization:** Real-time loss and accuracy visualization during training.
- **EarlyStopping Callback:** Prevents overfitting by stopping training when no further improvements are observed.
- **Deployment Ready:** The model is deployed online for easy access.

## Dataset

The dataset includes various customer attributes such as:
- Customer age
- Balance
- Number of products used
- Tenure, etc.

The target label indicates whether the customer has churned or not.

## Model Architecture

The ANN model is built with multiple hidden layers, each layer using ReLU activation for non-linearity, and a sigmoid activation function in the output layer to classify the probability of churn (1 for churn, 0 for no churn).

## How to Use

1. **Clone the repository:**

   ```bash
   git clone https://github.com/anonymous298/Bank_Churn.git
