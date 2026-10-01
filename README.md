# Credit Card Fraud Detection

## Overview

This project focuses on detecting fraudulent credit card transactions using machine learning.

The project was developed as a university project in collaboration with a professor. The main goal is to explore how machine learning models can be used to identify fraudulent transactions and distinguish them from legitimate transactions.

## Dataset

The project uses the **Credit Card Fraud Detection** dataset.

The dataset contains **284,807 transactions**, of which **492 are fraudulent**. This makes fraud detection a challenging classification problem because the dataset is highly imbalanced.

The dataset includes:

* `Time`
* `Amount`
* `V1` – `V28`
* `Class`

The `Class` column represents the transaction type:

* `0` — Legitimate transaction
* `1` — Fraudulent transaction

## Project Goals

The main objectives of this project are:

* Explore and understand the dataset
* Analyze fraudulent and legitimate transactions
* Prepare the data for machine learning
* Handle the highly imbalanced dataset
* Train machine learning models
* Evaluate and compare model performance
* Identify fraudulent transactions as accurately as possible

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Jupyter Notebook

## Project Workflow

The project follows these main steps:

1. Load the dataset
2. Explore and analyze the data
3. Perform data preprocessing
4. Analyze the class imbalance
5. Split the dataset into training and testing sets
6. Train machine learning models
7. Make predictions
8. Evaluate the models
9. Analyze the results

## Evaluation

Since fraudulent transactions represent only a very small percentage of the dataset, accuracy alone is not enough to evaluate the models.

The project uses evaluation metrics such as:

* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC

These metrics help evaluate how well the models detect fraudulent transactions while minimizing incorrect classifications.

## Results

The trained models are evaluated on unseen test data to determine their ability to identify fraudulent transactions.

The results are used to compare the performance of the selected machine learning approaches and understand the challenges of fraud detection on highly imbalanced data.

## Future Improvements

Possible future improvements include:

* Testing additional machine learning models
* Improving data preprocessing
* Experimenting with different methods for handling class imbalance
* Hyperparameter tuning
* Further feature analysis
* Improving fraud detection performance

## Author

**Amina Tynybekova**

University project developed in collaboration with a professor.
