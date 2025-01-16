# -Breast-Cancer-Classification-using-Machine-Learning-
Building and evaluating a Logistic Regression model to classify breast cancer data as either Malignant or Benign using the Breast Cancer Dataset from scikit-learn.



# Breast Cancer Prediction with Logistic Regression
This project uses the Breast Cancer Dataset from scikit-learn to build a machine learning model using Logistic Regression. The model classifies breast cancer instances into two categories: Malignant (0) or Benign (1).

Table of Contents
Project Overview
Technologies Used
Dataset
Installation
Usage
Model Evaluation
Predictive System
License

Project Overview
This project demonstrates the process of:
Loading and exploring a dataset
Preprocessing data (handling missing values, feature engineering, etc.)
Training a machine learning model using Logistic Regression
Evaluating the model's performance
Building a predictive system for breast cancer diagnosis

Technologies Used
Python (for all scripting)
scikit-learn (for machine learning models and datasets)
NumPy (for numerical operations)
Pandas (for data manipulation and analysis)

Dataset
The dataset used is the Breast Cancer Dataset provided by scikit-learn. It contains 30 features (measurements) that describe characteristics of the cell nuclei present in breast cancer biopsies. The target variable is a binary classification (0 for malignant and 1 for benign).
Features
The dataset contains 30 features, including:
Mean radius
Mean texture
Mean perimeter
Mean area
etc.
The target label is:
0 - Malignant
1 - Benign

Installation
To run this project locally, follow these steps:
Clone this repository to your local machine:

 git clone https://github.com/Oduobuk/ Breast-Cancer-Classification-using-Machine-Learning.git


Navigate to the project directory:

 cd breast-cancer-prediction


Install the necessary dependencies:

 pip install -r requirements.txt
 The requirements.txt file includes:


scikit-learn
pandas
numpy

Usage
1. Data Exploration and Preprocessing
Load the dataset using sklearn.datasets.load_breast_cancer().
Convert the dataset into a Pandas DataFrame and explore the data (e.g., check for missing values, describe statistics).
Split the data into features (X) and target labels (Y).
2. Model Training
Split the data into training and test sets using train_test_split().
Train the Logistic Regression model using model.fit().
3. Model Evaluation
Evaluate the model on both training and test data using accuracy_score().
4. Making Predictions
Input a set of feature values for a new breast cancer sample and get a prediction (Malignant or Benign).
To run the model and make a prediction, you can simply execute the script:
python breast_cancer_prediction.py

Sample Input:
input_data = (13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259)

Sample Output:
The Breast Cancer is Benign


Model Evaluation
The model is evaluated using the accuracy score:
Accuracy on training data
Accuracy on test data
The model will print the accuracy for both datasets and indicate the model's ability to generalize well to new data.

Predictive System
A predictive system is built using the trained Logistic Regression model, where you can input the features of a new breast cancer sample and the system will predict whether the cancer is Malignant or Benign.

License
This project is licensed under the MIT License - see the LICENSE file for details.




