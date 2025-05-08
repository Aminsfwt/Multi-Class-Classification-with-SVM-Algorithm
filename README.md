# Glass Identification Using SVM

This repository contains an implementation of a multi-class classification model using Support Vector Machines (SVM) for identifying types of glass based on their chemical composition.
The project uses Python libraries such as Pandas, NumPy, scikit-learn, seaborn, and matplotlib to preprocess the data, 
train the model, perform hyperparameter tuning with GridSearchCV, and evaluate the model performance.

## Repository Structure

```
 Glass Identification/
   ├──glass.csv                 # Dataset containing glass measurements and target 'Type'
   └─ MultiClass_SVM.py         # Main script for data preprocessing, training, hyperparameter tuning, and evaluation
```

## Overview

- **Data Preprocessing:**  
  - The dataset is read from the `glass.csv` file.
  - Data is shuffled and split into features and target.
  - Features are standardized using `StandardScaler`.

- **Model Training and Evaluation:**  
  - An initial SVM model (with an RBF kernel) is trained and its accuracy is evaluated.
  - Hyperparameter tuning is performed using `GridSearchCV` with a parameter grid that includes variations for `C`, `gamma`, and `kernel` with balanced class weights.
  - After tuning, the best parameters and model accuracy are printed, along with a detailed classification report.
  - Feature importances are computed using permutation importance.

## Requirements

- Python 3.x
- Pandas
- NumPy
- scikit-learn
- seaborn
- matplotlib

Install the required packages using:

````bash
pip install pandas numpy scikit-learn seaborn matplotlib
