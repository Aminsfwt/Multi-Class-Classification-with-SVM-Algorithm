import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.inspection import permutation_importance

# Read the data
data = pd.read_csv('E:/ML/Intro to Deep Learning/Labs/Codes/Applied ML/Glass Identification/glass.csv')

#Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

#extract features and target variable
X = data.drop(columns=['Type'])
y = data['Type']

#Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Train the SVM model 
#Use RBF Kernel (default for non-linear separation)
#gamma = 'scale' is calculated as 1 / (n_features * X.var())
#C is the regularization parameter, which controls the trade-off between maximizing the margin and minimizing the classification error.
#Higher values of C lead to a smaller margin and more support vectors, while lower values of C lead to a larger margin and fewer support vectors.
#The default value of C is 1.0.
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42) 
svm_model.fit(X_train, y_train)

#compute the accuracy of the model
svm_accuracy = svm_model.score(X_test, y_test)
print(f"SVM With RBF Kernel Model Accuracy: {svm_accuracy:.2f}")

#Hyperparameter tuning using GridSearchCV to find optimal parameters
param_grid = {
              'C': [0.1, 1, 10, 100],
              'gamma': ['scale', 'auto', .1, 1.0], 
              'kernel' : ['linear','rbf']}

grid_search = GridSearchCV(SVC(class_weight="balanced"), param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters found: ", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

#Get the accuracy after hyperparameter tuning
y_pred = best_model.predict(X_test)
print(f"Accuracy after hyperparameter tuning: {accuracy_score(y_test, y_pred):.2f}")

# Print classification report
print(classification_report(y_test, y_pred))

# Print feature importances using permutation importance
result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42)
for i, importance in enumerate(result.importances_mean):
    print(f"Feature {X.columns[i]}: {importance:.4f}")

"""
# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
"""

"""
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

fig, ax = plt.subplots(figsize=(20, 9))
for i in range(X_scaled_df.shape[1]):
    ax.scatter(X_scaled_df.iloc[:, i], X_scaled_df.iloc[:, (i + 1) % X_scaled_df.shape[1]], label=X_scaled_df.columns[i])
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Scatter plot of features')
ax.legend()
plt.show()
"""