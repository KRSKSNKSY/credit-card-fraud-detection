#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Replace 'your_file.csv' with the actual path or URL of your CSV file
file_path = r'C:\Users\user\Downloads\creditcard.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(data.head())


# In[2]:


x = data.drop(['Class'],axis = 1)
y = data['Class']
print(x.shape)
print(y.shape)

xData = x.values
yData = y.values


# In[3]:


data.shape


# In[4]:


data.describe()


# In[5]:


data.info()


# In[6]:


fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

Fractional_value = len(fraud)/len(valid)
print(Fractional_value)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transaction: {}'.format(len(data[data['Class'] == 0])))


# In[7]:


fraud.Amount.describe()


# In[8]:


valid.Amount.describe()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


correlation = data.corr()
plotting = plt.figure(figsize = (20,9))
sns.heatmap(correlation,vmax = 10, square = True)
plt.show()


# In[11]:


x = data.drop(['Class'],axis = 1)
y = data['Class']
print(x.shape)
print(y.shape)

xData = x.values
yData = y.values


# In[12]:


# Using Scikit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size = 0.2, random_state = 42)


# In[14]:


# Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
# Predictions
yPred = rfc.predict(xTest)


# In[17]:


# Evaluating the classifier
# printing every score of the classifier
# scoring in anything
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

n_outliers = len(fraud)
n_errors = (yPred != yTest).sum()
print("The model used is Random Forest Classifier")

acc = accuracy_score(yTest, yPred)
print("The accuracy is {}".format(acc))

prec = precision_score(yTest, yPred)
print("The precision is {}".format(prec))

rec = recall_score(yTest, yPred)
print("The recall is {}".format(rec))

f1 = f1_score (yTest, yPred)
print("The F1-Score is {}".format(f1))

MCC = matthews_corrcoef(yTest, yPred)
print("The Matthews correlation coefficient is {}".format(MCC))


# In[19]:


# printing the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize = (5, 5))
sns.heatmap(conf_matrix, xticklabels = LABELS,
            yticklabels = LABELS, annot = True, fmt = "d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[20]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset (replace 'creditcard.csv' with your actual file path)
data = pd.read_csv(r'C:\Users\user\Downloads\creditcard.csv')

# Assuming 'Class' is the target variable, and other columns are features
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)


# In[21]:


# Assuming the logistic regression model 'model' is already trained

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)

# Perform model validation (e.g., cross-validation)
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Assuming 'model' is the trained logistic regression model
cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')

# Print cross-validation results
print("\nCross-Validation Scores:")
print(cv_scores)
print(f"Mean Accuracy: {cv_scores.mean() * 100:.2f}%")


# In[22]:


from sklearn.model_selection import GridSearchCV

# Assuming X_train, y_train are the training data and labels

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    'penalty': ['l1', 'l2'],               # Regularization type
}

# Create a logistic regression model
model = LogisticRegression(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the best model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset (replace 'creditcard.csv' with your actual file path)
data = pd.read_csv(r'C:\Users\user\Downloads\creditcard.csv')

# Assuming 'Class' is the target variable, and other columns are features
X = data.drop('Class', axis=1)
y = data['Class']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Cleaning and Feature Engineering (you can customize this part based on your dataset)
# For simplicity, we'll use StandardScaler as a feature engineering step
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Data Balancing using SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Model Selection and Hyperparameter Tuning using Grid Search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
}

model = LogisticRegression(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_balanced, y_train_balanced)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Model Evaluation
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)


# In[ ]:




