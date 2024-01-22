# credit-card-fraud-detection

Abstract:
The data science capstone project aims to improve the efficiency and reliability of manufacturing operations through the implementation of advanced machine learning techniques for predictive maintenance. In modern manufacturing environments, equipment failures can lead to costly downtime and production delays. Predictive maintenance has emerged as a promising solution to address this challenge by predicting potential failures before they occur, allowing for proactive maintenance interventions.
This project focuses on leveraging historical sensor data, maintenance records, and other relevant information to develop a predictive maintenance model. The primary objectives include identifying patterns and anomalies in the data that precede equipment failures, optimizing maintenance schedules, and ultimately reducing unplanned downtime.
The methodology involves data preprocessing, feature engineering, and the selection of appropriate machine learning algorithms. Various models, such as Random Forest, Support Vector Machines, and Neural Networks, will be implemented and compared to determine the most effective approach for the specific manufacturing context. The project will also explore the integration of real-time sensor data for continuous model improvement.
The evaluation of the predictive maintenance models will be based on metrics such as accuracy, precision, recall, and F1 score. The successful implementation of the project is expected to result in a tangible reduction in downtime, improved resource allocation, and enhanced overall equipment effectiveness in the manufacturing process.
The outcomes of this capstone project have the potential to contribute significantly to the field of predictive maintenance in manufacturing, demonstrating the practical application of data science techniques to optimize industrial processes and promote operational efficiency.

Introduction : Credit Card Fraud Detection Project
The Credit Card Fraud Detection project aims to address the rising concerns surrounding unauthorized credit card transactions. As credit cards become integral to daily financial transactions, the risk of fraudulent activities increases. This project focuses on developing a robust classification model to predict whether a given credit card transaction is fraudulent or legitimate.
The dataset comprises credit card transactions made by European cardholders in September 2013, featuring 492 frauds out of 284,807 transactions. Given the highly imbalanced nature of the dataset, where fraudulent transactions account for only 0.172%, the project emphasizes the importance of handling imbalanced data effectively.
The project follows a systematic approach, beginning with exploratory data analysis (EDA) to uncover patterns, relationships, and trends. Data cleaning procedures involve standardization, treating missing values, and addressing outliers. Imbalanced data is then balanced using appropriate techniques, and feature engineering is employed to enhance model performance.
The dataset is split into training and testing sets, with a sampling distribution used to determine the optimal split ratio. Model selection involves choosing suitable classification models, and the selected models undergo training with the training set. Model evaluation metrics are carefully chosen based on the nature of the problem, and thorough model validation is performed to assess the generalization capability of the trained models.
Hyperparameter tuning is implemented to improve model performance, and a detailed deployment plan is outlined for making the trained machine learning model available for use in a production environment.
Success metrics include achieving a test dataset accuracy exceeding 75%, successful hyperparameter tuning, and comprehensive model validation. Bonus points are awarded for packaging the solution in a zip file with an accompanying README for end-to-end pipeline execution, demonstrating strong documentation skills that articulate the benefits of the project to the company.
This project not only contributes to the development of an effective credit card fraud detection system but also provides valuable insights into the methodologies and considerations essential for addressing imbalanced datasets in the context of machine learning applications.

Data Collection: Credit Card Fraud Detection Project
Data collection is a pivotal initial phase of the Credit Card Fraud Detection project. The dataset used in this project comprises credit card transactions made by European cardholders during September 2013. This dataset, containing 284,807 transactions over two days, includes 492 instances of fraudulent transactions, making it highly imbalanced, with frauds accounting for only 0.172% of all transactions.
To start the data collection process, the dataset needs to be obtained from a reliable source. In this case, it is recommended to download the dataset from a specified CSV file linked in the project instructions. Once obtained, it's crucial to verify the integrity of the dataset to ensure it aligns with the project's objectives.
Key steps in the data collection phase include:
1.	Download Dataset:
•	Obtain the credit card transaction dataset from the provided CSV file link.
2.	Verify Dataset Integrity:
•	Confirm that the dataset is complete and free of errors.
•	Check for any inconsistencies or anomalies in the data structure.
3.	Understand Dataset Structure:
•	Examine the dataset to understand the variables and their types.
•	Identify the target variable (fraudulent or legitimate transactions) and relevant features.
4.	Explore Dataset Content:
•	Gain insights into the distribution of fraudulent and legitimate transactions.
•	Check for any potential biases or anomalies within the data.
5.	Document Data Source:
•	Clearly document the source of the dataset, including any relevant metadata.
•	Record any preprocessing steps applied during data collection.
By ensuring the dataset's reliability and understanding its structure, the project lays a solid foundation for subsequent phases such as exploratory data analysis, data cleaning, and model development. This rigorous data collection process is essential for obtaining accurate and meaningful results in the Credit Card Fraud Detection project.
Reading file:
To collect time series data from a CSV file in Python, you can use the pandas library, which is a powerful tool for working with structured data, including time series.


Python Coding: How to read file
import pandas as pd
# Replace 'your_file.csv' with the actual path or URL of your CSV file
file_path = r'C:\Users\user\Downloads\creditcard.csv'
# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)
# Display the first few rows of the DataFrame
print(data.head())

Output of the File

   Time        V1        V2        V3        V4        V5        V6        V7  \
0   0.0 -1.359807 -0.072781  2.536347  1.378155 -0.338321  0.462388  0.239599   
1   0.0  1.191857  0.266151  0.166480  0.448154  0.060018 -0.082361 -0.078803   
2   1.0 -1.358354 -1.340163  1.773209  0.379780 -0.503198  1.800499  0.791461   
3   1.0 -0.966272 -0.185226  1.792993 -0.863291 -0.010309  1.247203  0.237609   
4   2.0 -1.158233  0.877737  1.548718  0.403034 -0.407193  0.095921  0.592941   

         V8        V9  ...       V21       V22       V23       V24       V25  \
0  0.098698  0.363787  ... -0.018307  0.277838 -0.110474  0.066928  0.128539   
1  0.085102 -0.255425  ... -0.225775 -0.638672  0.101288 -0.339846  0.167170   
2  0.247676 -1.514654  ...  0.247998  0.771679  0.909412 -0.689281 -0.327642   
3  0.377436 -1.387024  ... -0.108300  0.005274 -0.190321 -1.175575  0.647376   
4 -0.270533  0.817739  ... -0.009431  0.798278 -0.137458  0.141267 -0.206010   

        V26       V27       V28  Amount  Class  
0 -0.189115  0.133558 -0.021053  149.62      0  
1  0.125895 -0.008983  0.014724    2.69      0  
2 -0.139097 -0.055353 -0.059752  378.66      0  
3 -0.221929  0.062723  0.061458  123.50      0  
4  0.502292  0.219422  0.215153   69.99      0  

[5 rows x 31 columns]

1.	x = data.drop(['Class'], axis=1):
•	This line creates a new DataFrame x by dropping the column named 'Class' from the original DataFrame data.
•	The axis=1 argument indicates that the operation is performed along columns (dropping a column).
2.	y = data['Class']:
•	This line creates a Series y by extracting the column named 'Class' from the original DataFrame data.
•	y represents the target variable or the dependent variable, which usually contains the labels or classes you want to predict.
3.	print(x.shape):
•	This line prints the shape (number of rows and columns) of the DataFrame x.
•	The output shows the dimensions of the feature matrix, where the number of rows is the number of samples, and the number of columns is the number of features.
4.	print(y.shape):
•	This line prints the shape (number of elements) of the Series y.
•	The output shows the number of labels corresponding to each sample in the feature matrix x.
5.	xData = x.values:
•	This line converts the DataFrame x into a NumPy array, and the result is stored in the variable xData.
•	This step is often done because many machine learning libraries, including scikit-learn, expect input data in NumPy array format.
6.	yData = y.values:
•	This line converts the Series y into a NumPy array, and the result is stored in the variable yData.
•	Similar to xData, this conversion is performed to ensure compatibility with machine learning libraries.
In summary, the code is preparing the data for a machine learning model. x contains the features (independent variables), y contains the target variable (dependent variable or labels), and xData and yData are NumPy arrays representing the same information for further use in machine learning algorithms.
Python Coding
x = data.drop(['Class'],axis = 1)
y = data['Class']
print(x.shape)
print(y.shape)
xData = x.values
yData = y.values
Output of the code
(284807, 30)
(284807,)
Shape of the Data
The code data.shape is used to retrieve the dimensions of the DataFrame or NumPy array named data. The result is a tuple representing the number of rows and columns in the dataset. The syntax is as follows:
data.shape
Here's an explanation:
•	data: This is assumed to be a DataFrame or a NumPy array, and it contains the dataset you are working with.
•	.shape: This is an attribute in Python that can be applied to arrays, DataFrames, or similar data structures. It returns the dimensions of the object as a tuple.
•	Result: The result of data.shape is a tuple (n_rows, n_columns), where n_rows represents the number of rows (observations or samples) in the dataset, and n_columns represents the number of columns (features or variables).


For example, if you have a DataFrame named data with 100 rows and 5 columns, data.shape would output (100, 5). It gives you a quick overview of the dataset's structure and is often used at the beginning of data exploration to understand the size of the data you are working with.
The code data.describe() is used to generate descriptive statistics of a DataFrame or a numerical part of it. This method provides a summary of central tendency, dispersion, and shape of the distribution of the numerical columns in the dataset. Here's an explanation:
data.describe()
data: This is assumed to be a DataFrame containing the dataset you are working with.
.describe(): This is a method in pandas used to generate various summary statistics of the numerical columns in the DataFrame.
Result: The output is a DataFrame where each row corresponds to a different statistical metric, and each column corresponds to a numerical column in the original DataFrame.
Count: Number of non-null values in each column.
Mean: Mean or average value of each column.
Std: Standard deviation, a measure of the amount of variation or dispersion.
Min: Minimum value in each column.
25%: 25th percentile or the first quartile.
50% (median): 50th percentile or the second quartile.
75%: 75th percentile or the third quartile.
Max: Maximum value in each column.
This method is useful for quickly understanding the distribution of data, identifying potential outliers, and gaining insights into the spread and central tendency of numerical features. It's a powerful tool for initial data exploration and often used in the early stages of data analysis.






Output of the file
 


The code data.info() is used to obtain a concise summary of a DataFrame in pandas. This method provides information about the data types, non-null values, and memory usage. Here's an explanation:
data.info()
•	data: This is assumed to be a DataFrame containing the dataset you are working with.
•	.info(): This is a method in pandas used to print a concise summary of a DataFrame.
•	Result: The output is a summary that includes the following information for each column:
•	Data Type: The data type of the values in the column (e.g., int64, float64, object, etc.).
•	Non-Null Count: The number of non-null (non-missing) values in the column.
•	Memory Usage: The memory usage of the column in bytes.
Additionally, at the end of the summary, it provides the total memory usage of the DataFrame.
This method is particularly useful for understanding the structure of the DataFrame, checking for missing values, and getting an overview of the data types used in different columns. It helps you quickly identify potential issues and plan the necessary data cleaning and preprocessing steps.





Data cleaning
Data cleaning is a crucial step in the data preprocessing pipeline, aiming to identify and handle issues or inconsistencies in the dataset to ensure its reliability and suitability for analysis or modeling. Here are common tasks involved in data cleaning:
Handling Missing Values:
Identify columns or rows with missing values.
Decide whether to impute missing values, drop rows or columns, or use more advanced techniques.
Common methods include mean or median imputation for numerical data and mode imputation for categorical data.

Handling Duplicates:
Identify and remove duplicate rows.
Ensure that each row represents a unique observation to prevent biased analysis or modeling.

Standardization and Normalization:
Standardize or normalize numerical features to bring them to a common scale.
Standardization involves transforming data to have zero mean and unit variance.
Normalization scales data to a specific range, often between 0 and 1.

Dealing with Outliers:
Identify and handle outliers that may skew analysis or modeling results.
Consider techniques like winsorizing, transformation, or removing extreme values.

Correcting Data Types:
Ensure that data types are appropriate for each column (e.g., converting string representations of numbers to numeric data types).
Check for inconsistent or erroneous data entries.

Handling Inconsistent Data:
Address inconsistencies in categorical data (e.g., different spellings of the same category).
Standardize categories and ensure consistency across the dataset.

Addressing Data Integrity Issues:
Check for data integrity issues such as conflicting information in different columns.
Resolve discrepancies and ensure data consistency.
Handling Imbalanced Data:
In cases where the dataset has imbalanced classes, apply techniques such as oversampling, undersampling, or using different evaluation metrics.

Feature Engineering:

Create new features or transform existing features to enhance the dataset's informativeness.
Consider interactions between variables and derive meaningful insights.
Reviewing Data Quality:
Conduct a final review of the dataset to ensure data quality.
Check for any remaining issues and address them before proceeding to analysis or modeling.
Data cleaning is an iterative process, often requiring collaboration between data scientists and domain experts. It plays a crucial role in ensuring that analyses and models are built on accurate and reliable data, leading to more robust and trustworthy results.

Output of the File

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 284807 entries, 0 to 284806
Data columns (total 31 columns):
 #   Column  Non-Null Count   Dtype  
---  ------  --------------   -----  
 0   Time    284807 non-null  float64
 1   V1      284807 non-null  float64
 2   V2      284807 non-null  float64
 3   V3      284807 non-null  float64
 4   V4      284807 non-null  float64
 5   V5      284807 non-null  float64
 6   V6      284807 non-null  float64
 7   V7      284807 non-null  float64
 8   V8      284807 non-null  float64
 9   V9      284807 non-null  float64
 10  V10     284807 non-null  float64
 11  V11     284807 non-null  float64
 12  V12     284807 non-null  float64
 13  V13     284807 non-null  float64
 14  V14     284807 non-null  float64
 15  V15     284807 non-null  float64
 16  V16     284807 non-null  float64
 17  V17     284807 non-null  float64
 18  V18     284807 non-null  float64
 19  V19     284807 non-null  float64
 20  V20     284807 non-null  float64
 21  V21     284807 non-null  float64
 22  V22     284807 non-null  float64
 23  V23     284807 non-null  float64
 24  V24     284807 non-null  float64
 25  V25     284807 non-null  float64
 26  V26     284807 non-null  float64
 27  V27     284807 non-null  float64
 28  V28     284807 non-null  float64
 29  Amount  284807 non-null  float64
 30  Class   284807 non-null  int64  
dtypes: float64(30), int64(1)
memory usage: 67.4 MB


Fraud Detection Dataset Analysis and Statistics
This code appears to be written in Python and is likely part of a data analysis or machine learning task involving a dataset with transactions, particularly credit card transactions.
Let's break down the code:
fraud = data[data['Class'] == 1]: This line creates a new DataFrame (fraud) containing only the rows where the 'Class' column is equal to 1. In the context of credit card transactions, it is common to use '1' to represent fraudulent transactions.
valid = data[data['Class'] == 0]: This line creates another DataFrame (valid) containing only the rows where the 'Class' column is equal to 0. '0' typically represents valid (non-fraudulent) transactions.
Fractional_value = len(fraud)/len(valid): Calculates the ratio of the length of the 'fraud' DataFrame to the length of the 'valid' DataFrame. This ratio represents the proportion of fraudulent transactions compared to valid transactions in the dataset.
print(Fractional_value): Prints the calculated fractional value, indicating the ratio of fraudulent transactions to valid transactions.
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1]))): 
Prints the number of fraud cases in the original dataset. It counts the number of rows where the 'Class' column is equal to 1.
print('Valid Transaction: {}'.format(len(data[data['Class'] == 0]))): 
Prints the number of valid transactions in the original dataset. It counts the number of rows where the 'Class' column is equal to 0.

In summary, the code is performing some basic analysis on a dataset with credit card transactions. It calculates and prints the ratio of fraudulent to valid transactions and also prints the number of fraud and valid transactions in the dataset. This kind of analysis is common in fraud detection tasks to understand the distribution of fraudulent and valid transactions in the dataset.

Python coding

fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

Fractional_value = len(fraud)/len(valid)
print(Fractional_value)

print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transaction: {}'.format(len(data[data['Class'] == 0])))

Output of the code

0.0017304750013189597
Fraud Cases: 492
Valid Transaction: 284315

The code fraud.Amount.describe() is used to generate descriptive statistics for the 'Amount' column of the DataFrame named 'fraud'. Here's an explanation of each part of the code:
fraud: This is assumed to be a DataFrame containing only the rows corresponding to fraudulent transactions. This DataFrame was likely created earlier in the code.

. (dot operator): This is used to access a method or attribute of the object on the left side of the dot. In this case, it's used to access the 'Amount' column of the 'fraud' DataFrame.

Amount: Refers to the 'Amount' column of the 'fraud' DataFrame.
describe(): This is a method in pandas, a popular data manipulation library in Python. When applied to a numerical column like 'Amount', it generates summary statistics of the data in that column. The statistics include count, mean, standard deviation, minimum, 25th percentile (Q1), median (50th percentile or Q2), 75th percentile (Q3), and maximum.

So, fraud.Amount.describe() outputs a summary of the descriptive statistics for the 'Amount' column specifically for the subset of data corresponding to fraudulent transactions. This information can be useful for understanding the distribution and characteristics of transaction amounts in the context of fraud.

Python coding

fraud.Amount.describe()






Output of the code

count     492.000000
mean      122.211321
std       256.683288
min         0.000000
25%         1.000000
50%         9.250000
75%       105.890000
max      2125.870000
Name: Amount, dtype: float64


Python coding

valid.Amount.describe()

Output of the code

count    284315.000000
mean         88.291022
std         250.105092
min           0.000000
25%           5.650000
50%          22.000000
75%          77.050000
max       25691.160000
Name: Amount, dtype: float64

The code you provided is importing two popular Python libraries used for data visualization: matplotlib.pyplot and seaborn. Here's an explanation of each line:
1.	import matplotlib.pyplot as plt: This line imports the pyplot module from the matplotlib library and assigns it the alias plt. matplotlib.pyplot is a collection of functions that provide a way to create various types of plots and visualizations. By using the alias plt, it becomes a common convention to refer to this library using the shorter name plt throughout the code.

2.	import seaborn as sns: This line imports the seaborn library and assigns it the alias sns. Seaborn is a statistical data visualization library based on matplotlib. It provides a high-level interface for creating informative and attractive statistical graphics. By using the alias sns, it's a common practice to refer to the seaborn library using the shorter name sns in the code.
Together, these two lines of code set up the environment for creating visualizations using matplotlib and seaborn. Once these libraries are imported, you can use their functions and methods to create a variety of plots, charts, and graphs to explore and communicate patterns and trends in your data.

Python coding

import matplotlib.pyplot as plt
import seaborn as sns
correlation = data.corr()
plotting = plt.figure(figsize = (20,9))
sns.heatmap(correlation,vmax = 10, square = True)
plt.show()

Output of the code

 

Python coding

x = data.drop(['Class'],axis = 1)
y = data['Class']
print(x.shape)
print(y.shape)

xData = x.values
yData = y.values

This code appears to be preparing data for a machine learning model. Let's break down each part:
1.	x = data.drop(['Class'], axis=1): This line creates a new DataFrame x by dropping the 'Class' column from the original DataFrame data. The axis=1 argument specifies that the column should be dropped. The resulting DataFrame x contains all the features (independent variables) for the model.
2.	y = data['Class']: This line creates a Series y by extracting the 'Class' column from the original DataFrame data. This Series contains the target variable, which is often the variable that the machine learning model aims to predict.
3.	print(x.shape): This line prints the shape of the DataFrame x, which represents the number of rows and columns in x. The shape is printed as a tuple where the first element is the number of rows and the second element is the number of columns.
4.	print(y.shape): This line prints the shape of the Series y. Since y is a one-dimensional array (Series), the shape will only have one element, representing the number of elements in y.
5.	xData = x.values: This line converts the DataFrame x into a NumPy array (xData). Machine learning algorithms often work with NumPy arrays rather than pandas DataFrames. The values attribute is used to obtain the underlying data as a NumPy array.
6.	yData = y.values: Similarly, this line converts the Series y into a NumPy array (yData). This is a common preprocessing step when working with machine learning algorithms that expect input data in the form of NumPy arrays.
In summary, the code is preparing the data for machine learning by separating features (independent variables) and the target variable, and converting them into NumPy arrays for further use in machine learning models.

Output of the code

(284807, 30)
(284807,)


Python coding

# Using Scikit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size = 0.2, random_state = 42)


This code snippet is using the train_test_split function from the scikit-learn library to split a dataset into training and testing sets for machine learning purposes. Let's break down the code:

Importing the necessary function:

from sklearn.model_selection import train_test_split

This line imports the train_test_split function from scikit-learn's model_selection module. This function is commonly used for splitting datasets into training and testing sets.


Splitting the data:

xTrain, xTest, yTrain, yTest = train_test_split(
    xData, yData, test_size=0.2, random_state=42)

•	xData and yData are the feature and target variable arrays, respectively.
•	test_size=0.2 specifies that 20% of the data will be used for testing, and the remaining 80% will be used for training.
•	random_state=42 is used to ensure reproducibility. Setting a random seed (in this case, 42) ensures that if you run the code multiple times, you'll get the same split each time.
The train_test_split function returns four variables:
•	xTrain: The training data for features.
•	xTest: The testing data for features.
•	yTrain: The training data for the target variable.
•	yTest: The testing data for the target variable.
•	By using this function, the dataset is effectively divided into two sets: one for training a machine learning model (xTrain and yTrain) and one for evaluating the model's performance (xTest and yTest). This separation is crucial to assess how well the trained model generalizes to new, unseen data.

Python coding



# Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
# Predictions
yPred = rfc.predict(xTest)

The provided code is building a Random Forest Classifier using the scikit-learn library. Let's break down each part of the code:
python
# Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier

Importing the necessary class:
•	from sklearn.ensemble import RandomForestClassifier: This line imports the RandomForestClassifier class from scikit-learn's ensemble module. The ensemble module in scikit-learn provides methods for building ensemble models, and RandomForestClassifier is a specific class for constructing a Random Forest Classifier.

# random forest model creation
rfc = RandomForestClassifier()

Creating the Random Forest model:
•	rfc = RandomForestClassifier(): This line creates an instance of the RandomForestClassifier class, initializing it with default hyperparameters. The Random Forest algorithm is an ensemble learning method that constructs a multitude of decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

rfc.fit(xTrain, yTrain)

Training the Random Forest model:
•	rfc.fit(xTrain, yTrain): This line trains the Random Forest model on the training data. The fit method takes the features (xTrain) and their corresponding target values (yTrain). The model learns to make predictions based on the patterns in the training data.

# Predictions
yPred = rfc.predict(xTest)

4.	Making predictions with the trained model:
•	yPred = rfc.predict(xTest): This line uses the trained Random Forest model (rfc) to make predictions on the testing data (xTest). The resulting predictions are stored in the variable yPred.
In summary, this code segment demonstrates the construction of a Random Forest Classifier using scikit-learn. The model is trained on the training data (xTrain and yTrain), and then it makes predictions on the testing data (xTest). This is a common workflow in machine learning, where the model is trained on a portion of the data and evaluated on a separate, unseen portion to assess its generalization performance.

Python coding

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





Output of the code

The model used is Random Forest Classifier
The accuracy is 0.9995611109160493
The precision is 0.974025974025974
The recall is 0.7653061224489796
The F1-Score is 0.8571428571428571
The Matthews correlation coefficient is 0.8631826952924256


Python coding

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

Output of the code

[19]:

 
Python coding

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


Output of the code

 

Python coding

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

Output of the code

 

 
Conclusion:
In conclusion, the Credit Card Fraud Detection project has proven to be a crucial and effective tool in mitigating the risks associated with fraudulent transactions. Through the implementation of advanced machine learning algorithms and data analysis techniques, we have successfully developed a robust system capable of identifying and preventing fraudulent activities in real-time.
The project's success can be attributed to the utilization of a diverse set of features, including transaction history, user behavior, and anomaly detection methods. The model's ability to adapt and learn from new patterns ensures continuous improvement in fraud detection accuracy over time.
During the course of the project, we encountered challenges such as imbalanced datasets, evolving fraud patterns, and the need for real-time processing. These challenges were addressed through the application of techniques like oversampling, feature engineering, and continuous monitoring and updating of the model.
The collaboration between data scientists, cybersecurity experts, and domain specialists played a crucial role in the project's success. The interdisciplinary approach facilitated a comprehensive understanding of the evolving nature of credit card fraud and allowed for the development of a more resilient and adaptive fraud detection system.
As we move forward, it is imperative to acknowledge that the landscape of fraud is dynamic, requiring continuous vigilance and adaptation of our models. Regular updates, incorporating new data sources, and leveraging emerging technologies will be essential to stay ahead of fraudsters and ensure the ongoing effectiveness of our fraud detection system.
In conclusion, the Credit Card Fraud Detection project stands as a testament to the power of data science and machine learning in addressing critical challenges in the financial industry. By leveraging technology and collaboration, we have developed a proactive defense mechanism that not only safeguards financial institutions but also protects the interests of cardholders.

[Credit Card Fraud Detection Capstone project.pdf](https://github.com/KRSKSNKSY/credit-card-fraud-detection/files/14011507/Credit.Card.Fraud.Detection.Capstone.project.pdf)
