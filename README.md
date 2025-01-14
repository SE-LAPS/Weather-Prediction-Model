# Weather Prediction Model Report

## Table of Contents
* Instructions
* About the Data
* Importing Data
* Data Preprocessing
* One Hot Encoding
* Train and Test Data Split
* Train Models and Return Accuracy Scores
  - Linear Regression
  - KNN
  - Decision Tree
  - Logistic Regression
  - SVM

## Instructions
In this notebook, we will practice various classification algorithms learned throughout the course. Our objective is to create models based on the given dataset, evaluate their performance, and generate a comprehensive report.

### Models to be used:
1. Linear Regression
2. K-Nearest Neighbors (KNN)
3. Decision Trees
4. Logistic Regression
5. Support Vector Machine (SVM)

### Evaluation Metrics:
- Accuracy Score
- Jaccard Index
- F1-Score
- Log Loss (only for Logistic Regression)
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R2-Score

## About The Dataset
The dataset contains weather observations from 2008 to 2017, with various weather metrics. The objective is to predict if it will rain the next day (`RainTomorrow`). The target column is `RainTomorrow`, and it contains binary values (`Yes`/`No`).

### Dataset Fields:

| **Field**           | **Description**                                               | **Unit**           | **Type**  |
|---------------------|---------------------------------------------------------------|--------------------|-----------|
| Date                | Date of the observation in YYYY-MM-DD                         | Date               | object    |
| Location            | Location of the observation                                   | Location           | object    |
| MinTemp             | Minimum temperature                                           | Celsius            | float     |
| MaxTemp             | Maximum temperature                                           | Celsius            | float     |
| Rainfall            | Amount of rainfall                                            | Millimeters        | float     |
| Evaporation         | Amount of evaporation                                         | Millimeters        | float     |
| Sunshine            | Amount of bright sunshine                                     | Hours              | float     |
| WindGustDir         | Direction of the strongest gust                               | Compass Points     | object    |
| WindGustSpeed       | Speed of the strongest gust                                   | Kilometers/Hour    | float     |
| WindDir9am          | Wind direction averaged over 10 minutes prior to 9am          | Compass Points     | object    |
| WindDir3pm          | Wind direction averaged over 10 minutes prior to 3pm          | Compass Points     | object    |
| WindSpeed9am        | Wind speed averaged over 10 minutes prior to 9am              | Kilometers/Hour    | float     |
| WindSpeed3pm        | Wind speed averaged over 10 minutes prior to 3pm              | Kilometers/Hour    | float     |
| Humidity9am         | Humidity at 9am                                               | Percent            | float     |
| Humidity3pm         | Humidity at 3pm                                               | Percent            | float     |
| Pressure9am         | Atmospheric pressure reduced to mean sea level at 9am         | Hectopascal        | float     |
| Pressure3pm         | Atmospheric pressure reduced to mean sea level at 3pm         | Hectopascal        | float     |
| Cloud9am            | Fraction of the sky obscured by cloud at 9am                  | Eights             | float     |
| Cloud3pm            | Fraction of the sky obscured by cloud at 3pm                  | Eights             | float     |
| Temp9am             | Temperature at 9am                                            | Celsius            | float     |
| Temp3pm             | Temperature at 3pm                                            | Celsius            | float     |
| RainToday           | If there was rain today                                       | Yes/No             | object    |
| RainTomorrow        | If there is rain tomorrow                                     | Yes/No             | float     |

## Importing Data
```python
from pyodide.http import pyfetch

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f:
            f.write(await response.bytes())

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv'

await download(path, "Weather_Data.csv")
filename = "Weather_Data.csv"

import pandas as pd

# Load the dataset
df = pd.read_csv("Weather_Data.csv")
df.head()
```

## Data Preprocessing
```python
# Drop the Date column
df.drop('Date', axis=1, inplace=True)

# One Hot Encoding for categorical variables
df = pd.get_dummies(data=df, columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])

# Replace RainTomorrow values with binary values (0 for No, 1 for Yes)
df.replace(['No', 'Yes'], [0, 1], inplace=True)

# Convert all columns to float type
df = df.astype(float)

# Set features and target
features = df.drop(columns='RainTomorrow', axis=1)
Y = df['RainTomorrow']
```

## Train and Test Data Split
```python
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size=0.2, random_state=10)
```

## Model Training and Evaluation

### Linear Regression
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LinearReg = LinearRegression()
LinearReg.fit(x_train, y_train)
predictions = LinearReg.predict(x_test)

# Calculate metrics
LinearRegression_MAE = mean_absolute_error(y_test, predictions)
LinearRegression_MSE = mean_squared_error(y_test, predictions)
LinearRegression_R2 = r2_score(y_test, predictions)

# Display metrics in a tabular format
Report = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'R2'],
    'Linear Regression': [LinearRegression_MAE, LinearRegression_MSE, LinearRegression_R2]
})
Report
```

### K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier(n_neighbors=4)
KNN.fit(x_train, y_train)
predictions = KNN.predict(x_test)

# Calculate metrics
KNN_Accuracy_Score = accuracy_score(y_test, predictions)
KNN_JaccardIndex = jaccard_score(y_test, predictions)
KNN_F1_Score = f1_score(y_test, predictions)
```

### Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier

Tree = DecisionTreeClassifier()
Tree.fit(x_train, y_train)
predictions = Tree.predict(x_test)

# Calculate metrics
Tree_Accuracy_Score = accuracy_score(y_test, predictions)
Tree_JaccardIndex = jaccard_score(y_test, predictions)
Tree_F1_Score = f1_score(y_test, predictions)
```

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(solver='liblinear')
LR.fit(x_train, y_train)
predictions = LR.predict(x_test)
predict_proba = LR.predict_proba(x_test)

# Calculate metrics
LR_Accuracy_Score = accuracy_score(y_test, predictions)
LR_JaccardIndex = jaccard_score(y_test, predictions)
LR_F1_Score = f1_score(y_test, predictions)
LR_Log_Loss = log_loss(y_test, predict_proba)
```

### Support Vector Machine (SVM)
```python
from sklearn import svm

SVM = svm.SVC()
SVM.fit(x_train, y_train)
predictions = SVM.predict(x_test)

# Calculate metrics
SVM_Accuracy_Score = accuracy_score(y_test, predictions)
SVM_JaccardIndex = jaccard_score(y_test, predictions)
SVM_F1_Score = f1_score(y_test, predictions)
```

## Final Report
```python
Report = pd.DataFrame({
    'Model': ['Linear Regression', 'KNN', 'Decision Tree', 'Logistic Regression', 'SVM'],
    'Accuracy': [None, KNN_Accuracy_Score, Tree_Accuracy_Score, LR_Accuracy_Score, SVM_Accuracy_Score],
    'Jaccard Index': [None, KNN_JaccardIndex, Tree_JaccardIndex, LR_JaccardIndex, SVM_JaccardIndex],
    'F1-Score': [None, KNN_F1_Score, Tree_F1_Score, LR_F1_Score, SVM_F1_Score],
    'Log Loss': [None, None, None, LR_Log_Loss, None]
})
Report
