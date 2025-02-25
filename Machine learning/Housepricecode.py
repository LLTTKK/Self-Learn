# Import necessary libraries for data manipulation, visualization, and machine learning
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
from sklearn.preprocessing import OneHotEncoder

# Function to plot the number of buildings built in different years
def YearBuilt_plot():
    if 'YearBuilt' in dataset.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=dataset, x='YearBuilt', palette='viridis')
        plt.title('Number of Buildings Built in Different Years')
        plt.xlabel('Year Built')
        plt.ylabel('Number of Buildings')
        plt.xticks(rotation=90)
        plt.show()
    else:
        print("The dataset does not contain a 'YearBuilt' column.")

# Function to plot the number of locations in different zones
def Zonecount():
    if 'MSZoning' in dataset.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=dataset, x='MSZoning', palette='viridis')
        plt.title('Number of location in Different Zones')
        plt.xlabel('Zone')
        plt.ylabel('Number of locations')
        plt.show()
    else:
        print("The dataset does not contain a 'MSZoning' column.")

# Load the dataset from an Excel file and handle the case where the file is not found
try:
    dataset = pd.read_excel("HousePricePrediction.xlsx")
    print(dataset.head(5))
except FileNotFoundError:
    print("Error: The file 'HousePricePrediction.xlsx' was not found.")

# Display the dimensions of the dataset
dataset.shape

# Identify columns by data type: object, integer, and float
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
fl = (dataset.dtypes == 'float')
fl_cols = list(fl[fl].index)

# Function to print the number of categorical, integer, and float variables in the dataset
def Dataval():
    obj = (dataset.dtypes == 'object')
    object_cols = list(obj[obj].index)
    print("Categorical variables:", len(object_cols))

    int_ = (dataset.dtypes == 'int')
    num_cols = list(int_[int_].index)
    print("Integer variables:", len(num_cols))

    fl = (dataset.dtypes == 'float')
    fl_cols = list(fl[fl].index)
    print("Float variables:", len(fl_cols))

# Function to plot the number of subclasses in different classes
def MSSubClass():
    if 'MSSubClass' in dataset.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=dataset, x='MSSubClass', palette='viridis')
        plt.title('Number of SubClass in Different Classes')
        plt.xlabel('SubClass')
        plt.ylabel('Number of SubClass')
        plt.show()

# Function to plot the number of unique values for each categorical feature
def different_categorical_features():
    unique_values = []
    for col in object_cols:
        unique_values.append(dataset[col].unique().size)
    plt.figure(figsize=(10, 6))
    plt.title('No. Unique values of Categorical Features')
    plt.xticks(rotation=90)
    sns.barplot(x=object_cols, y=unique_values)
    plt.show()

# Data cleaning: drop the 'Id' column and fill missing 'SalePrice' values with the mean
dataset.drop(['Id'], axis=1, inplace=True)
dataset['SalePrice'] = dataset['SalePrice'].fillna(dataset['SalePrice'].mean())

# Remove rows with any remaining missing values
new_dataset = dataset.dropna()
new_dataset.isnull().sum()

# One-hot encode categorical variables
OH_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_dataset[object_cols]))
OH_cols.index = new_dataset.index
OH_cols.columns = OH_encoder.get_feature_names_out()
df_final = new_dataset.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)

# Import necessary libraries for machine learning models
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_percentage_error

# Function to train and test a Support Vector Regression model
def ASVRTrainTestSplit():
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    X = df_final.drop(['SalePrice'], axis=1)
    Y = df_final['SalePrice']

    # Split the data into training and validation sets
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X, Y, train_size=0.8, test_size=0.2, random_state=0)

    # Train the SVR model and predict on the validation set
    model_SVR = svm.SVR()
    model_SVR.fit(X_train, Y_train)
    Y_pred = model_SVR.predict(X_valid)

    # Print the mean absolute percentage error of the predictions
    print(mean_absolute_percentage_error(Y_valid, Y_pred))

# Function to train and test a Random Forest Regressor model
def RandomForestTrainTestSplit():    # This model predicts the 'SalePrice' of houses based on various features.
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    X = df_final.drop(['SalePrice'], axis=1)
    Y = df_final['SalePrice']

    # Split the data into training and validation sets
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X, Y, train_size=0.8, test_size=0.2, random_state=0)

    from sklearn.ensemble import RandomForestRegressor

    # Train the Random Forest model and predict on the validation set
    model_RFR = RandomForestRegressor(n_estimators=10)
    model_RFR.fit(X_train, Y_train)
    Y_pred = model_RFR.predict(X_valid)

    # Print the mean absolute percentage error of the predictions
    print(mean_absolute_percentage_error(Y_valid, Y_pred))

# Execute the functions to train and test the models
ASVRTrainTestSplit()
RandomForestTrainTestSplit()



