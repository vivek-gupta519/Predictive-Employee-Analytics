import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score

warnings.filterwarnings('ignore')

# Function to load the data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Display basic data information
def display_data_info(data):
    st.write("### Dataset Overview")
    st.write(data.head())
    st.write("Data Info:")
    st.write(data.info())
    st.write("Data Description:")
    st.write(data.describe())
    st.write("Number of duplicated rows: ", data.duplicated().sum())
    data.drop_duplicates(inplace=True)
    st.write("After dropping duplicates, data length:", len(data))
    st.write("Missing values count per column:")
    st.write(data.isnull().sum())

# Plotting charts using matplotlib and seaborn
def plot_charts(data):
    plt.figure(figsize=(15, 5))
    plt.rc("font", size=14)
    fig, ax = plt.subplots()
    sns.countplot(y='Attrition', data=data, ax=ax)
    st.pyplot(fig)

    plt.figure(figsize=(12, 5))
    fig, ax = plt.subplots()
    sns.countplot(x='Department', hue='Attrition', data=data, palette='hot', ax=ax)
    plt.title("Attrition w.r.t Department")
    st.pyplot(fig)

    plt.figure(figsize=(12, 5))
    fig, ax = plt.subplots()
    sns.countplot(x='EducationField', hue='Attrition', data=data, palette='hot', ax=ax)
    plt.title("Attrition w.r.t EducationField")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    plt.figure(figsize=(12, 5))
    fig, ax = plt.subplots()
    sns.countplot(x='JobRole', hue='Attrition', data=data, palette='hot', ax=ax)
    plt.title("JobRole w.r.t Attrition")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    plt.figure(figsize=(12, 5))
    fig, ax = plt.subplots()
    sns.countplot(x='Gender', hue='Attrition', data=data, palette='hot', ax=ax)
    plt.title("Gender w.r.t Attrition")
    st.pyplot(fig)

    plt.figure(figsize=(12, 5))
    fig, ax = plt.subplots()
    sns.distplot(data['Age'], hist=False, ax=ax)
    st.pyplot(fig)

# Data preprocessing and encoding categorical variables
def process_and_encode_data(data):
    # Target Variable(Attrition)
    data['Attrition'] = data['Attrition'].replace({'No': 0, 'Yes': 1})

    # Encode binary variables
    data['OverTime'] = data['OverTime'].map({'No': 0, 'Yes': 1})
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

    # Encode categorical columns using LabelEncoder
    encoding_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
    label_encoders = {}
    for column in encoding_cols:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    
    return data

# Train and evaluate the logistic regression model
def train_and_evaluate_model(data):
    # Split features and target
    X = data.drop(['Attrition', 'Over18'], axis=1)
    y = data['Attrition'].values
    
    # Handle imbalanced data using RandomOverSampler
    rus = RandomOverSampler(random_state=42)
    X_over, y_over = rus.fit_resample(X, y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)
    
    # Train the logistic regression model
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    # Make predictions and evaluate model
    prediction = logreg.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    st.write(f"Accuracy Score: {accuracy}")

    # Confusion matrix
    cnf_matrix = confusion_matrix(y_test, prediction)
    fig, ax = plt.subplots()
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='Blues', fmt='d', ax=ax)
    st.pyplot(fig)

    # ROC curve and AUC score
    y_pred_proba = logreg.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc}")
    plt.legend(loc=4)
    st.pyplot(fig)
    
    return accuracy
