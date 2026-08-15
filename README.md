# Linux Server Response Time Prediction

## Project Overview

Linux Server Response Time Prediction is a Machine Learning project developed using Python and Random Forest Regressor. The main purpose of this project is to predict the response time of a Linux server based on different server performance parameters. The system uses parameters such as CPU Usage, RAM Usage, Disk Usage, Network Traffic, Active Users, Running Processes, Server Uptime, and Error Count to predict the expected server response time. This project helps in understanding server performance and can support server monitoring and performance management.

## Problem Statement

Monitoring Linux server performance manually can be time-consuming and difficult when the number of servers and performance parameters increases. High CPU usage, RAM usage, disk usage, network traffic, active users, and error count can affect server performance and response time. Therefore, a Machine Learning model is developed to predict the response time of a Linux server using historical server performance data.

## Objectives

- To collect Linux server performance data.
- To create and preprocess the dataset.
- To clean messy data by handling missing values, duplicate records, and incorrect data formats.
- To identify important server performance parameters.
- To train a Random Forest Regressor model.
- To predict Linux server response time.
- To evaluate the model using MAE, MSE, RMSE, and R² Score.
- To develop a user-friendly Streamlit web application for prediction.

## Dataset

The dataset contains 150+ Linux server records. It includes different server performance parameters that influence server response time.

### Dataset Attributes

- Server_ID
- CPU_Usage
- RAM_Usage
- Disk_Usage
- Network_Traffic
- Active_Users
- Running_Processes
- Server_Uptime
- Response_Time
- Error_Count
- Server_Status

### Target Variable

`Response_Time`

The target variable represents the predicted response time of the Linux server.

## Data Cleaning

The original dataset contains messy data that requires preprocessing before applying Machine Learning. The following cleaning operations are performed:

- Removed duplicate records.
- Identified missing values.
- Filled missing numeric values using median values.
- Removed percentage (%) symbols from CPU Usage and Disk Usage.
- Converted numeric columns into proper numeric data types.
- Standardized inconsistent Server Status values.
- Prepared the cleaned dataset for Machine Learning.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regressor
- Streamlit
- Google Colab
- Visual Studio Code
- Matplotlib

## Machine Learning Algorithm

### Random Forest Regressor

Random Forest Regressor is an ensemble Machine Learning algorithm that combines multiple decision trees to produce accurate numerical predictions. In this project, Random Forest Regressor is used to predict Linux server response time based on multiple server performance parameters. The algorithm can handle complex relationships between the input features and the target variable.

## Features Used for Prediction

The following features are used as inputs to the Machine Learning model:

- CPU Usage
- RAM Usage
- Disk Usage
- Network Traffic
- Active Users
- Running Processes
- Server Uptime
- Error Count

The model predicts:

`Response Time`

## Project Workflow

```text
Messy Dataset
      ↓
Data Cleaning
      ↓
Missing Value Handling
      ↓
Duplicate Removal
      ↓
Data Preprocessing
      ↓
Feature Selection
      ↓
Train/Test Data
      ↓
Random Forest Regressor
      ↓
Model Training
      ↓
Response Time Prediction
      ↓
Model Evaluation
      ↓
Streamlit Web Application
