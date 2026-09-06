# Data-Science-Salary-Prediction

# 📊 Data Science Salary Prediction

An end-to-end Machine Learning project that predicts **Data Science job salaries** based on job, company, location, skills, seniority, and other job-related characteristics.

The project covers the complete Data Science workflow — from **data cleaning and exploratory data analysis to machine learning, model evaluation, and deployment through a Flask API**.

---

## 🚀 Project Overview

Finding the right salary for a Data Science job can be difficult because salaries depend on several factors such as:

* Job title and role
* Location
* Company size
* Company rating
* Industry and sector
* Seniority level
* Required technical skills
* Company revenue
* Job description
* Employment type

This project uses historical job data to understand these relationships and build a Machine Learning model capable of estimating the expected salary for a Data Science position.

### 🎯 Main Question

> **Can we predict the expected salary of a Data Science job using its job and company characteristics?**

The project answers this question by training and comparing multiple regression models and selecting the best-performing model.

---

# 🎯 Objectives

The main objectives of this project are:

1. Collect and prepare Data Science job-related data.
2. Clean and preprocess the dataset.
3. Perform Exploratory Data Analysis (EDA).
4. Identify important factors associated with salaries.
5. Convert categorical variables into numerical features.
6. Train multiple Machine Learning regression models.
7. Compare model performance using Mean Absolute Error (MAE).
8. Optimize the Random Forest model using GridSearchCV.
9. Select the best-performing model.
10. Save the trained model for future predictions.
11. Develop a Flask API to make the model accessible as an application.
12. Demonstrate how Machine Learning can be used for practical salary estimation.

---

# 🔄 Project Workflow

```text
Raw Job Data
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Categorical Encoding
     ↓
Train-Test Split
     ↓
Machine Learning Models
     ↓
Cross-Validation
     ↓
Hyperparameter Tuning
     ↓
Model Comparison
     ↓
Best Model Selection
     ↓
Model Serialization
     ↓
Flask API
     ↓
Salary Prediction
```

---

# 📁 Project Structure

```text
data-science-salary-prediction/
│
├── data/
│      
│
├── scripts/
│   ├── data_cleaning.py
│   ├── data_eda.py
│   └── modelbuilding.py
│
├── models/
│   └── model_file.p
│
├── Flaskapi/
│   ├── models/
│   ├── app.py
│   ├── data_input.py
│   ├── Procfile
│   ├── requirements.txt
│   └── wsgi.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

# 🧹 1. Data Cleaning

The first stage of the project prepares the raw job data for analysis and Machine Learning.

The cleaning process includes handling and preparing information related to:

* Salary
* Company rating
* Company size
* Ownership
* Industry
* Sector
* Revenue
* Job location
* Company age
* Technical skills
* Job role
* Seniority
* Job description

The cleaned data is then used for Exploratory Data Analysis and model development.

---

# 📊 2. Exploratory Data Analysis

Exploratory Data Analysis is performed to understand the dataset before building Machine Learning models.

The analysis focuses on questions such as:

* What salary ranges exist in the dataset?
* Which job roles have higher salaries?
* Does location affect salary?
* Does seniority affect salary?
* How does company rating relate to salary?
* Which technical skills are associated with Data Science jobs?
* How do industry and sector affect salary?
* Are there relationships between company characteristics and salary?

EDA helps us understand the data and decide which features can be useful for Machine Learning.

---

# ⚙️ 3. Feature Engineering

The project uses several job and company characteristics as input features.

The model uses features including:

```text
Rating
Size
Type of ownership
Industry
Sector
Revenue
Number of competitors
Hourly
Employer provided
Job state
Same state
Company age
Python
AWS
Spark
Excel
Job simplified role
Seniority
Job description length
```

The target variable is:

```text
avg_salary
```

The categorical variables are converted into numerical variables using **One-Hot Encoding** through pandas `get_dummies()`.

This allows Machine Learning algorithms to process categorical information numerically.

---

# 🧠 4. Machine Learning Models

Three main regression approaches were evaluated:

### 1. Linear Regression

Linear Regression provides a baseline model and attempts to model salary using linear relationships between the features and target.

### 2. Lasso Regression

Lasso Regression adds regularization and can help reduce the influence of less useful features.

The project also tests different alpha values to identify a better-performing Lasso configuration.

### 3. Random Forest Regression

Random Forest is an ensemble Machine Learning algorithm that combines multiple decision trees.

It is particularly useful when relationships between features and salary are not simply linear.

---

# 🔍 5. Model Validation

The project uses:

```text
Train/Test Split
Test Size = 20%
Random State = 42
```

Three-fold cross-validation is also used during model evaluation.

The main evaluation metric used in the current implementation is:

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between the actual salary and predicted salary.

In simple terms:

> **Lower MAE = better predictions**

For example, an MAE of 12 means the model's predictions are, on average, about 12 salary units away from the actual values.

---

# 📈 6. Model Performance

The following results were obtained from the current implementation.

## Cross-Validation Results

| Model             | 3-Fold CV MAE |
| ----------------- | ------------: |
| Linear Regression |     **20.77** |
| Lasso Regression  |     **19.27** |
| Random Forest     |     **15.07** |

### What does this tell us?

Random Forest achieved the lowest cross-validation MAE among these models.

Compared with Linear Regression:

**20.77 → 15.07**

This represents approximately a:

### **27.4% reduction in MAE**

This indicates that Random Forest captured the relationships in the dataset better than the linear baseline during cross-validation.

---

# 🔧 7. Hyperparameter Tuning

GridSearchCV was used to optimize the Random Forest model.

The search evaluated different combinations of:

* Number of estimators
* Criterion
* Maximum features

The search used:

```text
Cross-validation: 3 folds
Scoring: Negative Mean Absolute Error
```

The best configuration found was:

```text
RandomForestRegressor(
    n_estimators=160,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
```

The best GridSearchCV score corresponded to an MAE of approximately:

### **15.30**

---

# 🏆 8. Final Test Results

After training and evaluating the models, the following test-set MAEs were obtained:

| Model                           |  Test MAE |
| ------------------------------- | --------: |
| Linear Regression               | **18.85** |
| Lasso Regression                | **19.75** |
| Random Forest                   | **12.09** |
| Linear + Random Forest Ensemble | **14.33** |

## 🥇 Best Model: Random Forest

The Random Forest model achieved the lowest test MAE:

# **12.09**

This makes Random Forest the best-performing model among the models tested in the current implementation.

---

# 📊 Key Achievement

Random Forest achieved:

```text
Linear Regression MAE = 18.85
Random Forest MAE     = 12.09
```

This means the Random Forest model reduced the test-set MAE by approximately:

# **35.8%**

compared with Linear Regression.

### In simple terms:

> **The Random Forest model made substantially smaller average salary prediction errors than the Linear Regression baseline.**

---

# 🧪 Example Prediction

The trained model was saved and loaded again to test its prediction capability.

For one example record from the dataset, the model produced:

```text
Predicted salary = 51.27
```

This is an example prediction for one job record.

**Important:** 51.27 is a predicted salary value, **not an accuracy score or model performance metric**.

The exact monetary interpretation depends on the unit used in the original `avg_salary` column.

---

# 📌 What Have We Actually Achieved?

In simple terms, this project has achieved the following:

### ✅ 1. Built a complete salary prediction pipeline

We went from raw job data to a trained Machine Learning model.

### ✅ 2. Used real-world job-related features

Instead of predicting salary using only one or two variables, the model considers multiple characteristics such as:

* Location
* Seniority
* Company information
* Industry
* Revenue
* Technical skills
* Job role
* Job description

### ✅ 3. Compared multiple Machine Learning algorithms

We tested:

* Linear Regression
* Lasso Regression
* Random Forest Regression
* Linear + Random Forest Ensemble

### ✅ 4. Identified the best model

Random Forest achieved the best test MAE:

**12.09**

### ✅ 5. Improved prediction error

Random Forest reduced test MAE by approximately:

**35.8% compared with Linear Regression**

### ✅ 6. Performed hyperparameter tuning

GridSearchCV was used to find a better Random Forest configuration.

### ✅ 7. Saved the trained model

The final model was serialized using Python pickle so it can be reused without retraining every time.

### ✅ 8. Prepared the project for deployment

A Flask API was developed so that the trained Machine Learning model can be used through an application/API.

---

# 🌐 9. Flask API

The project includes a Flask-based API for serving the trained Machine Learning model.

The API allows users to provide job-related information and receive a predicted salary.

Conceptually:

```text
User Input
    ↓
Flask API
    ↓
Preprocessing
    ↓
Trained Random Forest Model
    ↓
Predicted Salary
```

This converts the Machine Learning model from a research experiment into something that can be used as an application.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Random Forest
* Linear Regression
* Lasso Regression
* GridSearchCV
* Cross-Validation

### Statistical Analysis

* Statsmodels

### Deployment

* Flask
* WSGI

### Model Serialization

* Pickle

### Development & Version Control

* Git
* GitHub

---

# 📚 Important Python Libraries

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
statsmodels
flask
```

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

# 📊 Current Results Summary

| Metric / Result                         |                Value |
| --------------------------------------- | -------------------: |
| Dataset observations used in OLS output |              **742** |
| Linear Regression CV MAE                |            **20.77** |
| Lasso CV MAE                            |            **19.27** |
| Random Forest CV MAE                    |            **15.07** |
| GridSearch best CV MAE                  |            **15.30** |
| Linear Regression Test MAE              |            **18.85** |
| Lasso Test MAE                          |            **19.75** |
| **Random Forest Test MAE**              |            **12.09** |
| Ensemble Test MAE                       |            **14.33** |
| RF improvement vs Linear Regression     | **~35.8% lower MAE** |
| Best RF estimators                      |              **160** |
| Best RF max_features                    |             **sqrt** |
| Example prediction                      |            **51.27** |

---

# ⚠️ Important Evaluation Note

The current project also contains an OLS statistical analysis with:

```text
R² = 0.708
Adjusted R² = 0.638
```

However, this OLS result should **not be directly compared with the Random Forest test MAE**, because the current OLS implementation evaluates the regression differently from the train/test evaluation used for the Machine Learning models.

For a future improved version, all models should be evaluated on the same held-out test set using consistent metrics such as:

* MAE
* RMSE
* R²

This will provide a stronger and more academically rigorous comparison.

---


# 💼 Real-World Applications

This type of system could be useful for:

### 👨‍💻 Job Seekers

Estimate a reasonable salary range based on job characteristics.

### 🏢 Recruiters

Understand salary patterns across roles and locations.

### 📊 HR Teams

Analyze compensation trends.

### 🎓 Students

Understand how skills, seniority, location, and company characteristics can relate to Data Science salaries.

### 📈 Career Planning

Identify factors associated with higher-paying Data Science positions.

---

# 📖 Learning Outcomes

Through this project, the following practical skills were developed:

* Data cleaning
* Exploratory Data Analysis
* Feature engineering
* Categorical encoding
* Regression
* Train-test splitting
* Cross-validation
* Hyperparameter tuning
* Model comparison
* Model evaluation
* Model serialization
* Flask API development
* Git/GitHub project organization
* End-to-end Machine Learning workflow

---

# 👨‍💻 Project Highlights

### 🔹 End-to-End Machine Learning

Developed a complete pipeline from raw job data to salary prediction and deployment.

### 🔹 Multiple Model Comparison

Compared Linear Regression, Lasso, Random Forest, and an ensemble approach.

### 🔹 Hyperparameter Optimization

Used GridSearchCV to optimize Random Forest parameters.

### 🔹 Best Test Performance

Random Forest achieved a test MAE of **12.09**.

### 🔹 35.8% Lower Error

Random Forest reduced test MAE by approximately **35.8% compared with Linear Regression**.

### 🔹 Deployment Ready

Integrated the trained model with a Flask API for practical use.

---

# 🏁 Conclusion

This project demonstrates how Machine Learning can be applied to a real-world problem of **Data Science salary prediction**.

By combining job characteristics, company information, technical skills, location, seniority, and other features, the project builds models capable of estimating expected salaries.

After comparing multiple regression techniques, **Random Forest performed the best on the current test set, achieving an MAE of 12.09**.

The project also demonstrates the complete Machine Learning lifecycle:

```text
Data
 ↓
Cleaning
 ↓
EDA
 ↓
Feature Engineering
 ↓
Model Building
 ↓
Evaluation
 ↓
Optimization
 ↓
Best Model
 ↓
Deployment
```

# 📜 Disclaimer

This project is intended for **educational, research, and demonstration purposes**.

Salary predictions are estimates and should not be considered guaranteed compensation or professional salary advice.

If the underlying job data was obtained from a third-party website, users should verify the applicable terms, licensing, and redistribution permissions before publishing or redistributing the dataset.

---

## 👨‍💻 Author

**Raj Patil**

Data Science | Machine Learning | Python | SQL | Power BI

---

⭐ **If you find this project useful, consider giving the repository a star!**
