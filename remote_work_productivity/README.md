# Remote Work Productivity Analysis

This folder contains a self-directed project that examines the impact of remote work on employee productivity using an enhanced synthetic dataset. The dataset includes 1000 employee records with additional characteristics such as age, department, weekly hours, productivity score, years at company, education level, gender, job level, satisfaction score, and training hours.

## Project_Overview

**Objective:**
Evaluate whether remote work leads to different productivity scores compared to in-office work, and analyse how factors like age and weekly working hours affect productivity.

**Key Steps:**
- **Data Generation:** Create an enhanced synthetic dataset with 1000 IDs using `generate_dataset.py`.
- **Data Preparation:** Clean the CSV data and create a binary variable for work arrangement.
- **Descriptive Analysis & Visualization:** Summarise and visualise productivity scores by work arrangement.
- **Hypothesis Testing:** Conduct a t-test to compare productivity scores.
- **Regression Analysis:** Use OLS regression (statsmodels) and Linear Regression (scikit-learn) to model productivity.
- **Model Evaluation:** Evaluate models using Mean Squared Error (MSE), R² score, and cross-validation.

## Files in the Repository

- `generate_dataset.py` - Script to generate the enhanced synthetic dataset (`synthetic_remote_work_data.csv`).
- `synthetic_remote_work_data.csv` - The generated dataset with 1000 records.
- `remote_work_productivity.py` - The main analysis script for data cleaning, visualisation, hypothesis testing, and regression modelling.
- `requirements.txt` - A list of required Python packages.
- `README.md` - Project documentation and instructions.
