import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# -----------------------
# 1. Load and Clean the Data
# -----------------------
df = pd.read_csv('synthetic_remote_work_data.csv')

# Clean column names by stripping extra whitespace
df.columns = df.columns.str.strip()
print("Columns:", df.columns.tolist())
print(df.head())

# Create a binary variable for Work_Arrangement: Remote=1, In-Office=0
df['Remote_Binary'] = df['Work_Arrangement'].apply(lambda x: 1 if x.strip().lower() == 'remote' else 0)

# -----------------------
# 2. Descriptive Statistics & Visualization
# -----------------------
desc_stats = df.groupby('Work_Arrangement')['Productivity_Score'].describe()
print("\nDescriptive Statistics for Productivity Score by Work Arrangement:")
print(desc_stats)

# Boxplot of Productivity_Score by Work_Arrangement
plt.figure(figsize=(8, 6))
sns.boxplot(x='Work_Arrangement', y='Productivity_Score', data=df)
plt.title('Productivity Score by Work Arrangement')
plt.savefig('boxplot_productivity.png')
plt.show()

# -----------------------
# 3. Hypothesis Testing (T-Test)
# -----------------------
remote_scores = df[df['Work_Arrangement'].str.strip().str.lower() == 'remote']['Productivity_Score']
office_scores = df[df['Work_Arrangement'].str.strip().str.lower() == 'in-office']['Productivity_Score']

t_stat, p_value = ttest_ind(remote_scores, office_scores)
print("\nT-Test Results:")
print("T-Statistic:", t_stat)
print("P-Value:", p_value)

# -----------------------
# 4. Regression Analysis using statsmodels (OLS)
# -----------------------
# Define predictors and target variable
# Here we focus on key predictors: Remote_Binary, Age, and Weekly_Hours.
X = df[['Remote_Binary', 'Age', 'Weekly_Hours']]
X = sm.add_constant(X)
y = df['Productivity_Score']

ols_model = sm.OLS(y, X).fit()
print("\nOLS Regression Summary (statsmodels):")
print(ols_model.summary())

# -----------------------
# 5. Regression Analysis using scikit-learn
# -----------------------
X_sk = df[['Remote_Binary', 'Age', 'Weekly_Hours']]
y_sk = df['Productivity_Score']

# Split into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_sk, y_sk, test_size=0.2, random_state=42)

# Scale features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = lr_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nScikit-learn Linear Regression Results:")
print("Coefficients:", lr_model.coef_)
print("Intercept:", lr_model.intercept_)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# -----------------------
# 6. Cross-Validation with scikit-learn
# -----------------------
cv_scores = cross_val_score(lr_model, scaler.transform(X_sk), y_sk, cv=5, scoring='r2')
print("\nCross-Validation R^2 Scores:", cv_scores)
print("Average R^2 Score:", np.mean(cv_scores))
