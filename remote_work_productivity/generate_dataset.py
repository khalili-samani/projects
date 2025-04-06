import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of records
n = 1000

# Generate synthetic data for basic characteristics
employee_ids = np.arange(1, n + 1)
ages = np.random.randint(22, 66, n)  # Ages between 22 and 65
departments = np.random.choice(['IT', 'Marketing', 'HR', 'Sales', 'Finance'], n)
work_arrangements = np.random.choice(['Remote', 'In-Office'], n)
weekly_hours = np.random.randint(30, 51, n)  # Hours between 30 and 50
productivity_scores = np.round(np.random.uniform(1, 10, n), 1)  # Productivity score between 1 and 10

# Additional characteristics
# Years at company (0 to 20 years)
years_at_company = np.random.randint(0, 21, n)

# Education Level with probabilities: High School (20%), Bachelor (50%), Master (25%), PhD (5%)
education_levels = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.2, 0.5, 0.25, 0.05])

# Gender: Male (48%), Female (48%), Other (4%)
genders = np.random.choice(['Male', 'Female', 'Other'], n, p=[0.48, 0.48, 0.04])

# Job Level: Entry, Mid, Senior, Manager with example probabilities
job_levels = np.random.choice(['Entry', 'Mid', 'Senior', 'Manager'], n, p=[0.4, 0.35, 0.2, 0.05])

# Satisfaction Score: a float between 1 and 10, rounded to 1 decimal
satisfaction_scores = np.round(np.random.uniform(1, 10, n), 1)

# Optionally, add a column representing training hours per year (e.g., 0 to 50 hours)
training_hours = np.random.randint(0, 51, n)

# Create the DataFrame with all characteristics
df = pd.DataFrame({
    'Employee_ID': employee_ids,
    'Age': ages,
    'Department': departments,
    'Work_Arrangement': work_arrangements,
    'Weekly_Hours': weekly_hours,
    'Productivity_Score': productivity_scores,
    'Years_at_Company': years_at_company,
    'Education_Level': education_levels,
    'Gender': genders,
    'Job_Level': job_levels,
    'Satisfaction_Score': satisfaction_scores,
    'Training_Hours': training_hours
})

# Save the DataFrame to a CSV file
df.to_csv('synthetic_remote_work_data.csv', index=False)
print("Synthetic dataset with 1000 IDs and additional characteristics saved as 'synthetic_remote_work_data.csv'.")
