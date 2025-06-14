# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency, f_oneway

# Load dataset
df = pd.read_csv("HR_Employee_Attrition.csv")

# Descriptive statistics
print("Summary Statistics:")
print(df.describe())

print("\nMode of each column:")
print(df.mode().iloc[0])

print("\nNull values:")
print(df.isnull().sum())

# Boxplot for Age by Attrition
sns.boxplot(x='Attrition', y='Age', data=df)
plt.title("Age vs Attrition")
plt.show()

# Histogram for Monthly Income
plt.hist(df['MonthlyIncome'], bins=10, color='skyblue')
plt.title("Distribution of Monthly Income")
plt.xlabel("Monthly Income")
plt.ylabel("Count")
plt.show()

# Hypothesis 1: t-test between Monthly Income of employees who left vs stayed
left = df[df['Attrition'] == 'Yes']['MonthlyIncome']
stayed = df[df['Attrition'] == 'No']['MonthlyIncome']
t_stat, p_val = ttest_ind(left, stayed)
print(f"\nT-Test: t-statistic = {t_stat:.4f}, p-value = {p_val:.4f}")
print("Conclusion:", "Reject H₀" if p_val < 0.05 else "Fail to Reject H₀")

# Hypothesis 2: Chi-square test between JobRole and Attrition
contingency = pd.crosstab(df['JobRole'], df['Attrition'])
chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-Square Test: chi2 = {chi2:.4f}, p-value = {p:.4f}")
print("Conclusion:", "Reject H₀" if p < 0.05 else "Fail to Reject H₀")

# Hypothesis 3: ANOVA between Monthly Income across Job Roles
groups = [group['MonthlyIncome'].values for name, group in df.groupby("JobRole")]
anova_stat, anova_p = f_oneway(*groups)
print(f"\nANOVA Test: F-statistic = {anova_stat:.4f}, p-value = {anova_p:.4f}")
print("Conclusion:", "Reject H₀" if anova_p < 0.05 else "Fail to Reject H₀")

# Correlation heatmap
corr = df[['Age', 'MonthlyIncome']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Pivot Table
pivot_table = df.pivot_table(values='MonthlyIncome', index='JobRole', columns='Attrition', aggfunc='mean')
print("\nPivot Table (Mean Monthly Income by JobRole and Attrition):")
print(pivot_table)

# Groupby Summary
print("\nGrouped Summary Statistics by JobRole:")
print(df.groupby("JobRole")["MonthlyIncome"].agg(['mean', 'median', 'std', 'min', 'max']))
