# Titanic Dataset EDA - Python Script Version for VS Code
# Author: Harika Reddi
# Description: Perform Exploratory Data Analysis on Titanic dataset

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: Improve plot styling
sns.set(style="whitegrid")

# Step 2: Load Dataset
df = pd.read_csv('titanic.csv')  # Make sure titanic.csv (train.csv) is in the same folder
print("✅ Dataset Loaded")

# Step 3: Data Cleaning
df.drop_duplicates(inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Cabin'].fillna('Unknown', inplace=True)
df['Survived'] = df['Survived'].astype(int)

# Step 4: Summary Statistics
print("\n🔍 Dataset Info:")
print(df.info())

print("\n📈 Descriptive Statistics:")
print(df.describe())

print("\n🔢 Value Counts:")
print("Sex:\n", df['Sex'].value_counts())
print("\nEmbarked:\n", df['Embarked'].value_counts())
print("\nPclass:\n", df['Pclass'].value_counts())

# Step 5: Visualizations

# Histogram
df.hist(figsize=(12, 8))
plt.suptitle("Histograms of Numeric Columns")
plt.tight_layout()
plt.show()

# Boxplot: Age by Pclass
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title("Age Distribution Across Classes")
plt.show()

# Countplot: Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Passenger Survival Count")
plt.xticks([0, 1], ['Not Survived', 'Survived'])
plt.show()

# Violin Plot: Age vs Survival
sns.violinplot(x='Survived', y='Age', data=df)
plt.title("Age Distribution by Survival")
plt.show()

# Swarm Plot: Age vs Pclass & Survival
sns.swarmplot(x='Pclass', y='Age', hue='Survived', data=df)
plt.title("Age by Class and Survival")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Pair Plot
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived')
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.show()

# Step 6: Groupby Analysis
print("\n🧮 Grouped Insights:")
print("\nSurvival Rate by Sex:\n", df.groupby('Sex')['Survived'].mean())
print("\nSurvival Rate by Class:\n", df.groupby('Pclass')['Survived'].mean())
print("\nEmbarked vs Survival:\n", df.groupby(['Embarked', 'Survived']).size().unstack())

# Step 7: Feature Engineering - Add FamilySize
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Plot: Survival by Family Size
sns.countplot(x='FamilySize', hue='Survived', data=df)
plt.title("Survival by Family Size")
plt.show()

# Step 8: Key Insights Summary
print("\n📝 Top 5 Insights:")
print("1. Women had a much higher survival rate than men (~74% vs ~19%).")
print("2. 1st Class passengers had the highest survival rate.")
print("3. Passengers from Cherbourg (Embarked = 'C') had better chances of survival.")
print("4. Passengers with smaller families (1-3 members) had higher survival rates.")
print("5. Younger passengers and children had better survival chances.")

print("\n🎯 EDA Completed Successfully!")
