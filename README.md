# Task-2

🧠 Titanic Dataset - Exploratory Data Analysis (EDA)
This project performs an in-depth exploratory data analysis (EDA) on the Titanic dataset.
The objective is to explore the dataset, understand the features, and uncover meaningful insights through visualizations and summary statistics.

📌 Objective

- Explore feature distributions and relationships
- Detect missing values and outliers
- Understand the relationship between features and survival
- Use visualizations to support pattern recognition

🛠️ Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
📁 Dataset
The dataset used is the Titanic Dataset, which includes details such as passenger class, age, fare, gender, family size, and survival status.

📜 Python Code (with Description)

# Step 1: Import libraries and set a visual style
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

# Step 2: Load the dataset with proper handling for missing values
df = pd.read_csv("Titanic-Dataset.csv", na_values=["", " ", "?", "NA", "n/a", "N/A"])
print("Dataset Shape:", df.shape)
print(df.head())

# Step 3: Display statistical summary
print(df.describe())
print(df.describe(include='all'))

# Step 4: Handle missing values (Age and Fare) for plotting
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Step 5: Plot histograms of numeric columns
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 2, i+1)
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# Step 6: Plot boxplots to detect outliers
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=df[col], color='salmon')
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# Step 7: Pairplot to visualize relationships and survival
sns.pairplot(df[['Age', 'Fare', 'SibSp', 'Parch', 'Survived']].dropna(), hue='Survived', palette='Set2')
plt.suptitle("Pairplot of Numeric Features Colored by Survival", y=1.02)
plt.show()

# Step 8: Correlation matrix heatmap
plt.figure(figsize=(10, 6))
corr_matrix = df[['Age', 'Fare', 'SibSp', 'Parch', 'Survived']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Step 9: Count plots for categorical vs survival
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Count by Gender")
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival Count by Passenger Class")
plt.show()

# Step 10: KDE plot of Age by survival
sns.kdeplot(df[df['Survived'] == 1]['Age'], label='Survived', fill=True)
sns.kdeplot(df[df['Survived'] == 0]['Age'], label='Not Survived', fill=True)
plt.title("Age Distribution by Survival Status")
plt.legend()
plt.show()

🧠 Key Insights

- Most survivors were females and from 1st class.
- Children had a slightly better chance of survival.
- Passengers who paid higher fares were more likely to survive.
- There are outliers in Fare and Age that might affect modeling.

📬 Contact
For help or collaboration, feel free to reach out.
