import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load data
df = pd.read_csv("data/students.csv")
print(" Basic Info:")
print(df.info())

# Summary statistics
print("\n📈 Summary Statistics:")
print(df.describe())

# Histograms for numerical features
df.hist(figsize=(12, 8), edgecolor='black')
plt.tight_layout()
plt.show()

# Boxplots to check outliers
sns.boxplot(data=df[['Age', 'Fare']])
plt.title('Boxplots of Age and Fare')
plt.show()

# Correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot to observe relationships
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']])
plt.show()

# Optional: Interactive plot using Plotly
fig = px.scatter(df, x='Age', y='Fare', color='Survived', title='Age vs Fare by Survival')
fig.show()
