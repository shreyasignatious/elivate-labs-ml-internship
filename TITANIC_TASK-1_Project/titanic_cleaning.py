import pandas as pd
from sklearn.preprocessing import StandardScaler

#Load the data 
df = pd.read_csv("data/train.csv")
print("First 5 rows:\n", df.head())

# missing values and data types
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)

# Encode variables
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Scale numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Remove outliers
df = df[df['Fare'] < df['Fare'].quantile(0.99)]

# Save the cleaned data
df.to_csv("data/cleaned_titanic.csv", index=False)
print("Data cleaned and saved to data/cleaned_titanic.csv")
