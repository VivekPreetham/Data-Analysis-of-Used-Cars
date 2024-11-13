# Data-Analysis-of-Used-Cars
The goal of this analysis was to explore and understand a dataset containing customer purchasing behavior, focusing on identifying key trends and patterns.

#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#copy the path of the dataset.
df = pd.read_csv("/content/test.csv")
print(f"The shape is {df.shape}")


#Displaying top 5 samples
df.head()


#Displaying random 5 samples
df.sample(5)


#Checking whether there are any null values in id column
ID = df['id']
df.drop(columns = ['id'], inplace = True)


#Displays information of the dataset
df.info()


#Displays mean, median, SD, etc.
df.describe()


#Checking whether there are any null values in each column of the dataset
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)


#Dividing numerical and categorical values
categorical_columns = [col for col in df.columns if df[col].dtype == 'O']
numerical_columns = [col for col in df.columns if col not in categorical_columns and col != 'price']


#VISUALISATION

#Number of cars by brand represented using a Count Plot.
plt.figure(figsize = [10, 5])
df['brand'].value_counts().plot(kind='bar')
plt.title('Number of Cars by Brand')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()


#Distribution of top 10 brands using Count Plot 
top_10_brands = df['brand'].value_counts().nlargest(10).index
plt.figure(figsize=(12, 6))
sns.countplot(data=df[df['brand'].isin(top_10_brands)], x='brand', order=top_10_brands, palette='viridis')
plt.title('Distribution of Top 10 Brands')
plt.xticks(rotation=45)
plt.show()


#Subplots of histograms of milage and model_year
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
fig.suptitle('Histograms of Mileage, Model Year')
axes[0].hist(df['milage'], bins=20, color='skyblue', edgecolor='black')
axes[0].set_title('Distribution of Mileage')
axes[0].set_xlabel('Mileage')
axes[0].set_ylabel('Frequency')
axes[1].hist(df['model_year'], bins=20, color='skyblue', edgecolor='black')
axes[1].set_title('Distribution of Model Year')
axes[1].set_xlabel('Model Year')
axes[1].set_ylabel('Frequency')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


#Correlation Matrix for Numerical Columns represented using Heatmap
df['engine_numeric'] = df['engine'].str.extract('(\d+\.\d+|\d+)').astype(float)
correlation_matrix = df[['model_year', 'milage', 'engine_numeric']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Numerical Columns')
plt.show()


#Model_year vs Price represented using Scatter Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='model_year', y='milage', data=df)
plt.title('Scatter Plot of Model Year vs Price')
plt.xlabel('Model Year')
plt.ylabel('Price')
plt.show()


#Milage vs model_year represented using Box Plot
plt.figure(figsize= [15, 5])
tmp = df.model_year
sns.boxplot(x=tmp, y=df.milage)
plt.xticks(rotation=45)
plt.title("Milage vs Model Year")
plt.show()


#Distribution of top 10 brands represented using Pie Chart
brand_counts = df['brand'].value_counts().nlargest(10)
plt.figure(figsize=(8, 8))
plt.pie(brand_counts, labels=brand_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Top 10 Car Brands')
plt.axis('equal')
plt.show()


#Number of cars by fuel_tyoe represented using Line Plot
plt.figure(figsize = [10, 5])
df['fuel_type'].value_counts().plot(kind='line')
plt.title('Number of Cars by fuel_type')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.show()
