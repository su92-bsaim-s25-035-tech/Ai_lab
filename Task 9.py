import pandas as pd 
data = pd.read_csv('amazon.csv')

print("="*50)
print(" Data Eploration Summary")
print("="*5+"\n")

print("---Displaying Top 5 rows---")
print(data.head())
print("\n")

print("---Displaying Last 5 rows---")
print(data.tail())
print("\n")

rows_count= data.shape[0]
cols_count= data.shape[1]
print("---Dataset Dimensions---")
print("Total Rows:", rows_count)
print("Total Columns:", cols_count)
print("\n")

print("---Null values Analysis---")
null_counts= data.isnull().sum()
print(null_counts)
print("\n")

print("---Column Datatypes---")
print(data.dtypes)
print("\n")

print("="*50)
print("    Exploration Completed")
print("="*50)
