import pandas as pd 
import numpy as np

data = pd.read_csv('Task 10/amazon.csv')
print("_Initial Data State_")
print("Total rows and columns:", data.shape)

missing_info = np.sum(pd.isnull(data))
print("\nMissing values per column:\n", missing_info)

most_frequent_val = data['rating'].mode()[0]
data['rating'] = data['rating'].fillna(most_frequent_val)
print("\nMissing values in 'rating' after filling:\n", np.sum(pd.isnull(data['rating'])))

if 'product_id' in data.columns:
    data.drop('product_id', axis=1, inplace= True)

text_features = data.select_dtypes(include=['object']).columns
data[text_features]= data[text_features].apply(lambda col: pd.factorize(col)[0])

print("\n--Data types after cleaning")
print(data.dtypes)
