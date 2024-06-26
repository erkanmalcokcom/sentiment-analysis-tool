import pandas as pd

data = pd.read_csv('data/raw/IMDB_Dataset.csv')
import os

print(os.system("clear"))
print("*" * 40)
# print(os.system('ls -lh data/IMDB Dataset.csv'))
# print(os.path.getsize('data/IMDB Dataset.csv'))



# Display the first few rows of the dataset
print(data.describe())

# Using shape
num_rows = data.shape[0]
print(f'Number of rows: {num_rows}')


# Reduce the size of the dataset
data = data.iloc[:30000]
data.to_csv('data/processed/IMDB_Dataset_reduced.csv', index=False)

# Display the first few rows of the reduced dataset
print("*" * 40)
print(data.describe())
num_rows = len(data)
print(f'Number of rows: {num_rows}')
# Using shape
num_rows = data.shape[0]
print(f'Number of rows: {num_rows}')
# Output:
#              review sentiment
# count        50000     50000
# unique       49582         2
# top     Great film  positive
