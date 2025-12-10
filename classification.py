import pandas as pd

column_names = []
with open('census-bureau.columns', 'r') as f:
    for line in f:
        column_names.append(line.strip())

file = pd.read_csv("D:\Work\ML\Census Classification & Segmentation\census-bureau.data", delimiter=",", )
file.columns = column_names

print(file.head(5))