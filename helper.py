import pandas as pd


string = 'Ich heiße Thomas'


# Use a breakpoint in the code line below to debug your script.
file_name = 'Result_6.csv'
# file_name = 'dataNormalized.csv'
# file_name = 'allNormalizedDifference.csv'
print('Reading data from file: ' + file_name)

data = pd.read_csv(file_name)

data['age'] = data['description'].str.findall(r'(\d+)Jahre').str[0].astype(int)
