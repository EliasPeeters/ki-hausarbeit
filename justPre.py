import mainNewOhneText

data = mainNewOhneText.read_data('Result_27.csv')
data = data.dropna()

price_raw, features_raw = mainNewOhneText.storePriceSeperate(data)
print('store price seperate')

features_raw = mainNewOhneText.preProcessData(features_raw)
print('pre process data')

features_raw = mainNewOhneText.removeUnusedColumns(features_raw)
print('remove unused columns')

# save features_raw to csv
features_raw.to_csv('preProcessed/features_raw.csv', index=False)

# save price_raw to csv
price_raw.to_csv('preProcessed/price_raw.csv', index=False)

print('saved')