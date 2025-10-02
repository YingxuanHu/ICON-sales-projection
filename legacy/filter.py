import pandas as pd

# Reading the CSV files
transcation_df = pd.read_csv('/Users/alvinhu/Desktop/ICON/Purchaser/transaction.csv', low_memory=False)
name_df = pd.read_csv('/Users/alvinhu/Desktop/ICON/Purchaser/names.csv', encoding='ISO-8859-1', low_memory=False)

# Filtering based on the keyword
keyword = 'Office Furniture'
transcation_df = transcation_df[transcation_df['description_en'].str.contains(keyword, case=False, na=False)]

# Removing NaNs and getting unique buyer names
transcation_df = transcation_df[transcation_df['buyer_name'].notna()]
transcation_df = transcation_df[transcation_df['original_value'] <= 25000]

name_df['full_name'] = name_df['Surname'].str.strip() + ' ' + name_df['GivenName'].str.strip()
pd.DataFrame(name_df).to_csv('name_df.csv', index=False)

transcation_df['buyer_name'] = transcation_df['buyer_name'].str.replace(',', '')
transcation_df = transcation_df['buyer_name'].unique()
pd.DataFrame(transcation_df).to_csv('transaction_df.csv', index=False)

matched_names_df = name_df[name_df['full_name'].isin(transcation_df)]
# Exporting to CSV
pd.DataFrame(matched_names_df).to_csv('ContactInfo.csv', index=False)
