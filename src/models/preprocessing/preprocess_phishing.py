import pandas as pd
import os

df = pd.read_csv('datasets/phishing.csv')

print(f'Initial shape: {df.shape}')
print(f'Missing values: {df.isna().sum().sum()}')
print(f'Duplicates: {df.duplicated().sum()}')

df = df.drop_duplicates()
print(f'After dropping duplicates: {df.shape}')

df['receiver'] = df['receiver'].fillna('unknown')
df['subject'] = df['subject'].fillna('')

df['body'] = df['body'].str.strip()
df['subject'] = df['subject'].str.strip()
df['date'] = pd.to_datetime(df['date'], errors='coerce')

df = df.dropna(subset=['date', 'body', 'label'])

df = df.reset_index(drop=True)

print(f'Final shape: {df.shape}')
print(f'Remaining missing: {df.isna().sum().sum()}')
print(f'Label distribution: {df["label"].value_counts().to_dict()}')

os.makedirs('datasets/processed', exist_ok=True)
df.to_csv('datasets/processed/phishing_clean.csv', index=False)
print('Saved to datasets/processed/phishing_clean.csv')
