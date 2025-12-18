import pandas as pd
import os

df = pd.read_csv('datasets/enron.csv')

print(f'Initial shape: {df.shape}')
print(f'Missing values: {df.isna().sum().sum()}')

df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('/', '_')

df['subject'] = df['subject'].fillna('')
df['message'] = df['message'].fillna('')

df = df[df['message'].str.len() > 10]

df['message'] = df['message'].str.strip()
df['subject'] = df['subject'].str.strip()
df['date'] = pd.to_datetime(df['date'], errors='coerce')

df = df.dropna(subset=['message', 'spam_ham', 'date'])

df['label'] = df['spam_ham'].map({'spam': 1, 'ham': 0})
df = df.drop('spam_ham', axis=1)

df = df.reset_index(drop=True)

print(f'Final shape: {df.shape}')
print(f'Remaining missing: {df.isna().sum().sum()}')
print(f'Label distribution: {df["label"].value_counts().to_dict()}')

os.makedirs('datasets/processed', exist_ok=True)
df.to_csv('datasets/processed/enron_clean.csv', index=False)
print('Saved to datasets/processed/enron_clean.csv')
