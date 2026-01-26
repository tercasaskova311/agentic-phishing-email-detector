import pandas as pd
from sklearn.utils import resample

# ---------------------------
# LOAD DATA
# ---------------------------
enron = pd.read_csv('data/enron.csv')
legit = pd.read_csv('data/legit.csv')
phishing = pd.read_csv('data/phishing.csv')
trec = pd.read_csv('data/trec.csv')

aigen = pd.concat([legit, phishing], ignore_index=True)
aigen.to_csv('data/aigen.csv', index=False)


print("Columns:")
print("Enron:", enron.columns)
print("TREC:", trec.columns)
print("aigen:", aigen.columns)

# ---------------------------
# HANDLE MISSING VALUES
# ---------------------------
print("\nMissing values:")
print(enron.isna().sum())
print(aigen.isna().sum())
print(trec.isna().sum())

# Drop rows where critical fields are missing
critical_cols = ['message', 'label']  # adjust to your real column names
enron = enron.dropna(subset=critical_cols)
aigen = aigen.dropna(subset=critical_cols)
trec = trec.dropna(subset=critical_cols)

# ---------------------------
# NORMALIZE LABELS
# ---------------------------
enron['label'] = enron['label'].replace({
    'spam': 'phishing_email',
    'ham': 'safe_email'
})

aigen['label'] = aigen['label'].replace({
    1 : 'phishing_email', #llmgenerated
    0 : 'safe_email'
})

trec['label'] = trec['label'].replace({
    1: 'phishing_email',
    0: 'safe_email'
})


info_preprocessing = (
    "ENRON columns:\n" + str(enron.columns.tolist()) + "\n\n" +
    "TREC columns:\n" + str(trec.columns.tolist()) + "\n\n" +
    "AIGEN columns:\n" + str(aigen.columns.tolist()) + "\n\n" +
    "Label counts:\n\n" +
    "Enron:\n" + str(enron['label'].value_counts()) + "\n\n" +
    "Aigen:\n" + str(aigen['label'].value_counts()) + "\n\n" +
    "TREC:\n" + str(trec['label'].value_counts()) + "\n"
)

with open("info_preprocessing.txt", "w") as f:
    f.write(info_preprocessing)