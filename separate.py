import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)
CSV_DIR = os.path.join(BASE_DIR, "csv_files")

def csv_in(name: str) -> str:
    p = os.path.join(CSV_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"CSV not found: {p}")
    return p

def csv_out(name: str) -> str:
    return os.path.join(CSV_DIR, name)

mix_df = pd.read_csv(csv_in("Mix.csv"))
hs_df = pd.read_csv(csv_in("HS.csv"))
ls_df = pd.read_csv(csv_in("LS.csv"))

all_data = []

def get_label(condition):
    if str(condition).strip().lower() == 'normal':
        return 'Normal'
    return 'Pathology'

for _, row in hs_df.iterrows():
    all_data.append({
        'filename': f"HS/{row['Heart Sound ID']}.wav",
        'label': get_label(row['Heart Sound Type']),
        'source': 'HS',
    })

for _, row in ls_df.iterrows():
    all_data.append({
        'filename': f"LS/{row['Lung Sound ID']}.wav",
        'label': get_label(row['Lung Sound Type']),
        'source': 'LS',
    })

for _, row in mix_df.iterrows():
    all_data.append({
        'filename': f"MIX/{row['Heart Sound ID']}.wav",
        'label': get_label(row['Heart Sound Type']),
        'source': 'MIX',
    })
    all_data.append({
        'filename': f"MIX/{row['Lung Sound ID']}.wav",
        'label': get_label(row['Lung Sound Type']),
        'source': 'MIX',
    })

full_df = pd.DataFrame(all_data).sample(frac=1, random_state=42).reset_index(drop=True)

n = len(full_df)
train_end = int(n * 0.7)
val_end = int(n * 0.8)

train_df = full_df.iloc[:train_end]
val_df = full_df.iloc[train_end:val_end]
test_df = full_df.iloc[val_end:]

train_df.to_csv(csv_out("train_partial.csv"), index=False)
val_df.to_csv(csv_out("val_partial.csv"), index=False)
test_df.to_csv(csv_out("test_partial.csv"), index=False)

print(f"Files created: Train_partial ({len(train_df)}), Va_partiall ({len(val_df)}), Test_partial ({len(test_df)})")
print("\nPreview of train.csv:")
print(train_df[['filename', 'label', 'source']].head())