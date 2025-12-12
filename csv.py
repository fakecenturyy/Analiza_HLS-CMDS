import pandas as pd
import os
BASE_PATH=os.getcwd()
train_df = pd.read_csv(os.path.join(BASE_PATH, 'csv_files/train.csv'))
mix_meta_df = pd.read_csv(os.path.join(BASE_PATH, 'csv_files/Mix.csv'))
hs_meta_df = pd.read_csv(os.path.join(BASE_PATH, 'csv_files/Heart.csv'))
ls_meta_df = pd.read_csv(os.path.join(BASE_PATH, 'csv_files/Lung.csv'))
new_col_sound=[]
new_col_location=[]
new_col_gender=[]
for index, row in train_df.iterrows():
    sound='NA'
    gender='NA'
    location='NA'
    type=row['source']
    filename = row['filename']
    clean_id = filename.split('/')[1].replace('.wav', '')
    if type == 'MIX':
        match = mix_meta_df[mix_meta_df['Lung Sound ID'] == clean_id]
        if not match.empty:
            gender=match.iloc[0]['Gender']
            sound=match.iloc[0]['Lung Sound Type']
            location=match.iloc[0]['Location']
        match = mix_meta_df[mix_meta_df['Heart Sound ID'] == clean_id]
        if not match.empty:
            gender=match.iloc[0]['Gender']
            sound=match.iloc[0]['Heart Sound Type']
            location=match.iloc[0]['Location']
    elif type=='HS':
        match = hs_meta_df[hs_meta_df['Heart Sound ID'] == clean_id]
        if not match.empty:
            gender=match.iloc[0]['Gender']
            sound=match.iloc[0]['Heart Sound Type']
            location=match.iloc[0]['Location']
    elif type=='LS':
        match = ls_meta_df[ls_meta_df['Lung Sound ID'] == clean_id]
        if not match.empty:
            gender=match.iloc[0]['Gender']
            sound=match.iloc[0]['Lung Sound Type']
            location=match.iloc[0]['Location']
    new_col_sound.append(sound)
    new_col_location.append(location)
    new_col_gender.append(gender)
train_df['sound'] = new_col_sound
train_df['body_location'] = new_col_location
train_df['patient_gender'] = new_col_gender
save_path = os.path.join(BASE_PATH, 'train_full.csv')  
train_df.to_csv(save_path, index=False)