import pandas as pd
import os

BASE_PATH = os.path.dirname(__file__)
CSV_DIR = os.path.join(BASE_PATH, 'csv_files')

train_df = pd.read_csv(os.path.join(CSV_DIR, 'train_partial.csv'))
val_df = pd.read_csv(os.path.join(CSV_DIR, 'val_partial.csv'))
test_df = pd.read_csv(os.path.join(CSV_DIR, 'test_partial.csv'))

mix_meta_df = pd.read_csv(os.path.join(CSV_DIR, 'Mix.csv'))
hs_meta_df = pd.read_csv(os.path.join(CSV_DIR, 'HS.csv'))
ls_meta_df = pd.read_csv(os.path.join(CSV_DIR, 'LS.csv'))

def annotate_df(df: pd.DataFrame) -> pd.DataFrame:
    hs_meta = hs_meta_df.copy()
    ls_meta = ls_meta_df.copy()
    mix_meta = mix_meta_df.copy()
    for col in ("Heart Sound ID", "Lung Sound ID"):
        if col in hs_meta.columns:
            hs_meta[col] = hs_meta[col].astype(str).str.strip()
        if col in ls_meta.columns:
            ls_meta[col] = ls_meta[col].astype(str).str.strip()
        if col in mix_meta.columns:
            mix_meta[col] = mix_meta[col].astype(str).str.strip()

    out_sound, out_loc, out_gender, out_id = [], [], [], []

    for _, row in df.iterrows():
        sound = "NA"
        gender = "NA"
        location = "NA"
        sound_id = "NA"

        type_ = str(row.get("source", "")).strip()
        filename = str(row.get("filename", "")).strip()

        parts = filename.split("/")
        clean_id = (parts[1] if len(parts) > 1 else parts[0]).replace(".wav", "").strip()

        if type_ == "MIX":
            m = mix_meta[mix_meta["Lung Sound ID"] == clean_id]
            if not m.empty:
                gender = str(m.iloc[0]["Gender"]).strip()
                sound = str(m.iloc[0]["Lung Sound Type"]).strip()
                location = str(m.iloc[0]["Location"]).strip()
                sound_id = str(m.iloc[0]["Lung Sound ID"]).strip()

            m = mix_meta[mix_meta["Heart Sound ID"] == clean_id]
            if not m.empty:
                gender = str(m.iloc[0]["Gender"]).strip()
                sound = str(m.iloc[0]["Heart Sound Type"]).strip()
                location = str(m.iloc[0]["Location"]).strip()
                sound_id = str(m.iloc[0]["Heart Sound ID"]).strip()

        elif type_ == "HS":
            m = hs_meta[hs_meta["Heart Sound ID"] == clean_id]
            if not m.empty:
                gender = str(m.iloc[0]["Gender"]).strip()
                sound = str(m.iloc[0]["Heart Sound Type"]).strip()
                location = str(m.iloc[0]["Location"]).strip()
                sound_id = str(m.iloc[0]["Heart Sound ID"]).strip()
            else:
                m = mix_meta[mix_meta["Heart Sound ID"] == clean_id]
                if not m.empty:
                    gender = str(m.iloc[0]["Gender"]).strip()
                    sound = str(m.iloc[0]["Heart Sound Type"]).strip()
                    location = str(m.iloc[0]["Location"]).strip()
                    sound_id = str(m.iloc[0]["Heart Sound ID"]).strip()

        elif type_ == "LS":
            m = ls_meta[ls_meta["Lung Sound ID"] == clean_id]
            if not m.empty:
                gender = str(m.iloc[0]["Gender"]).strip()
                sound = str(m.iloc[0]["Lung Sound Type"]).strip()
                location = str(m.iloc[0]["Location"]).strip()
                sound_id = str(m.iloc[0]["Lung Sound ID"]).strip()
            else:
                m = mix_meta[mix_meta["Lung Sound ID"] == clean_id]
                if not m.empty:
                    gender = str(m.iloc[0]["Gender"]).strip()
                    sound = str(m.iloc[0]["Lung Sound Type"]).strip()
                    location = str(m.iloc[0]["Location"]).strip()
                    sound_id = str(m.iloc[0]["Lung Sound ID"]).strip()

        out_id.append(sound_id)
        out_sound.append(sound)
        out_loc.append(location)
        out_gender.append(gender)

    df = df.copy()
    df["sound"] = out_sound
    df["body_location"] = out_loc
    df["patient_gender"] = out_gender
    df["sound id"] = out_id
    return df

train_df = annotate_df(train_df)
val_df = annotate_df(val_df)
test_df = annotate_df(test_df)

train_df.to_csv(os.path.join(CSV_DIR, 'train.csv'), index=False)
val_df.to_csv(os.path.join(CSV_DIR, 'val.csv'), index=False)
test_df.to_csv(os.path.join(CSV_DIR, 'test.csv'), index=False)