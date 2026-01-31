import os
import librosa
import soundfile as sf
import pandas as pd

def fix_imbalance_with_windows(csv_path, input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(csv_path)
    
    if 'Heart Sound ID' in df.columns:
        name_col = 'Heart Sound ID'
        label_col = 'Heart Sound Type'
    elif 'Lung Sound ID' in df.columns:
        name_col = 'Lung Sound ID'
        label_col = 'Lung Sound Type'
    else:
        print(f"BŁĄD: Nie znane kolumny w {csv_path}")
        return

    new_segments = []
    win_sec = 2.0  

    print(f"Przetwarzam {len(df)} nagrań z {csv_path}...")

    for _, row in df.iterrows():
        raw_id = str(row[name_col])
        fname = raw_id if raw_id.endswith('.wav') else raw_id + '.wav'
        path = os.path.join(input_dir, fname)
        
        if not os.path.exists(path):
            continue

        try:
            y, sr = librosa.load(path, sr=None)
        except Exception as e:
            continue

        win_len = int(win_sec * sr)
        
        label_val = str(row[label_col])
        is_normal = 'normal' in label_val.lower()
        
        step = int(0.5 * sr) if is_normal else win_len

        count = 0
        for start in range(0, len(y) - win_len + 1, step):
            segment = y[start : start + win_len]
            seg_name = f"{raw_id.replace('.wav', '')}_seg_{count}.wav"
            sf.write(os.path.join(output_dir, seg_name), segment, sr)
            
            new_segments.append({'filename': seg_name, 'label': 0 if is_normal else 1})
            count += 1

    new_df = pd.DataFrame(new_segments)
    output_csv = csv_path.replace(".csv", "_segmented.csv")
    new_df.to_csv(output_csv, index=False)
    
    print(f"Sukces! Stworzono {len(new_df)} fragmentów.")
    print(f"Podział klas (0=Normal, 1=Pathology):")
    print(new_df['label'].value_counts(normalize=True))
    print("-" * 30)

if __name__ == "__main__":
    fix_imbalance_with_windows('csv_files/HS.csv', 'Sound/HS', 'Sound/HS_segmented')
    fix_imbalance_with_windows('csv_files/LS.csv', 'Sound/LS', 'Sound/LS_segmented')