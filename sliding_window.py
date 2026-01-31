import os
import librosa
import soundfile as sf
import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_normal_label(label) -> bool:
    return str(label).strip().lower() == "normal"


def _modality_from_row(row: pd.Series, rel_path: str) -> str:
    # 1) Jeśli już jest w CSV, bierz bez kombinowania
    m = str(row.get("modality", "")).strip().upper()
    if m in ("HS", "LS"):
        return m

    # 2) Po source
    source = str(row.get("source", "")).strip().upper()
    if source in ("HS", "LS"):
        return source

    # 3) source=MIX: po ID lub nazwie pliku
    if source == "MIX":
        # w niesegmentowanych CSV masz często "sound id"
        sid = str(row.get("sound id", "")).strip()
        if sid:
            c = sid[0].upper()
            if c == "H":
                return "HS"
            if c == "L":
                return "LS"

        base = os.path.basename(rel_path).strip().lower()  # np. H0085.wav / L0106.wav
        if base.startswith("h"):
            return "HS"
        if base.startswith("l"):
            return "LS"

    return "UNKNOWN"


def segment_from_split_csv(
    csv_path: str,
    audio_root: str = "Sound",
    output_root: str = "Sound_segmented",
    win_sec: float = 2.0,
    normal_step_sec: float = 0.5,
) -> None:
    df = pd.read_csv(csv_path)

    required = {"filename", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing columns: {sorted(missing)}")

    split_name = os.path.splitext(os.path.basename(csv_path))[0]
    new_segments = []

    print(f"[{split_name}] Przetwarzam {len(df)} plików z {csv_path}...")

    for _, row in df.iterrows():
        rel_path = str(row["filename"]).strip().replace("\\", "/")
        label_text = row["label"]
        source = str(row["source"]).strip() if "source" in df.columns else "UNKNOWN"

        modality = _modality_from_row(row, rel_path)

        input_path = os.path.join(audio_root, rel_path)

        # Mała pomoc na wypadek MIX vs Mix
        if not os.path.exists(input_path) and rel_path.startswith("MIX/"):
            input_path_alt = os.path.join(audio_root, "Mix", rel_path[len("MIX/"):])
            if os.path.exists(input_path_alt):
                input_path = input_path_alt

        if not os.path.exists(input_path):
            continue

        try:
            y, sr = librosa.load(input_path, sr=None)
        except Exception:
            continue

        win_len = int(win_sec * sr)
        if win_len <= 0 or len(y) < win_len:
            continue

        is_normal = _is_normal_label(label_text)
        step = int(normal_step_sec * sr) if is_normal else win_len
        if step <= 0:
            step = win_len

        out_dir = os.path.join(output_root, split_name, f"{source}_segmented")
        _ensure_dir(out_dir)

        stem = os.path.splitext(os.path.basename(rel_path))[0]

        count = 0
        for start in range(0, len(y) - win_len + 1, step):
            segment = y[start : start + win_len]
            seg_name = f"{stem}_seg_{count}.wav"
            out_path = os.path.join(out_dir, seg_name)
            sf.write(out_path, segment, sr)

            seg_rel = os.path.join(split_name, f"{source}_segmented", seg_name).replace("\\", "/")
            new_segments.append(
                {
                    "filename": seg_rel,
                    "label": 0 if is_normal else 1,
                    "label_text": "Normal" if is_normal else "Pathology",
                    "source": source,
                    "modality": modality,              # <- NOWE
                    "original_filename": rel_path,
                }
            )
            count += 1

    out_csv = os.path.join(os.path.dirname(csv_path), f"{split_name}_segmented.csv")
    pd.DataFrame(new_segments).to_csv(out_csv, index=False)

    print(f"[{split_name}] Sukces! Stworzono {len(new_segments)} segmentów.")
    if new_segments:
        out_df = pd.DataFrame(new_segments)
        print("Rozkład label:", out_df["label"].value_counts(normalize=True))
        if "modality" in out_df.columns:
            print("Rozkład modality:", out_df["modality"].value_counts(dropna=False))
    print("-" * 30)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        p = os.path.join("csv_files", f"{split}.csv")
        if os.path.exists(p):
            segment_from_split_csv(p)
        else:
            print(f"Pomijam, brak pliku: {p}")