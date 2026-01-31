import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def extract_features(folder, csv_file):
    df = pd.read_csv(csv_file)
    features_list = []
    labels = []
    
    print(f"Ekstrakcja cech z {folder}...")
    for _, row in df.iterrows():
        file_path = os.path.join(folder, row['filename'])
        try:
            y, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            features_list.append(mfccs_scaled)
            labels.append(row['label'])
        except:
            continue
            
    return np.array(features_list), np.array(labels)

X_hs, y_hs = extract_features('Sound/HS_segmented', 'csv_files/HS_segmented.csv')

X_train, X_test, y_train, y_test = train_test_split(X_hs, y_hs, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n--- WYNIKI DLA SERCA (Drzewo Decyzyjne) ---")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Normal', 'Pathology'], yticklabels=['Normal', 'Pathology'])
plt.title('Macierz pomyłek - Serce')
plt.xlabel('Predykcja')
plt.ylabel('Rzeczywistość')
plt.savefig('confusion_matrix_HS.png')
plt.show()

plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=[f'MFCC_{i}' for i in range(20)], class_names=['Normal', 'Pathology'], fontsize=10)
plt.savefig('decision_tree_viz.png')
print("Wykresy zapisane jako pliki .png")

X_ls, y_ls = extract_features('Sound/LS_segmented', 'csv_files/LS_segmented.csv')

X_train_ls, X_test_ls, y_train_ls, y_test_ls = train_test_split(X_ls, y_ls, test_size=0.2, random_state=42)

clf_ls = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_ls.fit(X_train_ls, y_train_ls)

y_pred_ls = clf_ls.predict(X_test_ls)
print("\n--- WYNIKI DLA PŁUC (Drzewo Decyzyjne) ---")
print(classification_report(y_test_ls, y_pred_ls))

plt.figure(figsize=(6,4))
cm_ls = confusion_matrix(y_test_ls, y_pred_ls)
sns.heatmap(cm_ls, annot=True, fmt='d', cmap='Oranges', xticklabels=['Normal', 'Pathology'], yticklabels=['Normal', 'Pathology'])
plt.title('Macierz pomyłek - Płuca')
plt.savefig('confusion_matrix_LS.png')
plt.show()