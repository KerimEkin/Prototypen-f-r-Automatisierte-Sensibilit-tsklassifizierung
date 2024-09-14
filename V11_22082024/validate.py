import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Pfade anpassen (roh-Strings oder doppelte Backslashes verwenden)
csv_file_path = r"V:\penv\output\classified_documents.csv"  # Pfad zur CSV-Datei mit den Klassifikationsergebnissen
validation_dir = r"V:\penv\validation"  # Verzeichnis mit den manuell klassifizierten Dokumenten

# Laden der klassifizierten Dokumente
classified_documents_df = pd.read_csv(csv_file_path)

# Erstellung einer Ground-Truth basierend auf den Verzeichnissen
ground_truth = []

# Mapping der Ordnernamen zu den Labels
folder_mapping = {"0v_red": "RED", "1v_amber": "AMBER", "2v_green": "GREEN", "3v_white": "WHITE"}

# Erstelle Ground-Truth Liste basierend auf den Dateien in den Verzeichnissen
for folder_name, label in folder_mapping.items():
    folder_path = os.path.join(validation_dir, folder_name)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            document_path = os.path.join("V:/penv/sandbox", filename).lower()
            ground_truth.append({"Document Path": document_path, "True_Classification": label})
    else:
        print(f"Verzeichnis nicht gefunden: {folder_path}")

# Prüfen, ob Ground-Truth-Daten vorhanden sind
if not ground_truth:
    raise ValueError("Keine Ground-Truth-Daten gefunden. Bitte überprüfen Sie die Verzeichnispfade.")

# Erstelle DataFrame für Ground-Truth
ground_truth_df = pd.DataFrame(ground_truth)

# Normalisiere Pfade und Dateinamen für Konsistenz
classified_documents_df['Document Path'] = classified_documents_df['Document Path'].apply(lambda x: os.path.normpath(x).replace("\\", "/").lower())
ground_truth_df['Document Path'] = ground_truth_df['Document Path'].apply(lambda x: os.path.normpath(x).replace("\\", "/").lower())

# Identifiziere fehlende Dokumente
missing_in_csv = set(ground_truth_df['Document Path']) - set(classified_documents_df['Document Path'])
missing_in_validation = set(classified_documents_df['Document Path']) - set(ground_truth_df['Document Path'])

print(f"Fehlende Dokumente in CSV aber in Validierung vorhanden: {len(missing_in_csv)}")
print(f"Fehlende Dokumente in Validierung aber in CSV vorhanden: {len(missing_in_validation)}")

# Verbinde Ground-Truth mit den vorhergesagten Klassifikationen
merged_df = pd.merge(classified_documents_df, ground_truth_df, on="Document Path")

# Extrahiere vorhergesagte und wahre Labels
y_true = merged_df["True_Classification"]
y_pred = merged_df["Sensitivity Label"]

# Berechnung der Metriken
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Logge die Metriken in einer Datei
with open("classification_metrics_log.txt", "w") as log_file:
    log_file.write(f"Accuracy: {accuracy}\n")
    log_file.write(f"Precision: {precision}\n")
    log_file.write(f"Recall: {recall}\n")
    log_file.write(f"F1-Score: {f1}\n")

# Erstelle eine Konfusionsmatrix
conf_matrix = confusion_matrix(y_true, y_pred, labels=list(folder_mapping.values()))

# Plot der Konfusionsmatrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=folder_mapping.values(), yticklabels=folder_mapping.values())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.show()
