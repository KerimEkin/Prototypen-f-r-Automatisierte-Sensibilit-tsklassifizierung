import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
from PyPDF2 import PdfReader
from docx import Document

# DocumentDataset Klasse
class DocumentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return {
            **inputs,
            "labels": torch.tensor(label)
        }

# Funktion zum Extrahieren von Texten aus Dateien
def extract_text_from_file(filepath):
    text = ""
    if filepath.endswith('.txt'):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            print(f"UTF-8-Decodierung fehlgeschlagen für {filepath}, versuche es mit ISO-8859-1.")
            try:
                with open(filepath, 'r', encoding='ISO-8859-1') as file:
                    text = file.read()
            except UnicodeDecodeError as e:
                print(f"Fehler beim Lesen von {filepath}: {e}")
    elif filepath.endswith('.pdf'):
        try:
            pdf_reader = PdfReader(filepath)
            if not pdf_reader.pages:
                print(f"Leere PDF-Datei übersprungen: {filepath}")
                return ""
            for page in pdf_reader.pages:
                text += page.extract_text() if page.extract_text() else ''
        except Exception as e:
            print(f"Fehler beim Lesen von {filepath}: {e}")
    elif filepath.endswith('.docx'):
        try:
            doc = Document(filepath)
            for para in doc.paragraphs:
                text += para.text
        except Exception as e:
            print(f"Fehler beim Lesen von {filepath}: {e}")
    else:
        print(f"Nicht unterstützter Dateityp: {filepath}")
    return text

# Funktion zum Laden der Trainingsdaten aus dem Verzeichnis (V:/penv/training)
def load_data_from_directory(directory, label_map):
    data = []
    filenames = []
    labels = []

    for label_name, label_id in label_map.items():
        label_dir = os.path.join(directory, label_name)
        if os.path.exists(label_dir):
            for root, _, files in os.walk(label_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    text = extract_text_from_file(file_path)
                    if text.strip():
                        data.append(text)
                        filenames.append(file_path)
                        labels.append(label_id)
    return data, filenames, labels

# Trainingsdaten laden aus V:/penv/training
training_dir = "V:/penv/training"
label_map = {'red': 0, 'amber': 1, 'green': 2, 'white': 3}  # Mapping für das Training
texts, filenames, labels = load_data_from_directory(training_dir, label_map)

# Überprüfen, ob Daten geladen wurden
if len(texts) == 0 or len(labels) == 0:
    print("Keine Dokumente oder Labels gefunden. Überprüfen Sie das Verzeichnis und die Dateien.")
    exit()

# Dataset erstellen
train_texts, val_texts, train_labels, val_labels, train_filenames, val_filenames = train_test_split(
    texts, labels, filenames, test_size=0.1
)
train_dataset = DocumentDataset(train_texts, train_labels)
val_dataset = DocumentDataset(val_texts, val_labels)

# XLM-RoBERTa Modell laden
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=4)

# Trainingsargumente
training_args = TrainingArguments(
    output_dir="V:/penv/output",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_steps=200,
    logging_dir='./logs',
    logging_steps=200,
    save_total_limit=2,
    seed=42,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=500,
)

# Trainer initialisieren
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Modelltraining starten
trainer.train()

# Modell speichern
model.save_pretrained("V:/penv/output/model")

# Klassifizierung der Dokumente im Verzeichnis V:/penv/sandbox
def classify_documents(directory, model, tokenizer):
    new_texts, filenames, _ = load_data_from_directory(directory, label_map)
    dataset = DocumentDataset(new_texts, [0]*len(new_texts))  # Labels hier nicht relevant
    dataloader = DataLoader(dataset, batch_size=8)

    results = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to('cpu') for key, val in batch[0].items()}
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).tolist()
            results.extend(predictions)

    # Ergebnisse in CSV speichern
    output_file_path = "V:/penv/output/classified_documents.csv"
    df = pd.DataFrame({
        'Document': [os.path.relpath(path, start=directory) for path in filenames],
        'Classification': results
    })
    df.to_csv(output_file_path, index=False)
    print(f"Ergebnisse gespeichert in {output_file_path}")

# Klassifizierung der Dokumente in V:/penv/sandbox
sandbox_dir = "V:/penv/sandbox"
classify_documents(sandbox_dir, model, XLMRobertaTokenizer.from_pretrained('xlm-roberta-base'))
