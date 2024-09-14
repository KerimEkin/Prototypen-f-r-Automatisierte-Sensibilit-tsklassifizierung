import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, cohen_kappa_score
import json
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import logging
import fitz  # PyMuPDF für PDF-Extraktion
from docx import Document
import xlrd
import win32com.client
import pytesseract
from PIL import Image
import xml.etree.ElementTree as ET

# Laden der deutschen Stopwörter
nltk.download('stopwords')
german_stopwords = stopwords.words('german')

# Logging einrichten
log_file = "model_training.log"
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Konfiguration aus der Datei laden
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Verzeichnisse und Dateien aus der Konfiguration
documents_dir = config['documents_dir']
output_dir = config['output_dir']
output_file = os.path.join(output_dir, config['output_file'])
output_image = os.path.join(output_dir, config['output_image'])
training_directories = config['training_directories']
TLP_COLORS = config['tlp_colors']

# Funktion zur Berechnung des QCWK
def qcwk_scorer(y_true, y_pred):
    custom_weights = [8, 4, 2, 1]
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

# Funktion zum Extrahieren von Text aus PDF-Dateien
def extract_text_from_pdf(file_path):
    text = ""
    try:
        pdf_document = fitz.open(file_path)
        for page in pdf_document:
            text += page.get_text()
    except Exception as e:
        logging.error(f"Fehler beim Lesen von {file_path}: {e}")
    return text.lower()

# Funktion zum Extrahieren von Text aus DOCX-Dateien
def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text
    except Exception as e:
        logging.error(f"Fehler beim Lesen von {file_path}: {e}")
    return text.lower()

# Funktion zum Extrahieren von Text aus Excel-Dateien
def extract_text_from_xlsx(file_path):
    text = ""
    try:
        workbook = xlrd.open_workbook(file_path)
        sheet = workbook.sheet_by_index(0)
        text = "\n".join(["\t".join(map(str, sheet.row_values(row))) for row in range(sheet.nrows)])
    except Exception as e:
        logging.error(f"Fehler beim Lesen von {file_path}: {e}")
    return text.lower()

# Funktion zum Extrahieren von Text aus MSG-Dateien
def extract_text_from_msg(file_path):
    text = ""
    try:
        outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
        msg = outlook.OpenSharedItem(file_path)
        text = msg.Body
    except Exception as e:
        logging.error(f"Fehler beim Lesen von {file_path}: {e}")
    return text.lower()

# Funktion zum Extrahieren von Text aus Bild-Dateien
def extract_text_from_jpg(file_path):
    text = ""
    try:
        text = pytesseract.image_to_string(Image.open(file_path))
    except Exception as e:
        logging.error(f"Fehler beim Lesen von {file_path}: {e}")
    return text.lower()

# Funktion zum Extrahieren von Text aus Visio-Dateien
def extract_text_from_vsdx(file_path):
    text = ""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        text = " ".join([elem.text for elem in root.iter() if elem.text])
    except Exception as e:
        logging.error(f"Fehler beim Lesen von {file_path}: {e}")
    return text.lower()

# Funktion zum Extrahieren von Text und Metadaten aus allen Dateien in einem Verzeichnis
def extract_documents_text(directory):
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('~$'):
                continue
            path = os.path.join(root, file)
            cleaned_path = clean_path(path)
            text = ""
            if file.endswith('.pdf'):
                text = extract_text_from_pdf(path)
            elif file.endswith('.docx'):
                text = extract_text_from_docx(path)
            elif file.endswith('.txt'):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read().lower()
                except Exception as e:
                    text = ""
            elif file.endswith('.xlsx'):
                text = extract_text_from_xlsx(path)
            elif file.endswith('.msg'):
                text = extract_text_from_msg(path)
            elif file.endswith('.jpg'):
                text = extract_text_from_jpg(path)
            elif file.endswith('.vsdx'):
                text = extract_text_from_vsdx(path)
            
            if text:
                documents.append((cleaned_path, text))
            else:
                logging.warning(f"Could not extract text from: {path}")
    logging.debug(f"{len(documents)} Dokumente extrahiert.")
    return documents

# Funktion zum Extrahieren von Features (TF-IDF) aus den Dokumenten
def extract_features(documents):
    texts = [text for _, text in documents]
    vectorizer = TfidfVectorizer(stop_words=german_stopwords)
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

# Pfad bereinigen (Normalisieren und Kleinbuchstaben)
def clean_path(path):
    return os.path.normpath(path).replace("\\", "/").lower()

# Funktion zum Extrahieren von Trainingsdaten aus den angegebenen Verzeichnissen
def extract_training_data(training_directories):
    training_documents = []
    for label, directory in training_directories.items():
        documents = extract_documents_text(directory)
        for _, text in documents:
            training_documents.append((text, label))
    logging.debug(f"{len(training_documents)} Trainingsdokumente extrahiert.")
    return training_documents

# Sensitivitätsmodell trainieren mit Hyperparameter-Tuning
def train_sensitivity_model_with_tuning(training_documents):
    df = pd.DataFrame(training_documents, columns=['text', 'label'])

    # Features erstellen
    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=german_stopwords)),
        ('clf', LogisticRegression(max_iter=200))
    ])

    # Hyperparameter-Raster für das feinere Tuning
    param_grid = {
        'tfidf__max_df': np.linspace(0.65, 0.75, 6),  # Feinerer Bereich um 0.7, z.B. [0.65, 0.675, 0.7, 0.725, 0.75]
        'tfidf__ngram_range': [(1, 2), (1, 3), (2, 3)],  # Fokus auf Trigramme, zusätzlich (2, 3) für flexible Bigrams+Trigrams
        'clf__C': np.linspace(10, 25, 7)  # Feinerer Bereich um 17.78, z.B. [10, 13.3, 16.7, 20, 23.3, 25]
    }

    logging.debug("Starte GridSearchCV für feineres Hyperparameter-Tuning...")
    # GridSearchCV für das Hyperparameter-Tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=make_scorer(qcwk_scorer), n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    logging.info(f"Best Hyperparameters: {grid_search.best_params_}")
    logging.info(f"Best QCWK: {grid_search.best_score_}")

    return grid_search.best_estimator_

# Dokumente klassifizieren
def classify_documents(model, documents):
    texts = [text for _, text in documents]
    labels = model.predict(texts)
    return labels

# Cluster plotten und speichern
def plot_clusters(X, labels, output_file=output_image):
    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(X.toarray())
    x_axis = scatter_plot_points[:, 0]
    y_axis = scatter_plot_points[:, 1]
    
    fig, ax = plt.subplots(figsize=(20, 10))
    unique_labels = set(labels)
    
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(x_axis[indices], y_axis[indices], label=label, color=TLP_COLORS.get(label, 'black'))
    
    ax.legend()
    plt.savefig(output_file)

# Klassifizierungsergebnisse speichern
def save_classification_results(documents, labels, output_file):
    data = [(path, label) for (path, _), label in zip(documents, labels)]
    df = pd.DataFrame(data, columns=['Document Path', 'Sensitivity Label'])
    df.to_csv(output_file, index=False)

# Hauptfunktion zur Durchführung des gesamten Klassifizierungsprozesses
def main_full():
    logging.debug("Starte Extraktion der Dokumente...")
    documents = extract_documents_text(documents_dir)
    if not documents:
        logging.error("Keine Dokumente extrahiert, Abbruch.")
        return
    
    # Trainingsdaten extrahieren
    logging.debug("Starte Extraktion der Trainingsdaten...")
    training_documents = extract_training_data(training_directories)
    if not training_documents:
        logging.error("Keine Trainingsdokumente extrahiert, Abbruch.")
        return

    # Modell mit Hyperparameter-Tuning trainieren
    logging.debug("Starte Training des Sensitivitätsmodells mit Hyperparameter-Tuning...")
    sensitivity_model = train_sensitivity_model_with_tuning(training_documents)
    if sensitivity_model is None:
        logging.error("Training des Modells fehlgeschlagen, Abbruch.")
        return

    # Dokumente klassifizieren
    logging.debug("Starte Klassifikation der Dokumente...")
    labels = classify_documents(sensitivity_model, documents)

    # Klassifizierungsergebnisse speichern
    logging.debug("Speichere Klassifizierungsergebnisse...")
    save_classification_results(documents, labels, output_file)

    # Features extrahieren und Cluster plotten
    logging.debug("Starte Extraktion der Features und Erstellen des Cluster-Plots...")
    X, _ = extract_features(documents)
    plot_clusters(X, labels)

    logging.debug("Prozess erfolgreich abgeschlossen.")

# Hauptprogramm ausführen
if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main_full()