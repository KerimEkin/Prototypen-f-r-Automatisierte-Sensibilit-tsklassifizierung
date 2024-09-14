import os
import csv
import subprocess
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
import joblib
import re
import fitz  # PyMuPDF
from docx import Document
import json

# Load configuration from file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

documents_dir = config['documents_dir']
output_dir = config['output_dir']
output_file = os.path.join(output_dir, config['output_file'])
output_image = os.path.join(output_dir, config['output_image'])
keywords_files = config['keywords_files']

def load_keywords(file_path):
    print(f"Loading keywords from {file_path}")
    with open(file_path, 'r') as file:
        return [line.strip().lower() for line in file.readlines()]

keywords = {label: load_keywords(path) for label, path in keywords_files.items()}

def keyword_score(text):
    scores = defaultdict(int)
    for label, words in keywords.items():
        for word in words:
            scores[label] += text.count(word)
    return scores

def classify_based_on_keywords(text):
    scores = keyword_score(text)
    if not scores:
        return 'TLP:WHITE'
    return max(scores, key=scores.get)

def extract_text_from_pdf(file_path):
    try:
        print(f"Extracting text from PDF {file_path}")
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text.lower()
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path):
    try:
        print(f"Extracting text from DOCX {file_path}")
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.lower()
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return ""

def extract_documents_text(directory):
    print(f"Extracting documents from directory {directory}")
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith('~$'):
                continue
            path = os.path.join(root, file)
            cleaned_path = clean_path(path)
            if file.endswith('.pdf'):
                text = extract_text_from_pdf(path)
                if text:
                    documents.append((cleaned_path, text))
            elif file.endswith('.docx'):
                text = extract_text_from_docx(path)
                if text:
                    documents.append((cleaned_path, text))
            elif file.endswith('.txt'):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read().lower()
                    documents.append((cleaned_path, text))
                except Exception as e:
                    print(f"Error processing file {path}: {e}")
    print(f"Extracted {len(documents)} documents")
    return documents

def clean_path(path):
    return os.path.normpath(path).replace("\\", "/").lower()

def classify_documents(documents):
    print("Classifying documents based on keywords")
    data = []
    for path, text in documents:
        classification = classify_based_on_keywords(text)
        data.append((path, classification, text))
    return data

def extract_features(documents):
    texts = [text for _, _, text in documents]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

def cluster_documents(X, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans

def plot_clusters(X, kmeans, output_file=output_image):
    pca = PCA(n_components=2)
    scatter_plot_points = pca.fit_transform(X.toarray())
    colors = plt.cm.get_cmap('hsv', kmeans.n_clusters)
    x_axis = scatter_plot_points[:, 0]
    y_axis = scatter_plot_points[:, 1]
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.scatter(x_axis, y_axis, c=kmeans.labels_, cmap=colors)
    plt.savefig(output_file)
    print(f"Cluster plot saved as {output_file}")

def save_classification_results(data, output_file):
    print(f"Saving classification results to {output_file}")
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Document Path', 'Keyword Classification', 'Cluster Label'])
            for path, classification, cluster in data:
                writer.writerow([path, classification, cluster])
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

def optimize_kmeans(X):
    param_grid = {'n_clusters': range(2, 10)}
    kmeans = KMeans(random_state=42)
    grid_search = GridSearchCV(kmeans, param_grid, cv=5)
    grid_search.fit(X)
    return grid_search.best_estimator_

# Additional functions for modularity and extensibility
def evaluate_clustering_performance(X, kmeans):
    from sklearn.metrics import silhouette_score
    score = silhouette_score(X, kmeans.labels_)
    print(f"Silhouette Score: {score}")
    return score

def preprocess_text(text):
    text = re.sub(r'\W+', ' ', text)
    return text

def load_documents_from_csv(file_path):
    df = pd.read_csv(file_path)
    return list(zip(df['Document Path'], df['Text']))

def save_model(model, file_path):
    joblib.dump(model, file_path)

def load_model(file_path):
    return joblib.load(file_path)

def compare_classification_methods(documents):
    keyword_classifications = classify_documents(documents)
    X, vectorizer = extract_features(keyword_classifications)
    kmeans = optimize_kmeans(X)
    
    results = []
    for (path, keyword_classification, _), cluster_label in zip(keyword_classifications, kmeans.labels_):
        results.append((path, keyword_classification, cluster_label))
    
    return results

def visualize_comparison(results, output_file='comparison.png'):
    paths, keyword_classifications, cluster_labels = zip(*results)
    df = pd.DataFrame({
        'Document Path': paths,
        'Keyword Classification': keyword_classifications,
        'Cluster Label': cluster_labels
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))
    for label in df['Cluster Label'].unique():
        cluster_data = df[df['Cluster Label'] == label]
        ax.scatter(cluster_data['Keyword Classification'], cluster_data['Cluster Label'], label=f'Cluster {label}')
    
    ax.set_xlabel('Keyword Classification')
    ax.set_ylabel('Cluster Label')
    ax.set_title('Comparison of Classification Methods')
    ax.legend()
    plt.savefig(output_file)
    print(f"Comparison plot saved as {output_file}")

def load_additional_data(file_path):
    return pd.read_csv(file_path)

def combine_data(main_data, additional_data, key='Document Path'):
    return pd.merge(main_data, additional_data, on=key)

# Main program for complete processing and analysis
def main_full():
    print("Starting main_full process")
    documents = extract_documents_text(documents_dir)
    if not documents:
        print("No documents found or extracted.")
        return
    
    keyword_classifications = classify_documents(documents)
    if not keyword_classifications:
        print("No keyword classifications made.")
        return
    
    X, vectorizer = extract_features(keyword_classifications)
    if X.shape[0] == 0:
        print("No features extracted.")
        return
    
    kmeans = optimize_kmeans(X)
    evaluate_clustering_performance(X, kmeans)
    
    data_with_clusters = [(path, classification, cluster) for (path, classification, _), cluster in zip(keyword_classifications, kmeans.labels_)]
    try:
        save_classification_results(data_with_clusters, output_file)
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")
    
    plot_clusters(X, kmeans)
    
    # Trigger visualize.py script
    print("Triggering visualize.py script")
    script_path = 'visualize.py'
    try:
        subprocess.run(['python', script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running visualize.py script: {e}")
    
    print("main_full process completed")

if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    main_full()
