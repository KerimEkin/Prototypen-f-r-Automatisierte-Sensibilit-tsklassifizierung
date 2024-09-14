import os
import pandas as pd
from graphviz import Digraph
import json

# Konfiguration aus der Datei laden
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

documents_dir = config['documents_dir']
output_dir = config['output_dir']
output_file = os.path.join(output_dir, config['output_file'])
directory_structure_pdf = os.path.join(output_dir, config['directory_structure_pdf'])
TLP_COLORS = config['tlp_colors']

# Klassifizierte Dokumente aus CSV-Datei lesen
def read_classified_documents(csv_file):
    return pd.read_csv(csv_file)

# Pfad bereinigen (Normalisieren und Kleinbuchstaben)
def clean_path(path):
    return os.path.normpath(path).replace("\\", "/").lower()

# Verzeichnisstruktur erstellen und visualisieren
def create_directory_tree_and_visualize(classified_docs, output_pdf):
    dot = Digraph(comment='Directory Structure')
    dot.attr(rankdir='LR')  # Ändere die Richtung des Baums zu Top-to-Bottom

    def add_nodes_edges(dot, parent_path, parent_node_id):
        try:
            for item in os.listdir(parent_path):
                item_path = os.path.join(parent_path, item)
                item_node_id = f"{parent_node_id}_{item}".replace("\\", "/").replace(":", "").replace(".", "")
                cleaned_item_path = clean_path(item_path)
                item_class = classified_docs[classified_docs['Document Path'].str.contains(cleaned_item_path, regex=False)]
                if item_class.empty:
                    color = TLP_COLORS['UNKNOWN']
                else:
                    sensitivity_label = item_class['Sensitivity Label'].values[0]
                    color = TLP_COLORS.get(sensitivity_label, TLP_COLORS['UNKNOWN'])
                dot.node(item_node_id, label=item, color=color, fontcolor=color)
                dot.edge(parent_node_id, item_node_id)

                if os.path.isdir(item_path):
                    add_nodes_edges(dot, item_path, item_node_id)
        except PermissionError:
            pass

    root_node_id = "Root"
    dot.node(root_node_id, label=os.path.basename(documents_dir) if os.path.basename(documents_dir) else "Root")

    add_nodes_edges(dot, documents_dir, root_node_id)
    dot.render(output_pdf, view=False, format='pdf')

def main():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    classified_docs = read_classified_documents(output_file)
    classified_docs['Document Path'] = classified_docs['Document Path'].apply(clean_path)
    create_directory_tree_and_visualize(classified_docs, directory_structure_pdf)

if __name__ == "__main__":
    main()
