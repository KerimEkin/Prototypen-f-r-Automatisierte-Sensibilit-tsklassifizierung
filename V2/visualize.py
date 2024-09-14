import os
import pandas as pd
from graphviz import Digraph
import json

# Load configuration from file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

documents_dir = config['documents_dir']
output_dir = config['output_dir']
output_file = os.path.join(output_dir, config['output_file'])
directory_structure_pdf = os.path.join(output_dir, config['directory_structure_pdf'])

# Colors for TLP categories
TLP_COLORS = {
    'TLP:RED': 'red',
    'TLP:AMBER': 'orange',
    'TLP:GREEN': 'green',
    'TLP:WHITE': 'white',
    'UNKNOWN': 'black'  # Black for unknown TLP categories
}

# Function to read the classified documents CSV file
def read_classified_documents(csv_file):
    print(f"Reading classified documents from {csv_file}")
    return pd.read_csv(csv_file)

# Function to clean file paths
def clean_path(path):
    return os.path.normpath(path).replace("\\", "/").lower()

# Function to create and visualize the directory tree
def create_directory_tree_and_visualize(classified_docs, output_pdf):
    dot = Digraph(comment='Directory Structure')
    dot.attr(rankdir='TB')

    def add_nodes_edges(dot, parent_path, parent_node_id):
        try:
            for item in os.listdir(parent_path):
                item_path = os.path.join(parent_path, item)
                item_node_id = f"{parent_node_id}_{item}".replace("\\", "/").replace(":", "").replace(".", "")
                cleaned_item_path = clean_path(item_path)
                print(f"Processing item: {item}, path: {cleaned_item_path}")  # Debugging-Ausgabe
                item_class = classified_docs[classified_docs['Document Path'].str.contains(cleaned_item_path, regex=False)]
                if item_class.empty:
                    print(f"No classification found for: {cleaned_item_path}")  # Debugging-Ausgabe
                else:
                    print(f"Classification found for {cleaned_item_path}: {item_class['Keyword Classification'].values[0]}")  # Debugging-Ausgabe
                color = TLP_COLORS['UNKNOWN']
                if not item_class.empty:
                    color = TLP_COLORS.get(item_class['Keyword Classification'].values[0], TLP_COLORS['UNKNOWN'])
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
    print(f"PDF file created: {output_pdf}.pdf")

def main():
    print(f"Current working directory: {os.getcwd()}")
    print(f"Output directory: {output_dir}")
    print(f"CSV file: {output_file}")
    print(f"PDF file: {directory_structure_pdf}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    classified_docs = read_classified_documents(output_file)
    classified_docs['Document Path'] = classified_docs['Document Path'].apply(clean_path)
    create_directory_tree_and_visualize(classified_docs, directory_structure_pdf)

if __name__ == "__main__":
    main()
