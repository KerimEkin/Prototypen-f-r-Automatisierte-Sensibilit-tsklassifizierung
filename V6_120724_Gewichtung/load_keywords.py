import json

# Load configuration from file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

keyword_files = config['keyword_files']

def load_keywords(keyword_files):
    keywords = {}
    for label, file_path in keyword_files.items():
        with open(file_path, 'r', encoding='utf-8') as file:
            keywords[label] = [line.strip() for line in file]
    return keywords

if __name__ == "__main__":
    keyword_map = load_keywords(keyword_files)
    with open('keywords.json', 'w', encoding='utf-8') as outfile:
        json.dump(keyword_map, outfile, ensure_ascii=False, indent=4)
