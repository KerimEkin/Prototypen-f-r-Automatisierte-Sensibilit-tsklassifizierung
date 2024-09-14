import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, IntervalStrategy
from torch.utils.data import Dataset
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from docx import Document
import inspect

# DocumentDataset Klasse
class DocumentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        return inputs, torch.tensor(label)

# Start Skript
print("Skript gestartet...")

# Konfiguration
config = {
    "documents_dir": "V:/penv/sandbox",
    "output_dir": "V:/penv/output",
    "output_file": "classified_documents.csv",
    "output_image": "clusters.png",
    "training_directories": {
        "RED": "V:/penv/training/red",
        "AMBER": "V:/penv/training/amber",
        "GREEN": "V:/penv/training/green",
        "WHITE": "V:/penv/training/white"
    }
}

print("Konfiguration geladen...")

# Funktion zum Extrahieren von Texten aus Dateien
def extract_text_from_file(filepath):
    text = ""
    print(f"Versuche, Text aus Datei zu extrahieren: {filepath}")
    if filepath.endswith('.txt'):
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            print(f"Fehler beim Lesen von {filepath}: {e}")
    elif filepath.endswith('.pdf'):
        try:
            pdf_reader = PdfReader(filepath)
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
    print(f"Extrahierter Text (Anfang): {text[:100]}...")
    return text

# Funktion zum Laden der Daten aus einem Verzeichnis
def load_data_from_directory(directory):
    data = []
    filenames = []
    print(f"Durchsuche Verzeichnis: {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Gefundene Datei: {file_path}")
            text = extract_text_from_file(file_path)
            if text.strip():
                data.append(text)
                filenames.append(file_path)
            else:
                print(f"Kein Text extrahiert aus Datei {file_path}")
    print(f"Anzahl geladener Dateien: {len(filenames)}")
    return data, filenames

# Funktion zum Laden der Trainingsdaten
def load_training_data():
    data = []
    labels = []
    label_map = {'RED': 0, 'AMBER': 1, 'GREEN': 2, 'WHITE': 3}

    for label, directory in config["training_directories"].items():
        print(f"Lade Dateien aus {directory} für Label {label}")
        directory_data, _ = load_data_from_directory(directory)
        if directory_data:
            data.extend(directory_data)
            labels.extend([label_map[label]] * len(directory_data))
        else:
            print(f"Keine Daten in Verzeichnis {directory}")

    print(f"Anzahl der geladenen Trainingsdaten: {len(data)}")
    if not data:
        print("Keine Trainingsdaten gefunden!")
    return data, labels

# Daten und Labels laden
texts, labels = load_training_data()
print(f"Geladene Textdaten für Training: {texts[:2]}")  # Zeigt die ersten zwei Datensätze (ggf. kürzen)
if texts and labels:  # Fortfahren nur, wenn Daten geladen wurden
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

    train_dataset = DocumentDataset(train_texts, train_labels)
    val_dataset = DocumentDataset(val_texts, val_labels)

    # Modelltraining
    try:
        model = BertForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=4)
    except Exception as e:
        print(f"Fehler beim Laden des Modells: {e}")
        exit(1)

    # Trainingsargumente
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=5,
        per_device_train_batch_size=8,
        evaluation_strategy=IntervalStrategy.EPOCH,
        eval_steps=50,
        save_steps=50,
        logging_dir='./logs',
        logging_steps=50,
        save_total_limit=500,
        seed=42
    )

    print("Überprüfe Trainingsargumente:", training_args)
    
    # Trainer-Initialisierung überprüfen
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
    except Exception as e:
        print(f"Fehler bei der Initialisierung des Trainers: {e}")
        exit(1)
    
    print("Trainer erfolgreich initialisiert")

    # Modelltraining starten
    try:
        trainer.train()
    except Exception as e:
        print(f"Fehler während des Trainings: {e}")
        print("Debug-Informationen:")
        for name, value in inspect.getmembers(trainer):
            if not name.startswith('__') and not callable(value):
                print(f"{name}: {value}")

    # Klassifizierung der Dokumente im `documents_dir`
    def classify_documents(directory):
        print(f"Klassifiziere Dokumente im Verzeichnis: {directory}")
        new_texts, filenames = load_data_from_directory(directory)

        if not new_texts:
            print("Keine Dokumente zum Klassifizieren gefunden!")
            return []

        results = []
        for filename, text in zip(filenames, new_texts):
            try:
                inputs = train_dataset.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
                inputs = {key: val.to('cpu') for key, val in inputs.items()}  # Sicherstellen, dass die Inputs auf CPU sind
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
                print(f"Datei {filename} klassifiziert als {prediction}")
                results.append((filename, prediction))
            except Exception as e:
                print(f"Fehler bei der Klassifizierung von {filename}: {e}")

        return results

    classified_docs = classify_documents(config["documents_dir"])
    if classified_docs:
        df = pd.DataFrame(classified_docs, columns=['Document', 'Classification'])
        output_file_path = os.path.join(config["output_dir"], config["output_file"])
        df.to_csv(output_file_path, index=False)
        print(f"Ergebnisse gespeichert in {output_file_path}")
    else:
        print("Keine Dokumente klassifiziert.")

    # Clustering und Visualisierung
    def visualize_clusters(texts):
        print("Starte Visualisierung der Cluster")
        try:
            # Option 1: Without using stop words filtering
            #vectorizer = TfidfVectorizer(stop_words=None)  # No stop words filtering

            # Option 2: Using a custom list of German stop words
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            german_stop_words = ['ab', 'aber', 'abermaliges', 'abermals', 'abgerufen', 'abgerufene', 'abgerufener', 'abgerufenes', 'abgesehen', 'acht', 'aehnlich', 'aehnliche', 'aehnlichem', 'aehnlichen', 'aehnlicher', 'aehnliches', 'aehnlichste', 'aehnlichstem', 'aehnlichsten', 'aehnlichster', 'aehnlichstes', 'aeusserst', 'aeusserste', 'aeusserstem', 'aeussersten', 'aeusserster', 'aeusserstes', 'ähnlich', 'ähnliche', 'ähnlichem', 'ähnlichen', 'ähnlicher', 'ähnliches', 'ähnlichst', 'ähnlichste', 'ähnlichstem', 'ähnlichsten', 'ähnlichster', 'ähnlichstes', 'alle', 'allein', 'alleine', 'allem', 'allemal', 'allen', 'allenfalls', 'allenthalben', 'aller', 'allerdings', 'allerlei', 'alles', 'allesamt', 'allg', 'allg.', 'allgemein', 'allgemeine', 'allgemeinem', 'allgemeinen', 'allgemeiner', 'allgemeines', 'allgemeinste', 'allgemeinstem', 'allgemeinsten', 'allgemeinster', 'allgemeinstes', 'allmählich', 'allzeit', 'allzu', 'als', 'alsbald', 'also', 'am', 'an', 'and', 'andauernd', 'andauernde', 'andauerndem', 'andauernden', 'andauernder', 'andauerndes', 'ander', 'andere', 'anderem', 'anderen', 'anderenfalls', 'anderer', 'andererseits', 'anderes', 'anderm', 'andern', 'andernfalls', 'anderr', 'anders', 'anderst', 'anderweitig', 'anderweitige', 'anderweitigem', 'anderweitigen', 'anderweitiger', 'anderweitiges', 'anerkannt', 'anerkannte', 'anerkannter', 'anerkanntes', 'anfangen', 'anfing', 'angefangen', 'angesetze', 'angesetzt', 'angesetzten', 'angesetzter', 'ans', 'anscheinend', 'ansetzen', 'ansonst', 'ansonsten', 'anstatt', 'anstelle', 'arbeiten', 'auch', 'auf', 'aufgehört', 'aufgrund', 'aufhören', 'aufhörte', 'aufzusuchen', 'augenscheinlich', 'augenscheinliche', 'augenscheinlichem', 'augenscheinlichen', 'augenscheinlicher', 'augenscheinliches', 'augenscheinlichst', 'augenscheinlichste', 'augenscheinlichstem', 'augenscheinlichsten', 'augenscheinlichster', 'augenscheinlichstes', 'aus', 'ausdrücken', 'ausdrücklich', 'ausdrückliche', 'ausdrücklichem', 'ausdrücklichen', 'ausdrücklicher', 'ausdrückliches', 'ausdrückt', 'ausdrückte', 'ausgenommen', 'ausgenommene', 'ausgenommenem', 'ausgenommenen', 'ausgenommener', 'ausgenommenes', 'ausgerechnet', 'ausgerechnete', 'ausgerechnetem', 'ausgerechneten', 'ausgerechneter', 'ausgerechnetes', 'ausnahmslos', 'ausnahmslose', 'ausnahmslosem', 'ausnahmslosen', 'ausnahmsloser', 'ausnahmsloses', 'außen', 'ausser', 'ausserdem', 'außerhalb', 'äusserst', 'äusserste', 'äusserstem', 'äussersten', 'äusserster', 'äusserstes', 'author', 'autor', 'baelde', 'bald', 'bälde', 'bearbeite', 'bearbeiten', 'bearbeitete', 'bearbeiteten', 'bedarf', 'bedürfen', 'bedurfte', 'been', 'befahl', 'befiehlt', 'befiehlte', 'befohlene', 'befohlens', 'befragen', 'befragte', 'befragten', 'befragter', 'begann', 'beginnen', 'begonnen', 'behalten', 'behielt', 'bei', 'beide', 'beidem', 'beiden', 'beider', 'beiderlei', 'beides', 'beim', 'beinahe', 'beisammen', 'beispielsweise', 'beitragen', 'beitrugen', 'bekannt', 'bekannte', 'bekannter', 'bekanntlich', 'bekanntliche', 'bekanntlichem', 'bekanntlichen', 'bekanntlicher', 'bekanntliches', 'bekennen', 'benutzt', 'bereits', 'berichten', 'berichtet', 'berichtete', 'berichteten', 'besonders', 'besser', 'bessere', 'besserem', 'besseren', 'besserer', 'besseres', 'bestehen', 'besteht', 'bestenfalls', 'bestimmt', 'bestimmte', 'bestimmtem', 'bestimmten', 'bestimmter', 'bestimmtes', 'beträchtlich', 'beträchtliche', 'beträchtlichem', 'beträchtlichen', 'beträchtlicher', 'beträchtliches', 'betraechtlich', 'betraechtliche', 'betraechtlichem', 'betraechtlichen', 'betraechtlicher', 'betraechtliches', 'betreffend', 'betreffende', 'betreffendem', 'betreffenden', 'betreffender', 'betreffendes', 'bevor', 'bez', 'bez.', 'bezgl', 'bezgl.', 'bezueglich', 'bezüglich', 'bietet', 'bin', 'bis', 'bisher', 'bisherige', 'bisherigem', 'bisherigen', 'bisheriger', 'bisheriges', 'bislang', 'bisschen', 'bist', 'bitte', 'bleiben', 'bleibt', 'blieb', 'bloss', 'böden', 'boeden', 'brachte', 'brachten', 'brauchen', 'braucht', 'bräuchte', 'bringen', 'bsp', 'bsp.', 'bspw', 'bspw.', 'bzw', 'bzw.', 'ca', 'ca.', 'circa', 'da', 'dabei', 'dadurch', 'dafuer', 'dafür', 'dagegen', 'daher', 'dahin', 'dahingehend', 'dahingehende', 'dahingehendem', 'dahingehenden', 'dahingehender', 'dahingehendes', 'dahinter', 'damalige', 'damaligem', 'damaligen', 'damaliger', 'damaliges', 'damals', 'damit', 'danach', 'daneben', 'dank', 'danke', 'danken', 'dann', 'dannen', 'daran', 'darauf', 'daraus', 'darf', 'darfst', 'darin', 'darüber', 'darüberhinaus', 'darueber', 'darueberhinaus', 'darum', 'darunter', 'das', 'daß', 'dass', 'dasselbe', 'Dat', 'davon', 'davor', 'dazu', 'dazwischen', 'dein', 'deine', 'deinem', 'deinen', 'deiner', 'deines', 'dem', 'demgegenüber', 'demgegenueber', 'demgemaess', 'demgemäss', 'demnach', 'demselben', 'den', 'denen', 'denkbar', 'denkbare', 'denkbarem', 'denkbaren', 'denkbarer', 'denkbares', 'denn', 'dennoch', 'denselben', 'der', 'derart', 'derartig', 'derartige', 'derartigem', 'derartigen', 'derartiger', 'derem', 'deren', 'derer', 'derjenige', 'derjenigen', 'derselbe', 'derselben', 'derzeit', 'derzeitig', 'derzeitige', 'derzeitigem', 'derzeitigen', 'derzeitiges', 'des', 'deshalb', 'desselben', 'dessen', 'dessenungeachtet', 'desto', 'desungeachtet', 'deswegen', 'dich', 'die', 'diejenige', 'diejenigen', 'dies', 'diese', 'dieselbe', 'dieselben', 'diesem', 'diesen', 'dieser', 'dieses', 'diesseitig', 'diesseitige', 'diesseitigem', 'diesseitigen', 'diesseitiger', 'diesseitiges', 'diesseits', 'dinge', 'dir', 'direkt', 'direkte', 'direkten', 'direkter', 'doch', 'doppelt', 'dort', 'dorther', 'dorthin', 'dran', 'drauf', 'drei', 'dreißig', 'drin', 'dritte', 'drüber', 'drueber', 'drum', 'drunter', 'du', 'duerfte', 'duerften', 'duerftest', 'duerftet', 'dunklen', 'durch', 'durchaus', 'durchweg', 'durchwegs', 'dürfen', 'durfte', 'dürfte', 'durften', 'dürften', 'durftest', 'dürftest', 'durftet', 'dürftet', 'eben', 'ebenfalls', 'ebenso', 'ect', 'ect.', 'ehe', 'eher', 'eheste', 'ehestem', 'ehesten', 'ehester', 'ehestes', 'eigen', 'eigene', 'eigenem', 'eigenen', 'eigener', 'eigenes', 'eigenst', 'eigentlich', 'eigentliche', 'eigentlichem', 'eigentlichen', 'eigentlicher', 'eigentliches', 'ein', 'einbaün', 'eine', 'einem', 'einen', 'einer', 'einerlei', 'einerseits', 'eines', 'einfach', 'einführen', 'einführte', 'einführten', 'eingesetzt', 'einig', 'einige', 'einigem', 'einigen', 'einiger', 'einigermaßen', 'einiges', 'einmal', 'einmalig', 'einmalige', 'einmaligem', 'einmaligen', 'einmaliger', 'einmaliges', 'eins', 'einseitig', 'einseitige', 'einseitigen', 'einseitiger', 'einst', 'einstmals', 'einzig', 'empfunden', 'ende', 'entgegen', 'entlang', 'entsprechend', 'entsprechende', 'entsprechendem', 'entsprechenden', 'entsprechender', 'entsprechendes', 'entweder', 'er', 'ergänze', 'ergänzen', 'ergänzte', 'ergänzten', 'ergo', 'erhält', 'erhalten', 'erhielt', 'erhielten', 'erneut', 'eröffne', 'eröffnen', 'eröffnet', 'eröffnete', 'eröffnetes', 'erscheinen', 'erst', 'erste', 'erstem', 'ersten', 'erster', 'erstere', 'ersterem', 'ersteren', 'ersterer', 'ersteres', 'erstes', 'es', 'etc', 'etc.', 'etliche', 'etlichem', 'etlichen', 'etlicher', 'etliches', 'etwa', 'etwaige', 'etwas', 'euch', 'euer', 'eure', 'eurem', 'euren', 'eurer', 'eures', 'euretwegen', 'fall', 'falls', 'fand', 'fast', 'ferner', 'fertig', 'finde', 'finden', 'findest', 'findet', 'folgend', 'folgende', 'folgendem', 'folgenden', 'folgender', 'folgendermassen', 'folgendes', 'folglich', 'for', 'fordern', 'fordert', 'forderte', 'forderten', 'fort', 'fortsetzen', 'fortsetzt', 'fortsetzte', 'fortsetzten', 'fragte', 'frau', 'frei', 'freie', 'freier', 'freies', 'fuer', 'fuers', 'fünf', 'für', 'fürs', 'gab', 'gaenzlich', 'gaenzliche', 'gaenzlichem', 'gaenzlichen', 'gaenzlicher', 'gaenzliches', 'gängig', 'gängige', 'gängigen', 'gängiger', 'gängiges', 'ganz', 'ganze', 'ganzem', 'ganzen', 'ganzer', 'ganzes', 'gänzlich', 'gänzliche', 'gänzlichem', 'gänzlichen', 'gänzlicher', 'gänzliches', 'gar', 'gbr', 'geb', 'geben', 'geblieben', 'gebracht', 'gedurft', 'geehrt', 'geehrte', 'geehrten', 'geehrter', 'gefallen', 'gefälligst', 'gefällt', 'gefiel', 'gegeben', 'gegen', 'gegenüber', 'gegenueber', 'gehabt', 'gehalten', 'gehen', 'geht', 'gekommen', 'gekonnt', 'gemacht', 'gemaess', 'gemäss', 'gemeinhin', 'gemocht', 'genau', 'genommen', 'genug', 'gepriesener', 'gepriesenes', 'gerade', 'gern', 'gesagt', 'gesehen', 'gestern', 'gestrige', 'getan', 'geteilt', 'geteilte', 'getragen', 'getrennt', 'gewesen', 'gewiss', 'gewisse', 'gewissem', 'gewissen', 'gewisser', 'gewissermaßen', 'gewisses', 'gewollt', 'geworden', 'ggf', 'ggf.', 'gib', 'gibt', 'gilt', 'gleich', 'gleiche', 'gleichem', 'gleichen', 'gleicher', 'gleiches', 'gleichsam', 'gleichste', 'gleichstem', 'gleichsten', 'gleichster', 'gleichstes', 'gleichwohl', 'gleichzeitig', 'gleichzeitige', 'gleichzeitigem', 'gleichzeitigen', 'gleichzeitiger', 'gleichzeitiges', 'glücklicherweise', 'gluecklicherweise', 'gmbh', 'gottseidank', 'gratulieren', 'gratuliert', 'gratulierte', 'groesstenteils', 'grösstenteils', 'gruendlich', 'gründlich', 'gut', 'gute', 'guten', 'hab', 'habe', 'haben', 'habt', 'haette', 'haeufig', 'haeufige', 'haeufigem', 'haeufigen', 'haeufiger', 'haeufigere', 'haeufigeren', 'haeufigerer', 'haeufigeres', 'halb', 'hallo', 'halten', 'hast', 'hat', 'hätt', 'hatte', 'hätte', 'hatten', 'hätten', 'hattest', 'hattet', 'häufig', 'häufige', 'häufigem', 'häufigen', 'häufiger', 'häufigere', 'häufigeren', 'häufigerer', 'häufigeres', 'hen', 'her', 'heraus', 'herein', 'herum', 'heute', 'heutige', 'heutigem', 'heutigen', 'heutiger', 'heutiges', 'hier', 'hierbei', 'hiermit', 'hiesige', 'hiesigem', 'hiesigen', 'hiesiger', 'hiesiges', 'hin', 'hindurch', 'hinein', 'hingegen', 'hinlanglich', 'hinlänglich', 'hinten', 'hintendran', 'hinter', 'hinterher', 'hinterm', 'hintern', 'hinunter', 'hoch', 'höchst', 'höchstens', 'http', 'hundert', 'ich', 'igitt', 'ihm', 'ihn', 'ihnen', 'ihr', 'ihre', 'ihrem', 'ihren', 'ihrer', 'ihres', 'ihretwegen', 'ihrige', 'ihrigen', 'ihriges', 'im', 'immer', 'immerhin', 'immerwaehrend', 'immerwaehrende', 'immerwaehrendem', 'immerwaehrenden', 'immerwaehrender', 'immerwaehrendes', 'immerwährend', 'immerwährende', 'immerwährendem', 'immerwährenden', 'immerwährender', 'immerwährendes', 'immerzu', 'important', 'in', 'indem', 'indessen', 'Inf.', 'info', 'infolge', 'infolgedessen', 'information', 'innen', 'innerhalb', 'innerlich', 'ins', 'insbesondere', 'insgeheim', 'insgeheime', 'insgeheimer', 'insgesamt', 'insgesamte', 'insgesamter', 'insofern', 'inzwischen', 'irgend', 'irgendein', 'irgendeine', 'irgendeinem', 'irgendeiner', 'irgendeines', 'irgendetwas', 'irgendjemand', 'irgendjemandem', 'irgendwann', 'irgendwas', 'irgendwelche', 'irgendwen', 'irgendwenn', 'irgendwer', 'irgendwie', 'irgendwo', 'irgendwohin', 'ist', 'ja', 'jaehrig', 'jaehrige', 'jaehrigem', 'jaehrigen', 'jaehriger', 'jaehriges', 'jährig', 'jährige', 'jährigem', 'jährigen', 'jähriges', 'je', 'jede', 'jedem', 'jeden', 'jedenfalls', 'jeder', 'jederlei', 'jedes', 'jedesmal', 'jedoch', 'jeglichem', 'jeglichen', 'jeglicher', 'jegliches', 'jemals', 'jemand', 'jemandem', 'jemanden', 'jemandes', 'jene', 'jenem', 'jenen', 'jener', 'jenes', 'jenseitig', 'jenseitigem', 'jenseitiger', 'jenseits', 'jetzt', 'jung', 'junge', 'jungem', 'jungen', 'junger', 'junges', 'kaeumlich', 'kam', 'kann', 'kannst', 'kaum', 'käumlich', 'kein', 'keine', 'keinem', 'keinen', 'keiner', 'keinerlei', 'keines', 'keineswegs', 'klar', 'klare', 'klaren', 'klares', 'klein', 'kleinen', 'kleiner', 'kleines', 'koennen', 'koennt', 'koennte', 'koennten', 'koenntest', 'koenntet', 'komme', 'kommen', 'kommt', 'konkret', 'konkrete', 'konkreten', 'konkreter', 'konkretes', 'könn', 'können', 'könnt', 'konnte', 'könnte', 'konnten', 'könnten', 'konntest', 'könntest', 'konntet', 'könntet', 'kuenftig', 'kuerzlich', 'kuerzlichst', 'künftig', 'kürzlich', 'kürzlichst', 'laengst', 'lag', 'lagen', 'langsam', 'längst', 'längstens', 'lassen', 'laut', 'lediglich', 'leer', 'legen', 'legte', 'legten', 'leicht', 'leider', 'lesen', 'letze', 'letzte', 'letzten', 'letztendlich', 'letztens', 'letztere', 'letzterem', 'letzterer', 'letzteres', 'letztes', 'letztlich', 'lichten', 'liegt', 'liest', 'links', 'mache', 'machen', 'machst', 'macht', 'machte', 'machten', 'mag', 'magst', 'mal', 'man', 'manch', 'manche', 'manchem', 'manchen', 'mancher', 'mancherlei', 'mancherorts', 'manches', 'manchmal', 'mann', 'margin', 'massgebend', 'massgebende', 'massgebendem', 'massgebenden', 'massgebender', 'massgebendes', 'massgeblich', 'massgebliche', 'massgeblichem', 'massgeblichen', 'massgeblicher', 'mehr', 'mehrere', 'mehrerer', 'mehrfach', 'mehrmalig', 'mehrmaligem', 'mehrmaliger', 'mehrmaliges', 'mein', 'meine', 'meinem', 'meinen', 'meiner', 'meines', 'meinetwegen', 'meins', 'meist', 'meiste', 'meisten', 'meistens', 'meistenteils', 'meta', 'mich', 'mindestens', 'mir', 'mit', 'miteinander', 'mitgleich', 'mithin', 'mitnichten', 'mittels', 'mittelst', 'mitten', 'mittig', 'mitunter', 'mitwohl', 'mochte', 'möchte', 'möchten', 'möchtest', 'moechte', 'moeglich', 'moeglichst', 'moeglichste', 'moeglichstem', 'moeglichsten', 'moeglichster', 'mögen', 'möglich', 'mögliche', 'möglichen', 'möglicher', 'möglicherweise', 'möglichst', 'möglichste', 'möglichstem', 'möglichsten', 'möglichster', 'morgen', 'morgige', 'muessen', 'muesst', 'muesste', 'muss', 'müssen', 'musst', 'müßt', 'musste', 'müsste', 'mussten', 'müssten', 'nach', 'nachdem', 'nacher', 'nachher', 'nachhinein', 'nächste', 'nacht', 'naechste', 'naemlich', 'nahm', 'nämlich', 'naturgemaess', 'naturgemäss', 'natürlich', 'ncht', 'neben', 'nebenan', 'nehmen', 'nein', 'neu', 'neue', 'neuem', 'neuen', 'neuer', 'neuerdings', 'neuerlich', 'neuerliche', 'neuerlichem', 'neuerlicher', 'neuerliches', 'neues', 'neulich', 'neun', 'nicht', 'nichts', 'nichtsdestotrotz', 'nichtsdestoweniger', 'nie', 'niemals', 'niemand', 'niemandem', 'niemanden', 'niemandes', 'nimm', 'nimmer', 'nimmt', 'nirgends', 'nirgendwo', 'noch', 'noetigenfalls', 'nötigenfalls', 'nun', 'nur', 'nutzen', 'nutzt', 'nützt', 'nutzung', 'ob', 'oben', 'ober', 'oberen', 'oberer', 'oberhalb', 'oberste', 'obersten', 'oberster', 'obgleich', 'obs', 'obschon', 'obwohl', 'oder', 'oefter', 'oefters', 'off', 'offenkundig', 'offenkundige', 'offenkundigem', 'offenkundigen', 'offenkundiger', 'offenkundiges', 'offensichtlich', 'offensichtliche', 'offensichtlichem', 'offensichtlichen', 'offensichtlicher', 'offensichtliches', 'oft', 'öfter', 'öfters', 'oftmals', 'ohne', 'ohnedies', 'online', 'paar', 'partout', 'per', 'persoenlich', 'persoenliche', 'persoenlichem', 'persoenlicher', 'persoenliches', 'persönlich', 'persönliche', 'persönlicher', 'persönliches', 'pfui', 'ploetzlich', 'ploetzliche', 'ploetzlichem', 'ploetzlicher', 'ploetzliches', 'plötzlich', 'plötzliche', 'plötzlichem', 'plötzlicher', 'plötzliches', 'pro', 'quasi', 'reagiere', 'reagieren', 'reagiert', 'reagierte', 'recht', 'rechts', 'regelmäßig', 'reichlich', 'reichliche', 'reichlichem', 'reichlichen', 'reichlicher', 'restlos', 'restlose', 'restlosem', 'restlosen', 'restloser', 'restloses', 'richtiggehend', 'richtiggehende', 'richtiggehendem', 'richtiggehenden', 'richtiggehender', 'richtiggehendes', 'rief', 'rund', 'rundheraus', 'rundum', 'runter', 'sage', 'sagen', 'sagt', 'sagte', 'sagten', 'sagtest', 'sagtet', 'samt', 'sämtliche', 'sang', 'sangen', 'sattsam', 'schätzen', 'schätzt', 'schätzte', 'schätzten', 'scheinbar', 'scheinen', 'schlechter', 'schlicht', 'schlichtweg', 'schließlich', 'schlussendlich', 'schnell', 'schon', 'schreibe', 'schreiben', 'schreibens', 'schreiber', 'schwerlich', 'schwerliche', 'schwerlichem', 'schwerlichen', 'schwerlicher', 'schwerliches', 'schwierig', 'sechs', 'sect', 'sehe', 'sehen', 'sehr', 'sehrwohl', 'seht', 'sei', 'seid', 'seien', 'seiest', 'seiet', 'sein', 'seine', 'seinem', 'seinen', 'seiner', 'seines', 'seit', 'seitdem', 'seite', 'seiten', 'seither', 'selbe', 'selben', 'selber', 'selbst', 'selbstredend', 'selbstredende', 'selbstredendem', 'selbstredenden', 'selbstredender', 'selbstredendes', 'seltsamerweise', 'senke', 'senken', 'senkt', 'senkte', 'senkten', 'setzen', 'setzt', 'setzte', 'setzten', 'sich', 'sicher', 'sicherlich', 'sie', 'sieben', 'siebte', 'siehe', 'sieht', 'sind', 'singen', 'singt', 'so', 'sobald', 'sodaß', 'soeben', 'sofern', 'sofort', 'sog', 'sogar', 'sogleich', 'solange', 'solc', 'solc hen', 'solch', 'solche', 'solchem', 'solchen', 'solcher', 'solches', 'soll', 'sollen', 'sollst', 'sollt', 'sollte', 'sollten', 'solltest', 'solltet', 'somit', 'sondern', 'sonst', 'sonstig', 'sonstige', 'sonstigem', 'sonstiger', 'sonstwo', 'sooft', 'soviel', 'soweit', 'sowie', 'sowieso', 'sowohl', 'später', 'spielen', 'startet', 'startete', 'starteten', 'statt', 'stattdessen', 'steht', 'steige', 'steigen', 'steigt', 'stellenweise', 'stellenweisem', 'stellenweisen', 'stets', 'stieg', 'stiegen', 'such', 'suchen', 'tages', 'tat', 'tät', 'tatsächlich', 'tatsächlichen', 'tatsächlicher', 'tatsächliches', 'tatsaechlich', 'tatsaechlichen', 'tatsaechlicher', 'tatsaechliches', 'tausend', 'teile', 'teilen', 'teilte', 'teilten', 'tief', 'titel', 'toll', 'total', 'trage', 'tragen', 'trägt', 'trotzdem', 'trug', 'tun', 'tust', 'tut', 'txt', 'übel', 'über', 'überall', 'überallhin', 'überaus', 'überdies', 'überhaupt', 'überll', 'übermorgen', 'üblicherweise', 'übrig', 'übrigens', 'ueber', 'ueberall', 'ueberallhin', 'ueberaus', 'ueberdies', 'ueberhaupt', 'uebermorgen', 'ueblicherweise', 'uebrig', 'uebrigens', 'um', 'ums', 'umso', 'umstaendehalber', 'umständehalber', 'unbedingt', 'unbedingte', 'unbedingter', 'unbedingtes', 'und', 'unerhoert', 'unerhoerte', 'unerhoertem', 'unerhoerten', 'unerhoerter', 'unerhoertes', 'unerhört', 'unerhörte', 'unerhörtem', 'unerhörten', 'unerhörter', 'unerhörtes', 'ungefähr', 'ungemein', 'ungewoehnlich', 'ungewoehnliche', 'ungewoehnlichem', 'ungewoehnlichen', 'ungewoehnlicher', 'ungewoehnliches', 'ungewöhnlich', 'ungewöhnliche', 'ungewöhnlichem', 'ungewöhnlichen', 'ungewöhnlicher', 'ungewöhnliches', 'ungleich', 'ungleiche', 'ungleichem', 'ungleichen', 'ungleicher', 'ungleiches', 'unmassgeblich', 'unmassgebliche', 'unmassgeblichem', 'unmassgeblichen', 'unmassgeblicher', 'unmassgebliches', 'unmoeglich', 'unmoegliche', 'unmoeglichem', 'unmoeglichen', 'unmoeglicher', 'unmoegliches', 'unmöglich', 'unmögliche', 'unmöglichen', 'unmöglicher', 'unnötig', 'uns', 'unsaeglich', 'unsaegliche', 'unsaeglichem', 'unsaeglichen', 'unsaeglicher', 'unsaegliches', 'unsagbar', 'unsagbare', 'unsagbarem', 'unsagbaren', 'unsagbarer', 'unsagbares', 'unsäglich', 'unsägliche', 'unsäglichem', 'unsäglichen', 'unsäglicher', 'unsägliches', 'unse', 'unsem', 'unsen', 'unser', 'unsere', 'unserem', 'unseren', 'unserer', 'unseres', 'unserm', 'unses', 'unsre', 'unsrem', 'unsren', 'unsrer', 'unsres', 'unstreitig', 'unstreitige', 'unstreitigem', 'unstreitigen', 'unstreitiger', 'unstreitiges', 'unten', 'unter', 'unterbrach', 'unterbrechen', 'untere', 'unterem', 'unteres', 'unterhalb', 'unterste', 'unterster', 'unterstes', 'unwichtig', 'unzweifelhaft', 'unzweifelhafte', 'unzweifelhaftem', 'unzweifelhaften', 'unzweifelhafter', 'unzweifelhaftes', 'usw', 'usw.', 'vergangen', 'vergangene', 'vergangener', 'vergangenes', 'vermag', 'vermögen', 'vermutlich', 'vermutliche', 'vermutlichem', 'vermutlichen', 'vermutlicher', 'vermutliches', 'veröffentlichen', 'veröffentlicher', 'veröffentlicht', 'veröffentlichte', 'veröffentlichten', 'veröffentlichtes', 'verrate', 'verraten', 'verriet', 'verrieten', 'version', 'versorge', 'versorgen', 'versorgt', 'versorgte', 'versorgten', 'versorgtes', 'viel', 'viele', 'vielen', 'vieler', 'vielerlei', 'vieles', 'vielleicht', 'vielmalig', 'vielmals', 'vier', 'voellig', 'voellige', 'voelligem', 'voelligen', 'voelliger', 'voelliges', 'voelligst', 'vollends', 'völlig', 'völlige', 'völligem', 'völligen', 'völliger', 'völliges', 'völligst', 'vollstaendig', 'vollstaendige', 'vollstaendigem', 'vollstaendigen', 'vollstaendiger', 'vollstaendiges', 'vollständig', 'vollständige', 'vollständigem', 'vollständigen', 'vollständiger', 'vollständiges', 'vom', 'von', 'vor', 'voran', 'vorbei', 'vorgestern', 'vorher', 'vorherig', 'vorherige', 'vorherigem', 'vorheriger', 'vorne', 'vorüber', 'vorueber', 'wachen', 'waehrend', 'waehrenddessen', 'waere', 'während', 'währenddessen', 'wann', 'war', 'wär', 'wäre', 'waren', 'wären', 'warst', 'wart', 'warum', 'was', 'weder', 'weg', 'wegen', 'weil', 'weiß', 'weit', 'weiter', 'weitere', 'weiterem', 'weiteren', 'weiterer', 'weiteres', 'weiterhin', 'weitestgehend', 'weitestgehende', 'weitestgehendem', 'weitestgehenden', 'weitestgehender', 'weitestgehendes', 'weitgehend', 'weitgehende', 'weitgehendem', 'weitgehenden', 'weitgehender', 'weitgehendes', 'welche', 'welchem', 'welchen', 'welcher', 'welches', 'wem', 'wen', 'wenig', 'wenige', 'weniger', 'wenigstens', 'wenn', 'wenngleich', 'wer', 'werde', 'werden', 'werdet', 'weshalb', 'wessen', 'weswegen', 'wichtig', 'wie', 'wieder', 'wiederum', 'wieso', 'wieviel', 'wieviele', 'wievieler', 'wiewohl', 'will', 'willst', 'wir', 'wird', 'wirklich', 'wirklichem', 'wirklicher', 'wirkliches', 'wirst', 'wo', 'wobei', 'wodurch', 'wofuer', 'wofür', 'wogegen', 'woher', 'wohin', 'wohingegen', 'wohl', 'wohlgemerkt', 'wohlweislich', 'wolle', 'wollen', 'wollt', 'wollte', 'wollten', 'wolltest', 'wolltet', 'womit', 'womoeglich', 'womoegliche', 'womoeglichem', 'womoeglichen', 'womoeglicher', 'womoegliches', 'womöglich', 'womögliche', 'womöglichem', 'womöglichen', 'womöglicher', 'womögliches', 'woran', 'woraufhin', 'woraus', 'worden', 'worin', 'wuerde', 'wuerden', 'wuerdest', 'wuerdet', 'wurde', 'würde', 'wurden', 'würden', 'wurdest', 'würdest', 'wurdet', 'würdet', 'www', 'x', 'z.B.', 'zahlreich', 'zahlreichem', 'zahlreicher', 'zB', 'zb.', 'zehn', 'zeitweise', 'zeitweisem', 'zeitweisen', 'zeitweiser', 'ziehen', 'zieht', 'ziemlich', 'ziemliche', 'ziemlichem', 'ziemlichen', 'ziemlicher', 'ziemliches', 'zirka', 'zog', 'zogen', 'zu', 'zudem', 'zuerst', 'zufolge', 'zugleich', 'zuletzt', 'zum', 'zumal', 'zumeist', 'zumindest', 'zunächst', 'zunaechst', 'zur', 'zurück', 'zurueck', 'zusammen', 'zusehends', 'zuviel', 'zuviele', 'zuvieler', 'zuweilen', 'zwanzig', 'zwar', 'zwei', 'zweifelsfrei', 'zweifelsfreie', 'zweifelsfreiem', 'zweifelsfreien', 'zweifelsfreier', 'zweifelsfreies', 'zwischen', 'zwölf']
            vectorizer = TfidfVectorizer(stop_words=german_stop_words)

            X = vectorizer.fit_transform(texts)

            kmeans = KMeans(n_clusters=4, random_state=42)
            kmeans.fit(X)

            pca = PCA(n_components=2)
            scatter_plot_points = pca.fit_transform(X.toarray())

            colors = ["r", "y", "g", "gray"]
            x_axis = [o[0] for o in scatter_plot_points]
            y_axis = [o[1] for o in scatter_plot_points]

            plt.figure(figsize=(10, 3))
            plt.scatter(x_axis, y_axis, c=[colors[d] for d in kmeans.labels_])
            plt.title('Document Clusters')
            plt.savefig(os.path.join(config["output_dir"], config["output_image"]))
            plt.show()
            print("Cluster-Visualisierung abgeschlossen")
        except Exception as e:
            print(f"Fehler bei der Visualisierung: {e}")

    # Visualisierungsaufruf
    visualize_clusters(texts)
else:
    print("Training wurde abgebrochen, da keine Daten geladen werden konnten.")
          
