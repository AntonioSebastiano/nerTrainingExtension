import random
from pathlib import Path
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
from dataset.script_dataset import read_dataset

# Caricamento del modello spacy italiano
nlp = spacy.load('it_core_news_lg')

# Controllo se il modello ha il componente NER
if "ner" not in nlp.pipe_names:
    raise ValueError("Il modello non ha il componente NER. Assicurati di caricare un modello con NER.")

# Caricamento del dizionario italiano
ner = nlp.get_pipe("ner")

# Trasformo il dataset in un formato richiesto
TRAIN_DATA = read_dataset()

# Aggiunta delle etichette di base esistenti alle etichette estratte dal dataset
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Shuffle dei dati di addestramento
random.shuffle(TRAIN_DATA)

# Disabilita i componenti del pipeline che non devono essere modificati
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Training del modello
with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.begin_training()
    for iteration in range(100):
        random.shuffle(TRAIN_DATA)  # Shuffle dei dati all'inizio di ogni iterazione
        losses = {}
        for batch in minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001)):
            texts, annotations = zip(*batch)
            examples = []
            for i in range(len(texts)):
                doc = nlp.make_doc(texts[i])
                examples.append(Example.from_dict(doc, annotations[i]))
            nlp.update(examples, sgd=optimizer, drop=0.5, losses=losses)
        print("Losses", losses)


# Esecuzione di test del modello
doc = nlp("Antonio ha 18 anni e Martedì deve salire sul volo BK564 per Londra")
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

# Salvataggio del modello in una directory
output_dir = Path('./model/')
output_dir.mkdir(parents=True, exist_ok=True)
nlp.to_disk(output_dir)
print("Saved model to", output_dir)
