import json

# # Funzione che converte il dataset di annotation prodotto dal tool in un formato richiesto dal modello
# def read_dataset():

#     # Leggi i dati di input da un file JSON
#     with open('dataset/annotations.json', 'r') as json_file:
#         train_data = json.load(json_file)
#         print(type(train_data))
    
#         transformed_data = []
#         for item in train_data:
#             text = item["annotations"][0][0]
#             entities = item["annotations"][0][1]["entities"]
#             transformed_data.append((text, {"entities": entities}))
#         return transformed_data


import json

def read_dataset():
    # Leggi i dati di input da un file JSON
    with open('dataset/annotations.json', 'r', encoding='utf-8') as json_file:
        train_data = json.load(json_file)
        print(train_data)

        # Estrai i valori di TRAIN_DATA2
        annotations = train_data["annotations"]

        # Crea una lista vuota per i dati di TRAIN_DATA2 trasformati
        train_data_transformed = []

        # Itera attraverso le annotazioni di TRAIN_DATA2
        for annotation in annotations:
            text = annotation[0]
            entities = annotation[1]["entities"]
            entities_transformed = []

            # Trasforma le entità in un formato simile a TRAIN_DATA1
            for entity in entities:
                start = entity[0]
                end = entity[1]
                label = entity[2]
                entities_transformed.append((start, end, label))

            # Aggiungi i dati trasformati alla lista
            train_data_transformed.append((text, {"entities": entities_transformed}))

        return train_data_transformed


