import pickle
import spacy
import json
import random
from unicodedata import name
from spacy.tokens import DocBin
from tqdm import tqdm


class Converter:

    def load(self, path: str) -> list:
        """
            path - path to pickle file, type: str
            return data, type: list
        """
        with open(path, 'rb') as handle:
            train_data = pickle.load(handle)
            return train_data
        
    def balancing_data(self, data: list, min_count: int, labels: list) -> list:
        """data - loaded data, type: list
        labels - entities to be included in balanced data, type: list

        return balanced data which including only entities specified, type: list
        """
        balanced_data = []
        
        if labels:
            pass
        else:
            labels = []
            labels_cnt = {}

            for _, annotations in data:
                for ent in annotations.get("entities"):
                    if ent[2] not in labels_cnt.keys():
                        labels_cnt[ent[2]] = 0

                    labels_cnt[ent[2]] += 1
                    
            if min_count:
                for key in labels_cnt.keys():
                    if labels_cnt[key] > min_count:
                        labels.append(key)
            
            else:
                labels = labels_cnt.keys()

        print('labels : ', labels)

        for i in range(0, len(data)):
            entities = data[i][1]['entities']
            new_entities = []
            for (a, b, c) in entities:
                if c in labels:
                    new_entities.append((a, b, c))
            new_DATA = (data[i][0], {'entities': new_entities})
            balanced_data.append(new_DATA)                

        print('len(balanced_data) : ', len(balanced_data))
        print('balanced_data[0] : ', balanced_data[0])

        return balanced_data

    def convert(self, data: list) -> DocBin:
        """data - loaded data, type: list
        return Docbins
        """

        nlp = spacy.blank("en")

        db = DocBin()

        for text, annotations in tqdm(data):
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in annotations.get("entities"):
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    print("skipping entity")
                else:
                    ents.append(span)

            doc.ents = ents
            db.add(doc)

        return db
    
    def save(self, data: DocBin, path: str) -> None:
        data.to_disk(path)
        print(f"saved in {path}")
