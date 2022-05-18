import spacy
import os


def predict(doc):
    model_path = os.path.join(os.getcwd(), 'MODEL', 'model-best')
    nlp = spacy.load(model_path)

    doc = nlp(doc)
    entities = []

    for ent in doc.ents:
        entities.append(
            {
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            }
        )

    return entities

if __name__ == "__main__":
    print(predict("Mr. Jhon, who is owning the company. To find out what has changed, call us at 1-800-252-2551 or write to us at: PO BOX 100114 Columbia, SC 29202-3114."))