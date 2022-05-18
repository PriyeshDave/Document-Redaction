import json
import os
import pickle
import random
from flask import Flask, request, jsonify, flash, redirect
from flask import current_app as app
from preprocessing import *
from converter import Converter
from training import train
import predict
import sys
import signal
import fitz


app = Flask(__name__)

data_path = os.path.join(os.getcwd(), 'DATA')
if not os.path.exists(data_path):
    os.makedirs(data_path)


@app.route('/preprocess', methods=['POST'])
def preprocess_tagged_data():
    """
    Request inputs:
        file : json exported file from LabelBox
        tokenization: sentence tokenization or entire document to be considered. (optional)
                      Default value is "sentence" tokenization
    Request outputs:
        Training data saved as pickle file in DATA folder
        Training data is also sent in response.
    """

    print("Initializing file loading")
    tokenization = request.form.get('tokenization')
    source = request.form.get('source', 'LabelBox')

    preprocess_file = Preprocessing()

    if source.lower() == 'labelbox':
        file = request.files['file']
        tagged_data = json.loads(file.read())
        preprocess_file.convert_to_spacy(tagged_data, tokenization)

    elif source.lower() == 'AzureSQL':
        driver = '{ODBC Driver 18 for SQL Server}'
        server = request.form.get('server')
        database = request.form.get('database')
        username = request.form.get('username')
        password = request.form.get('password')
        preprocess_file.connect_db(server, database, username, password, driver)
        preprocess_file.train_data_from_db()
    else:
        return "Data source not supported"

    preprocess_file.fix_partial_word_selection()
    TRAIN_DATA = preprocess_file.fix_conflicting_annotation()

    # Store data (serialize) using pickle. Pickle is used to maintain the tuple format.
    # JSON doesnt support tuples. Hence while saving, tuples are converted to lists.
    out_file = os.path.join(data_path, 'training_data.pickle')
    with open(out_file, 'wb') as handle:
        pickle.dump(TRAIN_DATA, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return "Labelled data converted to spacy format successfully"


@app.route('/databalancing', methods=['POST'])
def data_balancing():
    """
    Request inputs:
        split_percent : Percentage split of training data to train and test sets.
        min_label_count: Minimum count of each class. The labels having less data will be removed from training data.
        labels: (Optional) List of labels to be included in the training data
    Request outputs:
        Training data saved as pickle file in DATA folder
        Training data is also sent in response.
    """

    split_percent = request.form.get('split_percent', '20')
    if split_percent.isnumeric():

        if 10 <= int(split_percent) <= 50:
            train_test_split = int(split_percent)/100
        else:
            train_test_split = 20/100
    else:
        train_test_split = 20/100

    min_count = request.form.get('min_label_count', '0')

    if min_count.isnumeric():
        min_label_count = int(min_count)
    else:
        min_label_count = 1

    labels = request.form.get('labels', [])

    converter = Converter()
    data = converter.load(os.path.join(data_path, 'training_data.pickle'))

    # get the balanced data
    balanced_data = converter.balancing_data(data, min_label_count, labels)

    if balanced_data:
        random.shuffle(balanced_data)
        split = int(len(balanced_data) * train_test_split)
        train_data = converter.convert(balanced_data[:split])
        converter.save(train_data, path=os.path.join(data_path, 'train.spacy'))

        validation_data = converter.convert(balanced_data[split:])
        converter.save(validation_data, path=os.path.join(data_path, "dev.spacy"))

    return "Training and testing data created successfully"


@app.route('/modeltraining', methods=['POST'])
def model_training():
    print("method accessed")

    # train model with balanced data
    model_path = os.path.join(os.getcwd(), 'MODEL')
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train()

    return "Method trained"


@app.route('/entityprediction', methods=['POST'])
def entity_prediction():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    process_type = request.form.get('type', 'page')

    file_path = os.path.join(os.getcwd(), 'DATA', file.filename)
    file.save(file_path)

    result = []

    with fitz.open(file_path) as doc:

        if process_type.lower() == 'page':
            for page in doc:
                text = page.get_text()
                value = predict.predict(text)
                page_no = page.number + 1

                if value:
                    result.append({'page': page_no, 'prediction': value})
        else:
            text = ''

            for page in doc:
                text += page.get_text()
            
            result = predict.predict(text)

    return json.dumps(result)


@app.route('/stopServer', methods=['GET'])
def stopServer():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({ "success": True, "message": "Server is shutting down..." })


if __name__ == "__main__":

    app.run(port=5432)




