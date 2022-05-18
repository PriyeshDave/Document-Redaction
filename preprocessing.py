import json
import requests
import copy
from bs4 import BeautifulSoup 
import re
import pickle
import os
import spacy
import unicodedata
from thinc.api import prefer_gpu
from unicodedata import name
import pyodbc
import pandas as pd
import fitz
from pathlib import Path


is_gpu = prefer_gpu()

nlp = spacy.load('en_core_web_lg')


def get_sentences(fulltext, entities):
    """
        This method takes the labelled file from LabelBox and tokenizes into sentences.
        And then picks only those sentences having entities tagged.
        The start and end positions of the entities are changed w.r.t the tokenized sentence.
    """
    doc = nlp(fulltext)
    sentences = []
    new_TRAIN_DATA = []

    for sent in doc.sents:
        sentences.append((sent.start_char, sent.end_char, sent.text))

    for sent in sentences:
        text = sent[2]
        new_entities = []
        
        for entity in entities:
            if (entity[0] in range(sent[0], sent[1])) & (entity[1] in range(sent[0], sent[1])):
                search_text = re.escape(fulltext[entity[0]:entity[1]].strip())

                res = re.search(search_text, text)

                if (entity[2] == 'EMAIL') & ("http://" in text[entity[0]:entity[1]]):
                    tag = 'DOMAIN_NAME'
                else:
                    tag = entity[2]
                new_entities.append((res.start(),res.end(),tag))

        if new_entities:
            new_TRAIN_DATA.append((text,{'entities':new_entities}))
    
    return new_TRAIN_DATA


def strip_spaces(train_text, start, end):
    """
        This method strips the extra spaces selected during annotation.
        It also includes the entire word if partial words are selected while tagging.
    """
    new_end = end
    new_start = start
    text = train_text[start:end]
    length = len(train_text)

    lspace_count = len(text) - len(text.lstrip())
    rspace_count = len(text) - len(text.rstrip())

    if lspace_count > 0:
        new_start = new_start + lspace_count

    if (rspace_count > 0):
        new_end = new_end - rspace_count        

    if (new_end == (len(train_text))):
        pass
    elif (new_end < (len(train_text))):
        if train_text[new_end].isspace():
            pass
        else:
            if length - new_end < 50:
                tr = train_text[new_end:]
            else:
                tr = train_text[new_end:new_end+50]

            chars = tr.split()                        
            if chars:
                new_end = new_end + len(chars[0])

    if new_start > 0:
        if train_text[new_start - 1].isspace():
            pass
        else:
            if new_start < 50:
                tr2 = train_text[:new_start]
            else:
                tr2 = train_text[new_start - 50:new_start]

            chars2 = tr2.split()
            
            if chars2:
                new_start = new_start - len(chars2[-1])

    return new_start, new_end


class Preprocessing:
    """
        Class for pre-processing steps of annotated LabelBox data
    """

    def __init__(self):
        self.TRAIN_DATA = []
        self.df = pd.DataFrame()
        # self.tagged_file = tagged_file

    def connect_db(self, server, database, username, password, driver):
        """
        objective: connects to the azure sql db, fetch the data from there and returns the pandas dataframe.
        server: server name
        database: database name
        username: username of the db
        password: pwd to the db
        driver: pyodbc driver 18
        """

        if (server, database, username, password, driver) != None:

            with pyodbc.connect(
                    'DRIVER={ODBC Driver 18 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password) as conn:
                self.df = pd.read_sql_query('Select * from demo order by document_id', conn)
                print("connection established")
        else:
            print("connection failed")

    def entity_add(self, doc, label, page_no, x1, x2, y1, y2):
        """
        objective: preprocess the pdf and creates the training data.
        label: entity type
        page_no: page number we are fetching the data from pdf.
        x1,x2,y1,y2: coordinates of the entity.

        """
        train_data = []
        page = doc[page_no]  # we want text from this page
        print('page dimension - ', page.rect.width, page.rect.height)

        # The co-ordinates in the DB are image co-ords. W.r.t pdf, y1 and y2 has to be interchanged.
        # To ensure entire word to be selected, slight correction in x1,y1,x2,y2 is done.

        rect = [float(x1) - 2, page.rect.height - float(y2) - 2, float(x2) + 2, page.rect.height - float(y1) + 2]
        print(rect)
        labelled_text = page.get_textbox(rect)

        # Removing all unprintable characters from the text
        string = re.sub('[^0-9a-zA-Z@?|\/<>.,()&^%$#!]+', ' ', labelled_text)
        print('string : ', string)

        fulltext = page.get_text()
        doc = nlp(fulltext)

        sentences = []
        for sent in doc.sents:
            if string.strip() in sent.text:
                text = re.sub('[^0-9a-zA-Z@?|\/<>.,()&^%$#!]+', ' ', sent.text)

        res = re.search(string.strip(), text)

        entities = []
        entities.append((res.start(), res.end(), label))

        train_data.append((text, {'entities': entities}))
        return train_data

    def train_data_from_db(self):
        prev_doc_id = []
        for index, data in self.df.iterrows():
            if data['type_of_selection'] == 'Text':
                document_id = data['document_id']
                x1, y1, x2, y2 = data['area'][1:-1].split(',')
                url = data['link']
                page_no = int(data['page']) - 1
                label = data['entity_type']
                filename = url.split('/')[-1]
                # global doc
                if data['document_id'] not in prev_doc_id:
                    prev_doc_id.append(data['document_id'])
                    response = requests.get(url)
                    doc = fitz.open(stream=response.content, filetype="pdf")
                    self.TRAIN_DATA.extend(self.entity_add(doc, label, page_no, x1, x2, y1, y2))
                else:
                    self.TRAIN_DATA.extend(self.entity_add(doc, label, page_no, x1, x2, y1, y2))

    def convert_to_spacy(self, data, tokenization='sentence'):
        """
            Convert the LabelBox json to spacy format
        """
        # with open(tagged_file) as f:
        #    data = json.load(f)

        for i in range(len(data)):

            if not data[i]['Skipped']:
                r = requests.get(data[i]['Labeled Data'])
                soup = BeautifulSoup(r.content, 'html.parser')
                text_data = unicodedata.normalize("NFKD", soup.text).replace('\t', ' ')

                ent_list = []
                for j in range(len(data[i]['Label']['objects'])):
                    data1 = data[i]['Label']['objects'][j]['data']['location']['start']
                    data2 = data[i]['Label']['objects'][j]['data']['location']['end']
                    data3 = data[i]['Label']['objects'][j]['title']

                    ent_list.append((data1, data2, data3))
                    entities = sorted(ent_list, key = lambda x: x[0])

                if tokenization == 'sentence':
                    TRAIN_DATA = get_sentences(text_data, entities)
                    self.TRAIN_DATA.extend(TRAIN_DATA)
                else:
                    self.TRAIN_DATA.append((text_data, {'entities' : ent_list}))

    def fix_partial_word_selection(self):
        """
            Fixing annotation errors
            1. Spacy requires entire word to be tagged. If partial words are tagged, then the start or end position is changed to point to the beginning or end of the word.
            2. There should not be any leading or trailing spaces included in the tagged token.
        """

        new_TRAIN_DATA_UPD = []

        for i in range(0,len(self.TRAIN_DATA)):
            train_text = copy.deepcopy(self.TRAIN_DATA[i][0])
            train_entities = copy.deepcopy(self.TRAIN_DATA[i][1]['entities'])

            new_entities = []

            for start, end, tag in train_entities:
                new_start, new_end = strip_spaces(train_text, start, end)
                new_entities.append((new_start, new_end, tag))

            new_TRAIN_DATA_UPD.append((train_text, {'entities':new_entities}))

        self.TRAIN_DATA = new_TRAIN_DATA_UPD

    def fix_conflicting_annotation(self):
        """
            Removing Conflicting annotations - If same token is tagged with two different labels, or tagged twice inadvertently, these conflicts are removed through the script.
        """
        new_TRAIN_DATA_NEW = []

        for i in range(0, len(self.TRAIN_DATA)):
            tagged_data = self.TRAIN_DATA[i]

            file_text = copy.deepcopy(tagged_data[0])
            annotations = copy.deepcopy(tagged_data[1])

            ent = sorted(annotations['entities'], key = lambda x: x[0])

            prev_start = 0
            prev_end = 0
            prev_tag = ''
            new_ent = []

            for (start, end, tag) in ent:
                if (start >= prev_start) & (end <= prev_end):
                    continue
                elif (start >= prev_start) & (start <= prev_end):
                    if (end >= prev_end) & (len(new_ent) > 0):
                      new_ent.pop()
                      new_ent.append((start,end,tag))
                else:
                    new_ent.append((start,end,tag))
                prev_start = start
                prev_end = end
                prev_tag = tag

            annotations['entities'] = new_ent

            new_TRAIN_DATA_NEW.append((file_text, annotations))

        self.TRAIN_DATA = new_TRAIN_DATA_NEW

        return self.TRAIN_DATA
