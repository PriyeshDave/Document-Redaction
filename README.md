# Document-Redaction
![Python](https://img.shields.io/badge/Python-3.9-blueviolet)
![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-blue)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-lightgrey)
![Natural Language Processing](https://img.shields.io/badge/NLP-SpaCy-blue)
![Cloud](https://img.shields.io/badge/Database-Azure-blue)
![Streamlit](https://img.shields.io/badge/Framework-Flask-yellow)

![Page-Document-Redaction1](https://user-images.githubusercontent.com/81012989/165745898-fed65b54-553a-42e4-a5dd-0fa4b4b9f89e.jpg)

This project revolves around the ability to recognise sensitive words within documents. To do this I am making  use of Natural Language Processing (NLP) where the focus is on Named Entity Recognition which searches a body of text and classifies named entities into predefined categories.

# Problem Statement:
Redaction is the actual process of removing sensitive information from documents. It typically involves someone manually going through a document word by word looking for sensitive information to remove — which of course is a very time consuming and tedious task that is prone to human error. The objective of this project is to build a Machine Learning model that identifies the confidential words from the documents by considering the fact that information of the document is retained.

# Approach
The project is divided into 5 main steps:

### 1.) Data Collection:
* We used LabelBox for labeling the entities in the documents.

### 2.) Data Preprocessing:
* The labelled data is converted to a format used by **Spacy**.
* It was found that while labelling process some of the words were mapped with multiple entities. One of the idea was to remove such sentence but due to data crunch     the such words were preprocessed by removing the multi entity mappings.
* Also, sentence tokenization was done by only selecting the sentences with confidential information.

### 3.) Model Training:
To help me with this we came across a really awesome open-source python library called **SpaCy**, which is a library for advanced Natural Language Processing.
SpaCy is amazing and helps to show the grammatical structure of text through a range of linguistic annotations. It can identify different attributes about the text such as: the base word form; if the word contains alphabetic characters or digits; sentence boundary detection and can tag parts-of-speech words e.g. if the word is a noun, verb, adjective etc.

### 4.) Retraining Model
Retraining the models is not about getting them to memorise more examples but getting the individual models to improve their algorithms so that they can be used to generalise across more and more documents.
* We setup a demo database on Azure for fetching the identified confidential words for model retraining.
* To actually use these words to retrain an existing model, the start index and the end index of each of the words in relation to the whole file needs to be acquired.
* How the retraining happens is, it takes the words from the database and assigns them the label based on the existing knowledge our model has. Then it compares it with the actual label, if the predicted and actual lables don't match it updates it weights to gain the correct result next time. The updates to the model are then saved to the model to update it.

### 5.) Building Pipeline:
 We built REST API's using **flask** for each of the tasks described above.
* **/preprocess** : Used for achieving the preprocessing the labelled data.
* **/databalancing** : The databalancing was done as the count of entities that were been labelled were not balanced. 
* **/modeltraining** : This is used for entire model training.
* **/entityprediction** : It makes use to trained model to identify the confidential words in the document.
* **/stopServer** : Fianlly this endpoint is called for stopping the server.






