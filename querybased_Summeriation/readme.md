﻿# Query based summarization: -
Query based summarization is defined as the extracting the particular information from within words that means essential information that answers the query from original text. With the increasing the demands of solution of the problems in the field of artificial intelligence and natural language processing is one of the most challenging tasks. Query based text summarizer is the most defined and explored topic in the field of natural language processing which involves processing of text document with an appropriate result based on an input query. Query based text summarizer is based on the two techniques that is sentence- sentence and sentence- word – relationship using graph structure. 

# Architecure 
In This i have used Seq2Seq arichitecure with attention mecanisam. First text and wuery passed to endoer to encode the data the their we apply attension to question part and send to the decoder. the decoder is a regression to find the start and end of the text to answer the query.

<p align="center">
  <img src="qbatten.png">
</p>


# Getting started: -
There are mainly two steps in query-based summarization process i.e. Identification of relevant sections from the documents and then the generation of summary. Basically, this summarizer involves selection of the key phrases and sentence that is related to the query from the given information and ensuring them to be in a readable form that is understandable by the user. 

# Prerequisites: -
## Install the pytorch 
https://pytorch.org/?utm_source=Google&utm_medium=PaidSearch&utm_campaign=%2A%2ALP+-+TM+-+General+-+HV+-+IN&utm_adgroup=Install+PyTorch&utm_keyword=install%20pytorch&utm_offering=AI&utm_Product=PyTorch&gclid=CjwKCAjwk93rBRBLEiwAcMapUT2M4zGsIbIVmO-mdQgCaKXWEhquyhMj902KZzA0sL3nGrQre2tuJhoCOMcQAvD_BwE
# 

pip install msgpack<br>
pip install flask<br>
pip install flask-core

# DataSet
Download SQuAD dataset from and unzip it.
https://rajpurkar.github.io/SQuAD-explorer/

# Training
python DataPrepration.py
python train.py

# Deployment: -
PredictApi.py
