#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:48:57 2023

@author: lobeto
"""
from flask import Flask, request, jsonify
from annoy import AnnoyIndex
from PIL import Image
import io 
import torchvision.transforms as transforms

import gradio as gr
import requests
import random
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import model
from model_2 import model_2
#from model.py import model
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torchvision.models as models

from transformers import DistilBertModel, DistilBertTokenizerFast
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
import io

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords
import re


app = Flask(__name__)

# Load the Annoy database
dim=576
annoy_db = AnnoyIndex(576, metric='angular')  # Here 2 is the dimension of the vectors in the database in my example
                                            # you would replace it with the dimension of your vectors
annoy_db.load('TabIndex1712.ann')  # Replace 'annoy_db.ann' with the path to your Annoy database

movies_metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

@app.route('/reco_poster', methods=['POST']) # This route is used to get recommendations
def reco_poster():
    vector = request.json['vector'] # Get the vector from the request
    print('ok')
    closest_indices = annoy_db.get_nns_by_vector(vector[0], 3)
    print(closest_indices) # Get the 2 closest elements indices
    reco = [ closest_indices[0],closest_indices[1],closest_indices[2]]  # Assuming the indices are integers
    #Il faudrait effectuer un filtrage pour exclure les images similaires avec hash perceptuel
    print(reco)
    return jsonify(reco) # Return the reco as a JSON

# Fonction pour obtenir les titres de films à partir des indices
def get_movie_titles_from_indices(indices):
    return movies_metadata.loc[indices, 'title'].tolist()

# Route pour la page d'accueil
@app.route('/')
def index():
    return 'Hello world!'

# Construire l'index Annoy
annoy_bert = AnnoyIndex(768, metric='angular')
annoy_bert.load('annoy_bert.ann')

annoy_tfidf = AnnoyIndex(200, metric='angular')  
annoy_tfidf.load('annoy_tfidf.ann')

# Route pour la recommandation de films
@app.route('/reco_texte', methods=['POST'])
def reco_texte():
    vector = request.json['vector']
    method_choice = request.json.get('method_choice')  # Default to Bert if not specified
    #print(method_choice)
    if method_choice == 'Bert':
        closest_indices = annoy_bert.get_nns_by_vector(vector, 5)
    elif method_choice == 'TFIDF':
        closest_indices = annoy_tfidf.get_nns_by_vector(vector, 5)
    else:
        return jsonify('Méthode non supportée')
    
    reco = get_movie_titles_from_indices(closest_indices)
    return jsonify(reco)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) # Run the server on port 5000 and make it accessible externallys

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

'''
##### OLD VERSION #####
# Construire l'index Annoy
dim=768
annoy_bert = AnnoyIndex(768, metric='angular')  
annoy_bert.load('annoy_bert.ann')

@app.route('/') # This is the home route, it just returns 'Hello world!'
def index():    # I use it to check that the server is running and accessible it's not necessary
    return 'Hello world!' 
    
# Fonction pour obtenir les titres de films à partir des indices
def get_movie_titles_from_indices(indices):
    return movies_metadata.loc[indices, 'title'].tolist()

# Route pour la recommandation de films
@app.route('/reco_texte', methods=['POST'])
def reco_texte():
    try:
        vector = request.json['vector']
        print(f'Received request for reco with vector: {vector}')
        closest_indices = annoy_bert.get_nns_by_vector(vector[0], 3)
        reco = get_movie_titles_from_indices(closest_indices)
        print(f'Recommendations: {reco}')
        return jsonify(reco)
    except Exception as e:
        print(f'Error in reco route: {e}')
        return 'Internal server error'
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 
        '''