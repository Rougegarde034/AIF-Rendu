#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:46:57 2023

@author: lobeto
"""

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
#import torchvision.models as models

from transformers import DistilBertModel, DistilBertTokenizerFast
import numpy as np
from annoy import AnnoyIndex
from flask import Flask, jsonify, request, send_from_directory
import io


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from nltk.corpus import stopwords
import re

df=pd.read_csv('DF_path1712.csv')


#Définition fonction transform qui normalise
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]
normalize = transforms.Normalize(mean, std)
inv_normalize = transforms.Normalize(
   mean= [-m/s for m, s in zip(mean, std)],
   std= [1/s for s in std]
)

transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                normalize])

def process_image(image):
    # try:
        image_transfo = transform(image)
        vector = model(image_transfo.unsqueeze(0)).cpu().detach().numpy().tolist()

        # Now we send the vector to the API
        response = requests.post('http://annoy-db:5000/reco_poster', json={'vector': vector})
        print(response.status_code)
        if response.status_code == 200:
            print('ok')
            indices = response.json()

            # Retrieve paths for the indices
            #paths = df[df['index'].isin(indices)]['path'].tolist()
            paths=list()
            for i in indices:
                paths.append(str(df.path[i]))
            #print(paths)
            
            # Plot the images
            fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
            for i, path in enumerate(paths):
                img = Image.open(path)
                axs[i].imshow(img)
                axs[i].axis('off')
            return fig
        else:
            print('erreur figure')
            # Return an error figure
            return plt.figure()
    # except Exception as e:
    #     print(f"Error in process_image: {e}")
    #     # Return an error figure
    #     return plt.figure()

# Load df
movies_metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

nltk.download('punkt')
nltk.download('stopwords')

# Download stopwords list
stop_words = set(stopwords.words('english'))

# Interface lemma tokenizer from nltk with sklearn
class StemTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

    def __init__(self):
        self.stemmer = SnowballStemmer('english')

    def __call__(self, doc):
        doc = doc.lower()
        return [self.stemmer.stem(t) for t in word_tokenize(re.sub("[^a-z' ]", "", doc)) if t not in self.ignore_tokens]

tokenizer = StemTokenizer()
tokenizer2 = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

token_stop = tokenizer(' '.join(stop_words))

# Fonction pour obtenir les embeddings DistilBERT
def get_distilbert_embeddings(text):
    if pd.isna(text):
        return np.zeros(model_2.config.dim).tolist()  # Convertir le tableau en liste
    
    inputs = tokenizer2(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: value.cpu() for key, value in inputs.items()}  # Transférer les données sur le CPU
    with torch.no_grad():
        outputs = model_2(**inputs)
    
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy().tolist()  # Convertir le tableau en liste

# Fonction pour obtenir les titres de films à partir des indices
def get_movie_titles_from_indices(indices):
    return movies_metadata.loc[indices, 'title'].tolist()

# Fonction pour obtenir les embeddings TFIDF
def get_tfidf_embeddings(text):
    tfidf = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer, max_features=200)
    train_tfidf_matrix = tfidf.fit_transform(movies_metadata.overview.values.astype('U'))
    
    # Transform the input description using the fitted TF-IDF vectorizer
    vector = tfidf.transform([text])
    dense_vector = vector.toarray().flatten()
    return dense_vector.tolist()

#model_2 = DistilBertModel.from_pretrained('distilbert-base-uncased')

def process_description(description, method_choice):
    try:
        if method_choice == "Bert":
            #print("webapp bert")
            vector = get_distilbert_embeddings(description)
            #response = requests.post('http://annoy-bert:5000/reco_texte', json={'vector': vector,'method_choice': method_choice})
            print(method_choice)
        elif method_choice == "TFIDF":
            print(method_choice)
            vector = get_tfidf_embeddings(description)
            #response = requests.post('http://annoy-tfidf:5000/reco_texte', json={'vector': vector,'method_choice': method_choice})
        else:
            return 'Méthode non supportée'
        response = requests.post('http://annoy-db:5000/reco_texte', json={'vector': vector,'method_choice': method_choice})
        
        
        
        
        if response.status_code == 200:
            titles = response.json()
            return ", ".join(titles)
        else:
            return 'Erreur lors de la récupération des recommandations.'
    
    except Exception as e:
        print(f"Error in process_description: {e}")
        return 'Erreur lors du traitement de la description.'

iface_poster = gr.Interface(fn=process_image, inputs="image", outputs="plot",title="Système de recommandation de films",description="Faire glisser une affiche de film pour trouver un film ayant une affiche similaire")
iface_texte =  gr.Interface(
    fn=process_description,
    inputs=[
        gr.Textbox(placeholder="Saisissez une description de film..."),
        gr.Dropdown(["Bert", "TFIDF"], label="Select Method")
    ],
    outputs=gr.Textbox())

demo=gr.TabbedInterface([iface_poster,iface_texte],["Poster","Description"])
demo.launch(server_name="0.0.0.0")

'''
##### OLD VERSION #####

# Charger la base de données de films        
movies_metadata=pd.read_csv('movies_metadata.csv')
movies_metadata=movies_metadata['title','overview']

# Fonction pour obtenir les embeddings DistilBERT
def get_distilbert_embeddings(text):
    if pd.isna(text):
        print('Empty text received.')
        return np.zeros(model_2.config.dim).tolist()  # Convertir le tableau en liste
    
    print(f'Received text for embeddings: {text}')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: value.cpu() for key, value in inputs.items()}  # Transférer les données sur le CPU
    with torch.no_grad():
        outputs = model_2(**inputs)
    
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy().tolist()
    print(f'Generated embeddings: {embeddings}')
    return embeddings

def process_description(description):
    try:
        vector = get_distilbert_embeddings(description)
        print(f'Obtained embeddings vector: {vector}')

        response = requests.post('http://annoy-bert:5000/reco_texte', json={'vector': vector})
        print(f'Received response from NLPapi: {response.status_code}')

        if response.status_code == 200:
            indices = response.json()
            print(f'Received indices from NLPapi: {indices}')

            titles = get_movie_titles_from_indices(indices)
            print(f'Recommended titles: {titles}')

            return ", ".join(titles)
        else:
            print('Error in reco route. Status code:', response.status_code)
            return 'Error in reco route.'

    except Exception as e:
        print(f"Error in process_description: {e}")
        return 'Error in process_description.'''

'''def process_image(image):
    # Here you would extract the vector from the image for example using a mobile net
    # For the example I just generate a random vector
    # ça dégage vector = [random.random(), random.random()] #vecteur embedding de l'image pour laquelle on veut avoir reco grâce model.py (rzo neurones mobilenet)
    
    image_transfo=transform(image)
    vector=model(image_transfo.unsqueeze(0)).cpu().detach().numpy()#TO ARRAY!!!

    # Now we send the vector to the API
    # Replace 'annoy-db:5000' with your Flask server address if different (see docker-compose.yml)
    response = requests.post('http://10.1.2.14:5000', json={'vector': vector.tolist()})
    if response.status_code == 200:
        indices = response.json()

        # Retrieve paths for the indices
        paths = df[df['index'].isin(indices)]['path'].tolist()

        # Plot the images
        fig, axs = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5))
        for i, path in enumerate(paths):
            img = Image.open(path)
            axs[i].imshow(img)
            axs[i].axis('off')
        return fig
    else:
        return "Error in API request"

iface = gr.Interface(fn=process_image, inputs="image", outputs="plot")
iface.launch(server_name="0.0.0.0") # the server will be accessible externally under this address'''
