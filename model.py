#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:47:16 2023

@author: lobeto
"""
import torch
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
#import torchvision.models as models
import pandas as pd
from transformers import DistilBertModel, DistilBertTokenizerFast
import numpy as np
import gradio as gr
import torch
import requests

from annoy import AnnoyIndex
from flask import Flask, jsonify, request, send_from_directory
import io




mobilenet = mobilenet_v3_small(MobileNet_V3_Small_Weights)
model = mobilenet.features
model=torch.nn.Sequential(mobilenet.features, torch.nn.AdaptiveAvgPool2d(output_size=1),torch.nn.Flatten()) #extrait les features et on veut que ce soit 1x1 et non pas 7x7
model.eval()

'''x = torch.zeros(3, 3, 224,224)
y = model(x)
y.shape
'''

