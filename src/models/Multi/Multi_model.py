import torch
import torch.nn as nn
import numpy as np
import pickle
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class Multi(nn.Module):
    def __init__(self, args, autoint_data, cnn_fm_data, deepconn_data, dropout=0.5):
        
        super().__init__()
        self.Autoint = torch.load('/data/ephemeral/home/Book_Rating_Prediction/saved_models/20231220_200029_AutoInt_model.pt')
        self.CNN_FM = torch.load('/data/ephemeral/home/Book_Rating_Prediction/saved_models/20231221_041454_CNN_FM_model.pt')
        self.DeepCoNN = torch.load('/data/ephemeral/home/code/src/models/Multi/20231220_160138_DeepCoNN_model.pt')
        
        layers = list()
        self.input_layer = self.data.shape[1]
        self.emmbed_dims = [256,64,1]
        input_dim = self.input_layer
        
        for embed_dim in self.embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
        self.mlp = torch.nn.Sequential(*layers)


    def forward(self, autoint_data, cnn_fm_data, deepconn_data):

        _ , autoint_out = self.Autoint(autoint_data)
        _ , cnn_fm_out = self.CNN_FM(cnn_fm_data)
        _ , deepconn_out = self.DeepCoNN(deepconn_data)
        data = pd.concat([autoint_out, cnn_fm_out, deepconn_out], axis=1)
        return self.mlp(data)
