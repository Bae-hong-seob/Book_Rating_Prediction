import os
import gdown
import streamlit as st
import yaml
import sys
from src.models import *


with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 구글 드라이브를 이용한 모델 다운로드
def download_model_file(url):
    output = "model.json"
    gdown.download(url, output, quiet=False)

@st.cache_resource
def load_model(args):
    '''
    Return:
        model: 구글 드라이브에서 가져온 모델 return 
    '''
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("model.json"):
        download_model_file(config['model_path'])
    
    model_path = 'model.json'
    model = CatBoost(args)
    model.load_model(model_path)

    return model

def get_prediction(model, data) -> float:
    '''
    Args:
        model : 학습된 모델
        data : 사용자 정보 

    Return:
        rating
    '''

    return model(data)