import torch

autoint_model = torch.load("/data/ephemeral/home/Book_Rating_Prediction/saved_models/20231220_200029_AutoInt_model.pt")
autoint_model.eval()

cnn_model = torch.load("/data/ephemeral/home/Book_Rating_Prediction/saved_models/20231221_041454_CNN_FM_model.pt")
cnn_model.eval()

text_model = torch.load("/data/ephemeral/home/Book_Rating_Prediction/saved_models/20231221_052629_DeepCoNN_model.pt")
text_model.eval()