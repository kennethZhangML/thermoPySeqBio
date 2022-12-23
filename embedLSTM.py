import torch 
import torch.nn as nn
import os 
import dsFile
import pandas as pd

class EnzymeStabilityRegressor(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.n_layers = 1
        self.embedding = nn.Embedding(input_channels - 1, 256)
        self.lstm1 = nn.LSTM(256, 128, self.n_layers, bidirectional = True, batch_first = True)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        outputs = self.embedding(x)
        outputs, (_h, _c) = self.lstm1(outputs)
        outputs = self.fc1(outputs[:, -1, :])
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        return outputs

def create_preds(test_dataset, test_df, protein_sequences, protein_lengths, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = model(test_dataset.to(device)).to(device)
    result = output.cpu().data.numpy()
    df_result = pd.DataFrame(result, columns=['tm'])
    pred_df = pd.DataFrame({"Thermostability Coefficient": df_result["tm"],
    "Protein Sequence": protein_sequences, "Protein Lengths": protein_lengths})
    return pred_df

