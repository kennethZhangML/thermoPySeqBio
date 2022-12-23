import dsFile
import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import re
import py3Dmol
import stmol
import plotly.express as px
import torch 
import torch.nn as nn
import embedLSTM as el
import os
from torchviz import make_dot
from torchsummary import summary

st.set_page_config(layout = "wide")

st.sidebar.image("logo.png")
st.sidebar.title('Navigation')

st.sidebar.success("Thermostability refers to the ability of an enzyme to maintain its activity and structural integrity under high temperatures. Enzymes are proteins that catalyze chemical reactions in living cells, and they are typically optimally active within a certain temperature range. However, some enzymes are able to retain their activity and structure at higher temperatures, making them thermostable.")
st.sidebar.success("This demo application attempts to demonstrate the effectiveness of utilizing a LSTM-Embed Neural Network Regressor + Gradient Boosting to accurately predict the stability of different enzymes given their protein sequence in different pH environments.")

st.sidebar.write("Developed and Engineered by Kenneth Zhang © Copyright 2022-2023")

st.markdown("<h1 style='text-align: center; color: white;'>ThermoPySeq Demo App</h1>", unsafe_allow_html=True)
st.write("--------------------------------------------------")

col1, col2 = st.columns(2)

def wrangle_data():
    train_df = dsFile.load_fixed_dataframe()
    train_df = dsFile.return_amino_acid_df(train_df)
    train_df["protein_length"] = train_df["protein_sequence"].apply(lambda x : len(x))

    test_df = pd.read_csv("data/test.csv")
    test_df["protein_length"] = test_df["protein_sequence"].apply(lambda x : len(x))
    protein_lengths = test_df["protein_length"]
    protein_sequences = test_df["protein_sequence"]

    pH_list = list(train_df['protein_sequence'].str.split('').explode('protein_sequence').unique())
    pH_list.remove('')
    pH_map = {pH: i + 1 for i, pH in enumerate(pH_list)}
    pH_map[None] = 0

    test_df = pd.read_csv("data/test.csv")
    test_df = dsFile.clean_train_df(test_df)
    test_df = dsFile.encode(test_df, pH_map = pH_map)

    test_dataset = torch.tensor((test_df.values), dtype=torch.long)
    return train_df, test_df, pH_map, test_df, test_dataset, protein_sequences, protein_lengths
        

PATH = "thermoPySeqBio/state_dict"
model = el.EnzymeStabilityRegressor(222)
model.load_state_dict(torch.load(os.path.join(PATH, "lstmEmbed.pth")))

train_df, test_df, pH_map, test_df, test_dataset, protein_sequences, protein_lengths = wrangle_data()
pred_df = el.create_preds(test_dataset, test_df, protein_sequences, protein_lengths, model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with col1:
    st.subheader("pH vs Thermostability Coefficient")
    heat1 = px.imshow(train_df[["pH", "tm", "protein_length"]].corr(), text_auto = True)
    st.write(heat1)
    st.write("This shows that the correlation between the protein lengths, tm (thermostability coefficient), and pH have no strong correlation, implying that the breakdown of the sequences themselves can provide a more influential insight into how thermostability is determined and predicted.")
    
    st.write("--------------------------------------------------")
    st.markdown("<h2 style='text-align: center; color: white;'>LSTM-Embedding Model</h2>", unsafe_allow_html=True)

    code = '''class EnzymeStabilityRegressor(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        self.n_layers = 1
        self.embedding = nn.Embedding(input_channels - 1, 256)
        self.lstm1 = nn.LSTM(256, 128, self.n_layers, 
        bidirectional = True, batch_first = True)
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
        return outputs'''
    
    st.code(code, language = "python")


def make_input_pred(input_sequence, pH_map):
    input_list = [input_sequence]
    input_df = pd.DataFrame()
    input_df["protein_sequence"] = input_list
    input_seq_df = pd.DataFrame(input_df["protein_sequence"].apply(list).tolist())
    input_seq_df = input_seq_df.replace(pH_map)
    input_df = input_df.join(input_seq_df)
    input_df = input_df.drop(columns=['protein_sequence'])
    return input_df

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False


with col2:
    st.subheader("Distribution of pH's of Enzymes")
    hist1 = px.histogram(train_df, x = "tm", color = "pH")
    st.write(hist1)
    st.write("Visualizing the pH of enzymes shows how most enzymes are stable in an environment with pH = 7.0, which is a assumed since most enzymes are assumed to possess such a characteristic. Therefore, more in-depth breakdown of the amino acid sequences is required in order for a model to be developed.")

    st.write("--------------------------------------------------")
    st.markdown("<h2 style='text-align: center; color: white;'>Generate Predictions</h2>", unsafe_allow_html=True)
    
    if st.button("Make Predictions on Sample Test Dataset"):
        st.write(pred_df)
    else: 
        st.write("Or, paste a Protein Sequence here")

        input_sequence = st.text_input("Protein Sequence: ")

        if input_sequence:
            input_df = make_input_pred(input_sequence, pH_map)
            input_test = torch.tensor((input_df.values), dtype = torch.long)
            st.write(model.load_state_dict(torch.load(os.path.join(PATH, "lstmEmbed.pth"))))
            output = model(input_test.to(device)).to(device)
            result = output.cpu().data.numpy()
            
            st.write("Predicted Thermostability Coefficient: ", result[0][0])
            st.write("Breakdown of Inputted Protein Sequence: ", input_df)
            st.write("*The breakdown of the extracted features from the protein sequence are in the dataframe generated above and can be downloaded.")
            
            input_csv = convert_df(input_df)
            
            st.download_button(
                label="Download Breakdown as CSV",
                data = input_csv,
                file_name='sequence_Feature_breakdown.csv'
            )

        else:
            st.write("")
            

st.write("--------------------------------------------------")
st.markdown("<h2 style='text-align: center; color: white;'>Enzyme Dataset Summary Stats</h2>", unsafe_allow_html=True)
st.write(train_df.describe())













