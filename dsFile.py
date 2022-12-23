import streamlit as st 
import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import re
import py3Dmol
import stmol
import plotly.express as px

def load_fixed_dataframe(original = "data/train.csv", updated = "data/train_updates_20220929.csv", was_fixed = False):
    def fix_tm_ph(row, update_map):
        update_vals = update_map.get(row["seq_id"], None)
        if update_vals is not None:
            row["tm"] = update_vals["tm"] #processing thermochemical stability metric (Spearman Correlation Coefficient)
            row["pH"] = update_vals["pH"] #iterating through pH values and re-evaluating for precision
        return row
    
    df = pd.read_csv(original)
    updated_df = pd.read_csv(updated)
    seq_id_phtm = updated_df[~pd.isna(updated_df["pH"])].groupby("seq_id")[["pH", "tm"]].first().to_dict("index")

    bad_seqs = updated_df[pd.isna(updated_df["pH"])]["seq_id"].to_list()

    df = df[~df["seq_id"].isin(bad_seqs)].reset_index(drop = True)
    df = df.apply(lambda x : fix_tm_ph(x, seq_id_phtm), axis = 1)

    if was_fixed: df["was_fixed"] = df["seq_id"].isin(bad_seqs + list(seq_id_phtm.keys()))
    return df 


def return_amino_acid_df(df):    
    search_amino=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    for amino_acid in search_amino:
         df[amino_acid] = df['protein_sequence'].str.count(amino_acid,re.I)
    return df

def clean_train_df(df):
    df = df.drop(["seq_id", "data_source"], axis = 1)
    df = df.loc[~df['pH'].isna()]
    df = df.loc[df['pH'] <= 14]
    df = df.loc[df['protein_sequence'].str.len() <= 221]
    df = df.reset_index(drop = True)
    return df

def encode(df, pH_map):
    sequences_df = pd.DataFrame(df['protein_sequence'].apply(list).tolist())
    sequences_df = sequences_df.replace(pH_map)
    df = df.join(sequences_df)
    df = df.drop(columns=['protein_sequence'])
    return df










