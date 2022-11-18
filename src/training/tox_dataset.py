import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class ToxData(Dataset):

    def __init__(self, sample_index, molecule_file):

        #sample_index = pd.read_csv(sample_index_file, sep=",", header=0)
        molecule_df = pd.read_hdf(molecule_file, key="df")
        keys = sample_index.index.tolist()

        self.sample_index = sample_index
        self.molecule_df = molecule_df
        self.keys = keys
        print(len(keys))
        self.n_classes = sample_index.loc[:, "NR.AhR": "SR.p53"].shape[1]
        self.missing = []
        if len(keys) > 6000:
            print(keys[5284])
            print(self.sample_index.loc[5284])

    def __len__(self):
        return len(self.sample_index)


    def __getitem__(self, idx):
        sample_key = self.keys[idx]
        mol = self.molecule_df.loc[sample_key].values
        #print(f"Error not in molecules{idx}")
        labels = self.sample_index.loc[sample_key, "NR.AhR": "SR.p53"].fillna(99)
        #print(f"Error not in labels{idx}")
        labels = np.array(labels)

        return mol, labels


    @property
    def num_classes(self):
        return self.n_classes

class PubChemData(Dataset):

    def __init__(self, sample_index, molecule_file):

        #sample_index = pd.read_csv(sample_index_file, sep=",", header=0)
        molecule_df = pd.read_hdf(molecule_file, key="df")
        keys = sample_index.index.tolist()

        self.sample_index = sample_index
        self.molecule_df = molecule_df
        self.keys = keys
        print(len(keys))
        self.n_classes = 1
        print(self.n_classes)
        self.missing = []


    def __len__(self):
        return len(self.sample_index)


    def __getitem__(self, idx):
        sample_key = self.keys[idx]
        mol = self.molecule_df.loc[sample_key].values
        #print(f"Error not in molecules{idx}")
        labels = self.sample_index.loc[sample_key, "Activity"]
        #print(f"Error not in labels{idx}")
        labels = np.array(labels)

        return (mol, labels)


    @property
    def num_classes(self):
        return self.n_classes
