#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# DNA sequence embedding using transformers
# Path: embeddna.py
#
# A neural network to encode DNA sequences of variable length in a
# vector space. Expose the resulting vector space in a nearest
# neighbor database and measure the ability of the system to look up
# new sequences and find similar ones via the vector space
# embeddings. Do it in python with pytorch and use all GPUs. You might
# want to use a Transformer embedding approach.
#
# The input is a fasta file with DNA sequences. The output is a
# nearest neighbor database with the vector space embeddings of the
# sequences. The nearest neighbor database is a vector database.
#

import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import faiss

# Set the random seed manually for reproducibility.
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Now, let's define our dataset class. We'll use the BioPython library
# to read the DNA sequences from a FASTA file and convert them to
# numerical inputs. We'll pad the sequences to a fixed length and use
# one-hot encoding to represent each nucleotide. We'll also define a
# custom collate function to ensure that all sequences in a batch have
# the same length:
class DNADataset(Dataset):
    def __init__(self, file_path, max_length=1000):
        self.sequences = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequence = record.seq
            if len(sequence) > max_length:
                sequence = sequence[:max_length]
            self.sequences.append(str(sequence))

        self.max_length = max_length
        self.num_samples = len(self.sequences)

        self.char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        self.idx_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        sequence = self.one_hot_encode(sequence)
        return sequence

    def one_hot_encode(self, sequence):
        seq_len = len(sequence)
        encoding = np.zeros((self.max_length, 4), dtype=np.float32)

        for i in range(seq_len):
            char = sequence[i]
            if char in self.char_to_idx:
                encoding[i, self.char_to_idx[char]] = 1.0
            else:
                encoding[i, :] = 0.25  # assign equal probability to each nucleotide for unknown characters

        if seq_len < self.max_length:
            encoding[seq_len:, :] = 0.25  # pad with equal probability for unknown nucleotides

        return encoding

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)
        max_len = max([seq.shape[0] for seq in batch])
        input_data = np.zeros((batch_size, max_len, 4), dtype=np.float32)

        for i, seq in enumerate(batch):
            input_data[i, :seq.shape[0], :] = seq

        return torch.tensor(input_data)


# Next, let's define the neural network model. We'll use a
# Transformer-based architecture to encode the DNA sequences into a
# vector space. We'll also add a linear layer to reduce the
# dimensionality of the embeddings:
class DNAEncoder(nn.Module):
    def __init__(self, max_length=1000, embedding_dim=64, num_heads=8, num_layers=3):
        super(DNAEncoder, self).__init__()

        self.max_length = max_length
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(4, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_length)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads
            ),
            num_layers=num_layers
        )
        self.fc = nn.Linear(embedding_dim, 32)

    def forward(self, x):
        # x has shape (batch_size, max_length, 4)
        x = x.permute(1, 0, 2)  # shape (max_length, batch_size, 4)
        x = self.embedding(x) * np.sqrt(self.embedding_dim)  # shape (max_length, batch_size, embedding_dim)
        x = self.positional_encoding(x)  # shape (max_length, batch_size, embedding_dim)
        x = self.transformer_encoder(x)  # shape (max_length, batch_size, embedding_dim)
        x = x.mean(dim=0)  # shape (batch_size, embedding_dim)
        x = self.fc(x)  # shape (batch_size, 32)

        return x

# We've also defined a PositionalEncoding class to add positional
# encoding to the input embeddings:
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# Now, let's define a function to train the neural network using the provided DNA sequences:
def train(model, dataset, num_epochs=10, batch_size=32, lr=1e-4, device=torch.device('cuda')):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=DNADataset.collate_fn)

    model = model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()

            embeddings = model(batch)
            loss = criterion(embeddings, embeddings)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= num_batches

        print('Epoch {}, loss={:.6f}'.format(epoch + 1, epoch_loss))

# We'll use mean squared error (MSE) as the loss function. Note that
# we're also computing the loss between the embeddings and themselves,
# which is a form of unsupervised learning. This will encourage the
# neural network to generate embeddings that are similar for DNA
# sequences that share common characteristics.

# Next, let's define a function to generate embeddings for a given set
# of DNA sequences using our trained neural network:

def generate_embeddings(model, dataset, batch_size=32, device=torch.device('cuda')):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=DNADataset.collate_fn)
    embeddings = np.zeros((len(dataset), 32), dtype=np.float32)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)

            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))

            embeddings[start_idx:end_idx] = model(batch).cpu().numpy()

    return embeddings

def nearest_neighbors(query, index, k=10):
    distances, indices = index.search(query, k)
    return indices.tolist()[0]
