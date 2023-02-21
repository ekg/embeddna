import torch
import torch.nn as nn
import torch.optim as optim
from Bio import SeqIO
import gzip
import numpy as np
from annoy import AnnoyIndex

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size).to("cuda")
        self.pos_embedding = nn.Embedding(1000, embedding_size).to("cuda")
        self.transformer = nn.Transformer(embedding_size, num_heads, num_layers, num_layers, hidden_size, dropout).to("cuda")
        self.linear = nn.Linear(embedding_size, vocab_size).to("cuda")
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
    
    def forward(self, x):
        x = self.embedding(x)
        x = x * torch.sqrt(torch.tensor(self.embedding_size, dtype=x.dtype)).to(x.device)
        seq_len = x.size(1)
        pos = torch.arange(seq_len).unsqueeze(0).repeat(x.size(0), 1).to(x.device)
        x = x + self.pos_embedding(pos)
        x = x.transpose(0, 1)
        output = self.transformer(x, x)
        output = self.linear(output)
        return output

# Define the hyperparameters
batch_size = 768
max_len = 512
vocab_size = 5
embedding_size = 128
hidden_size = 256
num_layers = 6
num_heads = 8
dropout = 0.1
lr = 0.001
num_epochs = 1
k = 5

# Load the input DNA sequence from a fasta file
sequence = ""
with gzip.open("input.fa.gz", "rt") as handle:
    record = next(SeqIO.parse(handle, "fasta"))
    sequence += str(record.seq) # TODO add record delimiter

print("sequence length", len(sequence))

# Map non-ATGC characters to a common token
char_to_token = {"A": 0, "T": 1, "G": 2, "C": 3}
tokens = []
for base in sequence:
    if base in char_to_token:
        tokens.append(char_to_token[base])
    else:
        tokens.append(4)
x = torch.tensor(tokens, dtype=torch.long).to("cuda")
print(x.device)

# Create the data loader
dataset = torch.utils.data.TensorDataset(x[:-1], x[1:])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# Initialize the model and optimizer
model = TransformerModel(vocab_size, embedding_size, hidden_size, num_layers, num_heads, dropout)
optimizer = optim.Adam(model.parameters(), lr=lr)
print("about to epoch")

# Train the model
for epoch in range(num_epochs):
    total_loss = 0
    for i, (input, target) in enumerate(dataloader):
        if input.size(0) < batch_size:
            continue
        input = input.view(batch_size, -1).t().to(x.device)
        #print("input", input)
        target = target.view(batch_size, -1).t().to(x.device)
        output = model(input)
        #print("output", output)
        loss = nn.functional.cross_entropy(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        #print("epoch", epoch, "loss", total_loss)
    print("Epoch {} loss: {:.4f}".format(epoch, total_loss))

kmer_stride = 100
# Generate embeddings for all 500bp segments of the input sequence
print("making embeddings for all 500bp segments by stride", kmer_stride)
k_mers = [sequence[i:i+500] for i in range(0, len(sequence)-500+1, kmer_stride)]
embeddings = []
with torch.no_grad():
    for i in range(0, len(sequence)-500+1, kmer_stride):
        x = torch.tensor([tokens[j] for j in range(i, i+500)], dtype=torch.long).unsqueeze(0).to(x.device)
        embedding = model.embedding(x) * torch.sqrt(torch.tensor(embedding_size, dtype=x.dtype)).to(x.device)
        pos = torch.arange(500).unsqueeze(0).to(x.device)
        embedding = embedding + model.pos_embedding(pos)
        #print("before transpose", embedding)
        #print(embedding.shape)
        embedding = embedding.transpose(0, 2)
        #print("after transpose", embedding)
        #print(embedding.shape)
        embeddings.append(embedding.mean(dim=2).mean(dim=1).cpu().numpy())
        #print("embedding", embeddings[-1], "embedding length", len(embeddings[-1]), "embedding shape", embeddings[-1].shape)

# Build the index
print("building embedding index")
index = AnnoyIndex(embedding_size, metric='angular')
for i in range(len(embeddings)):
    index.add_item(i, embeddings[i].reshape(embedding_size))

index.build(10)

# Query the index for nearest neighbors
nearest_neighbors = {}
for i, k_mer in enumerate(k_mers):
    query_embedding = embeddings[i].reshape(embedding_size)
    nn_indices = index.get_nns_by_vector(query_embedding, k+1, include_distances=False)
    nn_k_mers = [k_mers[j] for j in nn_indices if j != i][:k]
    nearest_neighbors[k_mer] = nn_k_mers

# Print the results
for k_mer, nn_k_mers in nearest_neighbors.items():
    print("kmer", k_mer)
    for nn_k_mer in nn_k_mers:
        print("nn_kmer", nn_k_mer)
    print('')
