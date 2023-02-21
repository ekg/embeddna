# DNA embeddings from scratch

This small experiment develops embeddings for subsequences of many genomes and compares them to each other with a k-nearest-neighbor vector comparison library.

You can run it with `python embed.py`.

It needs CUDA, PyTorch, Biopython, Annoy to work, and a GPU to work at a reasonable speed.